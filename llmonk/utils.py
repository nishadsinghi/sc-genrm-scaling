from pathlib import Path
import yaml
import re
from pydra import Config, REQUIRED
import signal
import os


def load_yaml(path: Path):
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.CLoader)

    return data


def save_yaml(path: Path, data, sort_keys=True):
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=sort_keys)


def append_yaml(path, data, check_if_prompt_matches_existing=False):
    # if path doesn't exist, create the yaml
    if not os.path.exists(path):
        print('yaml file didnt exist, creating and saving')
        save_yaml(path, data)

    # if path exists, load the yaml, append to it, and save it
    else:
        print('yaml file already exists, appending')
        try:
            existing_data = load_yaml(path)
        except:
            save_yaml(path, data)
            return

        # if file exists but is None (or empty)
        if existing_data is None:
            save_yaml(path, data)
        
        # file exists and is not None
        else:
            if not data['prompt'] == existing_data['prompt']:
                print("path = ", path)
                print("ERROR: PROMPTS DO NOT MATCH WHILE APPENDING")
                print('existing prompt: ', existing_data['prompt'])
                print('\n\n\n\n')
                print('new prompt: ', data['prompt'])
                print('======================')
                if check_if_prompt_matches_existing:
                    raise Exception("ERROR: PROMPTS DO NOT MATCH WHILE APPENDING")
            for key in ('verifications', 'samples'):
                if key not in data:
                    continue
                # Ensure the key exists in the loaded data
                if key in existing_data:
                    existing_data[key].extend(data.get(key, []))
                else:
                    # If key doesn't exist, initialize it with the data
                    existing_data[key] = data.get(key, [])
        
            # Save the updated data back to the file
            save_yaml(path, existing_data)


def dataclass_to_dict(obj) -> dict:
    """
    Converts a dataclass to a dictionary. Will recurse through
    lists, dicts, and nested dataclasses.
    """

    if hasattr(obj, "__dataclass_fields__"):
        return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def get_theorem_name(id):
    """
    Map the huggingface datasets id to the theorem name in the Lean repository
    """
    if "math" in id or "algebra" in id or "numbertheory" in id or "induction" in id:
        return id
    elif "imo_1968_5_1" in id:
        return "imo_1968_p5_1"
    elif "imo_2007_6" in id:
        return "imosl_2007_algebra_p6"
    elif "aime" in id:
        return "aime_" + id.split("aime")[1].split("_")[0] + "_p" + id.split("_")[-1]
    else:
        return "_".join(id.split("_")[:-1]) + "_p" + id.split("_")[-1]


def is_valid_python(snippet):
    try:
        compile(snippet, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_first_code(output_string: str):
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code = code_match.group(1).strip()

        # sometimes the block of code is ```python ... ``` instead of ``` ... ```
        # in this case strip the python out

        if code.startswith("python"):
            code = code[len("python") :].strip()

        return code

    if is_valid_python(trimmed):
        return trimmed

    return None


class GenerateScriptConfig(Config):
    model = REQUIRED
    save_dir = REQUIRED
    dataset = REQUIRED
    apply_chat_template = REQUIRED

    num_workers = None
    gpus = None
    vllm_args = None
    vllm_port = 8000

    seed = 0
    limit = None
    offset = None
    stride = None

    save_every_n_samples = 4
    num_few_shot = 2
    max_tokens = 1024
    stop_strings = []
    num_samples = 2
    batch_size = 2
    top_p = 0.95
    temperature = 0.6

    rerun = False # if this is true, then if a file already exists, it will still generate samples for that file

    def finalize(self):
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

    zero_shot = False


class VerifyScriptConfig(Config):
    model = REQUIRED
    samples_dir = REQUIRED
    output_dir = REQUIRED
    num_verifications = REQUIRED
    verification_template = REQUIRED
    num_problems = REQUIRED
    num_solutions = REQUIRED

    add_generation_prompt = True
    num_workers = None
    gpus = None
    vllm_args = None
    vllm_port = 8000

    seed = 0
    limit = None
    offset = None
    stride = None

    max_tokens = 4096
    stop_strings = []
    batch_size = 2
    top_p = 0.95
    top_k = 40
    temperature = 0.7

    skip_special_tokens = True
    problem_offset = (
        0  # if this is set to K, the model process starts from Kth problem onwards (skip first K problems)
    )
    problem_stride = 1
    solution_offset = 0

    # skip_already_processed_files = True
    logprobs = None
    force_generate = False

    # for reasoning models
    remove_think_tag_from_solution = True

    def finalize(self):
        self.save_dir = Path(self.output_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)


class EvaluateScriptConfig(Config):
    samples_dir: Path = REQUIRED
    save_dir: Path = REQUIRED

    num_workers: int = 1

    offset: int = 0
    stride: int = 1
    limit: int = 100_000

    def finalize(self):
        self.samples_dir = Path(self.samples_dir)
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)


class Timeout():
  """Timeout class using ALARM signal"""
  class Timeout(Exception): pass

  def __init__(self, sec):
    self.sec = sec

  def __enter__(self):
    signal.signal(signal.SIGALRM, self.raise_timeout)
    signal.alarm(self.sec)

  def __exit__(self, *args):
    signal.alarm(0) # disable alarm

  def raise_timeout(self, *args):
    raise Timeout.Timeout()
