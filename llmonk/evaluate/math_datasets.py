'''
python llmonk/evaluate/math_datasets.py samples_dir=$SAVE_DIR/math_samples save_dir=$SAVE_DIR/math_eval dset=math num_workers=32
'''

from pathlib import Path
from tqdm import tqdm
import multiprocessing
import pydra
from copy import deepcopy
import re
from lm_eval.tasks.minerva_math.utils import (
    last_boxed_only_string,
    normalize_final_answer,
    get_unnormalized_answer,
    get_unnormalized_answer_GPT,
    remove_boxed,
    is_equiv,
)

from llmonk.utils import load_yaml, save_yaml, EvaluateScriptConfig
from eval_utils import is_correct


class ScriptConfig(EvaluateScriptConfig):
    dset: str = "gsm8k"


def process_sample(config: ScriptConfig):
    if config.save_path.exists():
        return

    result = load_yaml(config.sample_path)
    corrects = []

    for sample in result["samples"]:
        try:
            correct = is_correct(sample, result["gt_answer"], config.dset)
        except:
            print("SKIPPING SAMPLE PATH BECAUSE CANNOT ASSESS CORRECTNESS: ", config.sample_path)
            return 
            # raise Exception
        corrects.append(correct)

    result["is_corrects"] = corrects

    save_yaml(config.save_path, result)


def get_tasks(config):
    sample_paths = Path(config.samples_dir).glob("*.yaml")

    tasks = []
    for sample_path in tqdm(sample_paths, desc="Loading generations"):
        save_path = config.save_dir / sample_path.name

        task_config = deepcopy(config)
        task_config.sample_path = sample_path
        task_config.save_path = save_path

        tasks.append(task_config)

    return tasks


@pydra.main(base=ScriptConfig)
def main(config: ScriptConfig):

    tasks = get_tasks(config)
    tasks = sorted(
        tasks, key=lambda x: x.save_path
    )  # sort so the same offset references the same tasks across machines
    tasks = tasks[config.offset : config.limit : config.stride]

    print(f"Evaling on {len(tasks)} problems.")

    if config.num_workers not in [0, None]:
        with multiprocessing.Pool(processes=config.num_workers) as pool:
            _ = list(tqdm(pool.map(process_sample, tasks), total=len(tasks)))
    else:
        for task in tqdm(tasks):
            process_sample(task)


if __name__ == "__main__":
    main()