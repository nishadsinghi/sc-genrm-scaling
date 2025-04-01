import multiprocessing
import os
import pickle
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import pydra
import requests
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from llmonk.generate.prompts import MATH_COT_PROMPT
from llmonk.generate.vllm_utils import vllm_manager
from llmonk.utils import (
    GenerateScriptConfig,
    VerifyScriptConfig,
    append_yaml,
    save_yaml,
)


class BatchLoader:
    def __init__(self, num_problems, num_solutions, num_verifications, batch_size, problem_offset, solution_offset):
        self.num_problems = num_problems
        self.num_solutions = num_solutions
        self.batch_size = batch_size
        self.problem_offset = problem_offset
        self.solution_offset = solution_offset

    def get_ids(self, i):
        problem_id = i // (self.num_solutions) + self.problem_offset
        c = i % (self.num_solutions)
        solution_id = c % self.num_solutions + self.solution_offset

        print(f"problem_id: {problem_id}, solution_id: {solution_id}")
        return (problem_id, solution_id)

    def __call__(self, idx):
        """
        for a given idx, it should return a list of length batch_size
        every element in the list is a tuple containing (problem_id, solution_id, verification_id)
        """
        return [self.get_ids(i) for i in range(idx * self.batch_size, (idx + 1) * self.batch_size)]


def prepare_prompt(
    samples_dir, problem_id, solution_id, tokenizer, prompt_template_path, remove_think_tag_from_solution
):
    file_path = os.path.join(samples_dir, f"{problem_id}.yaml")
    if os.path.exists(file_path) == False:
        print("FILE DOESNT EXIST: ", file_path)
        return None

    # Load the YAML file
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)

    # Extract the question and the correct solution from the sample
    # print('Extract the question and the correct solution from the sample')
    if "question" in data:
        question = data["question"]
    elif "problem" in data:
        question = data["problem"]
    else:
        raise Exception
    try:
        solution = data["samples"][solution_id]
        if remove_think_tag_from_solution:
            solution = solution.split('</think>\n\n')[-1]

    except Exception as e:
        print(f"ERROR filename: {file_path}, solution_id: {solution_id}")
        raise Exception(e)


    with open(prompt_template_path, "r") as file:
        prompt_template = file.read()

    if "Question: {}" in prompt_template:
        prompt = prompt_template.replace("Question: {}", f"Question: {question}")
        prompt = prompt.replace("Solution: {}", f"Solution: {solution}")

    else:
        prompt = prompt_template.format(question, solution)

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    return (formatted_prompt, problem_id, solution_id)


def prepare_save_path(output_dir, problem_id, solution_id, verification_id):
    save_path = os.path.join(output_dir, f"problem_{problem_id}", f"solution_{solution_id}.yaml")
    directory = os.path.dirname(save_path)
    Path(directory).mkdir(parents=True, exist_ok=True)
    return save_path


def extract_P_yes(logprobs, tokenizer):
    def extract_p(final_verdict_position):
        p_yes, p_no = -1, -1
        logprobs_final_verdict = logprobs[final_verdict_position]
        for key, value in logprobs_final_verdict.items():
            if tokenizer.decode(int(key)) == " Yes" or tokenizer.decode(int(key)) == "Yes":
                p_yes = np.exp(value)
            if tokenizer.decode(int(key)) == " No" or tokenizer.decode(int(key)) == "No":
                p_no = np.exp(value)
        return p_yes, p_no

    # in the case of a direct verifier (which just outputs yes or no without verification CoT), check the first position
    p_yes, p_no = extract_p(0)
    if p_yes != -1 or p_no != -1:
        return p_yes, p_no

    # usually the yes/ no verification is found at this position
    final_verdict_position = -2
    p_yes, p_no = extract_p(final_verdict_position)
    if p_yes == -1 and p_no == -1:
        # sometimes it's found at this position (if the last position has a token like "**")
        final_verdict_position = -3
        p_yes, p_no = extract_p(final_verdict_position)

    return p_yes, p_no


def run_inference(batch_idx, config, batch_loader, tokenizer, prompt_template_path):
    batch = batch_loader(batch_idx)  # list containing tuples of (problem_id, solution_id, verification_id)
    
    prompts = [
        prepare_prompt(
            config.samples_dir,
            i[0],
            i[1],
            tokenizer,
            prompt_template_path,
            remove_think_tag_from_solution=config.remove_think_tag_from_solution
        )
        for i in batch
    ]

    url = f"http://localhost:{config.vllm_port}/generate"
    batch_size = config.batch_size

    def prepare_yaml(output):
        try:
            p_yes, p_no = extract_P_yes(output['logprobs'], tokenizer)
        except:
            p_yes, p_no = -1, -1

        return {
            "prompt": output["prompt"],
            "verifications": [
                {
                    "verification": output["verification"],
                    "p(yes)": float(p_yes),
                    "p(no)": float(p_no),
                }
            ],
        }

    for instance in prompts:    # every instance is a solution for which we are generating verifications
        samples = []
        save_paths = []
        verification_ids = []
        verification_id = 0

        num_samples = config.num_verifications
        save_path = prepare_save_path(config.output_dir, instance[1], instance[2], verification_id)
        
        # check if sufficient number of verifications have been generated at this path
        # Check if the file exists
        if os.path.exists(save_path):
            # Load the YAML file
            try:
                with open(save_path, "r") as file:
                    data = yaml.safe_load(file)
                    yaml_loaded = True
            except:  # if the yaml file cannot be loaded
                yaml_loaded = False

            if yaml_loaded:
                # Check if the 'verifications' field exists and is a list
                if data is not None and "verifications" in data and isinstance(data["verifications"], list):
                    # Count the length of the 'verifications' field
                    verification_count = len(data["verifications"])

                    if verification_count >= config.num_verifications:
                        print("skipping file because it has enough verifications: ", save_path)
                        continue

                    elif verification_count > 0:
                        num_samples = config.num_verifications - verification_count
                        print(
                            f"file already has {verification_count} verifications, now generating {num_samples} more"
                        )

        # compute the number of batches ('calls' to the server)
        num_calls = num_samples // batch_size   
        if num_samples % batch_size != 0:
            num_calls += 1

        for _ in tqdm(range(num_calls)):

            body = {
                "prompt": instance[0],
                "max_tokens": config.max_tokens,
                "n": batch_size,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "stop": config.stop_strings,
                "logprobs": config.logprobs,
            }

            response = requests.post(url, json=body)
            respj = response.json()
            print(respj.keys())
            texts = respj["text"]
            logprobs = respj["logprobs"]

            for j in range(batch_size):
                text_j, logprobs_j = texts[j], logprobs[j]
                save_paths.append(prepare_save_path(config.output_dir, instance[1], instance[2], verification_id))
                verification_ids.append(verification_id)
                verification_id += 1
                samples.append({"prompt": instance[0], "verification": text_j, "logprobs": logprobs_j})

        for path, sample in zip(save_paths, samples):
            append_yaml(path, prepare_yaml(sample), check_if_prompt_matches_existing=True)


@pydra.main(VerifyScriptConfig)
def main(config: VerifyScriptConfig):

    random.seed(config.seed)

    if config.budget_forcing:
        config.batch_size = 1

    num_problems = config.num_problems
    num_solutions = config.num_solutions
    prompt_template_path = config.verification_template

    print("Num Problems: ", num_problems)
    print("Num Samples: ", num_solutions)

    assert num_problems * num_solutions % config.batch_size == 0
    num_batches = num_problems * num_solutions // config.batch_size
    print("Num Batches: ", num_batches)
    
    batch_loader = BatchLoader(
        num_problems,
        num_solutions,
        config.num_verifications,
        config.batch_size,
        config.problem_offset,
        config.solution_offset,
    )
    
    batch_idxs = list(range(num_batches))
    print(batch_idxs)

    tokenizer = AutoTokenizer.from_pretrained(config.model)

    with vllm_manager(config) as vllm_port:
        config.vllm_port = vllm_port

        go_func = partial(
            run_inference,
            config=config,
            batch_loader=batch_loader,
            tokenizer=tokenizer,
            prompt_template_path=prompt_template_path,
        )

        if config.num_workers not in [0, None]:
            with multiprocessing.Pool(config.num_workers) as pool:
                predictions = list(
                    tqdm(
                        pool.imap_unordered(go_func, batch_idxs),
                        total=len(batch_idxs),
                    )
                )
        else:
            predictions = []
            for item in tqdm(batch_idxs):
                predictions.append(go_func(item))


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("time taken = ", end_time - start_time)