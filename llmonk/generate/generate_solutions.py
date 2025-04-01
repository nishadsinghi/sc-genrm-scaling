import multiprocessing
import random
import time
from functools import partial
import numpy as np

import pydra
import requests
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from llmonk.generate.prompts import MATH_COT_PROMPT, GPQA_ZERO_SHOT_PROMPT, QWEN_2p5_MATH_PROMPT
from llmonk.generate.vllm_utils import vllm_manager
from llmonk.utils import GenerateScriptConfig, append_yaml, load_yaml, save_yaml


def crop_samples(samples):
    """Process each sample string to crop after 'I hope it is correct.'"""
    processed = []
    for sample in samples:
        if "I hope it is correct." in sample:
            # Split on the phrase and take only the first part plus the phrase itself
            processed.append(sample.split("I hope it is correct.")[0] + "I hope it is correct.")
        else:
            processed.append(sample)
    return processed


def generate_multiple_choice_answers(item):
    random.seed(42)
    """Generate multiple choice string and correct answer letter."""
    answers = [
        item["Correct Answer"],
        item["Incorrect Answer 1"],
        item["Incorrect Answer 2"],
        item["Incorrect Answer 3"],
    ]
    
    random.shuffle(answers)

    options = ["A", "B", "C", "D"]
    options_to_answers = {letter: answer for letter, answer in zip(options, answers)}

    multiple_choice_string = ", ".join(f"{letter}) {options_to_answers[letter]}" for letter in options)
    correct_answer_letter = next(
        letter for letter, answer in options_to_answers.items() if answer == item["Correct Answer"]
    )

    return multiple_choice_string, correct_answer_letter


def save_samples(item, config, samples, prompt, outpath):
    samples = crop_samples(samples)
    if "math" in config.dataset:
        gt_answer = item["solution"]
    elif config.dataset == 'aime24':
        gt_answer = item["answer"]
    elif config.dataset == 'aime25':
        gt_answer = item["answer"]
    elif config.dataset == 'gpqa_diamond_64':
        gt_answer = item["answer"]
    else:
        raise Exception("Unknown dataset")

    out = {
        "prompt": prompt,
        "question": item["problem"],
        "samples": samples,
        "gt_answer": gt_answer,
    }

    append_yaml(outpath, out, check_if_prompt_matches_existing=True)



def run_inference(item, config: GenerateScriptConfig, tokenizer):
    num_samples = config.num_samples

    outpath = config.save_dir / f"{item['id']}.yaml"
    print("generating for path: ", outpath)

    if 'question' in item:
        item['problem'] = item['question']
    if 'Question' in item:
        item['problem'] = item['Question']

    if config.dataset == "gpqa_diamond_64":
        multiple_choice_string, correct_answer_letter = generate_multiple_choice_answers(item)
        item['answer'] = correct_answer_letter
        prompt = GPQA_ZERO_SHOT_PROMPT.format(problem=item["problem"], options=multiple_choice_string)

    else:
        if config.zero_shot:
            prompt = f"""Problem: {item['problem']}\nMark your solution with \\boxed\nAnswer:"""
        else:
            print("model = ", config.model)
            if "Qwen2.5" in config.model:
                prompt = QWEN_2p5_MATH_PROMPT + f"\n\nProblem:\n{item['problem']}\n\nSolution:"
            else:
                prompt = MATH_COT_PROMPT + f"\n\nProblem:\n{item['problem']}\n\nSolution:"


    if config.apply_chat_template:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    
    if config.rerun == False:   # if config.rerun == False, that means the code will skip over solutions that have already been generated
        if outpath.exists():
            file = load_yaml(outpath)
            if file is not None:
                print("path exists: ", outpath)
                if "samples" in file:
                    # check whether the file has all the samples that are needed
                    num_samples_in_file = len(file["samples"])
                    if num_samples_in_file >= config.num_samples:
                        print("file has enough samples")
                        return
                    else:
                        num_samples = config.num_samples - num_samples_in_file
                        print(f"file already has {num_samples_in_file} samples, adding {num_samples} more now")

                        # For GPQA, the prompt can change everytime since it shuffles the choices, so it's important to use the previous prompt
                        if 'gpqa' in config.dataset and 'prompt' in file:
                            prompt = file['prompt']


    url = f"http://localhost:{config.vllm_port}/generate"
    batch_size = config.batch_size

    samples = []    

    num_iters = int(np.ceil(num_samples / batch_size))  # if num_samples is not a multiple of batch size, generate the closest number of batches (rounding above)
    
    for _ in tqdm(range(num_iters), desc=f"Item {item['id']}"):

        body = {
            "prompt": prompt,
            "max_tokens": config.max_tokens,
            "n": batch_size,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stop": config.stop_strings,
            "logprobs": 1,
        }

        response = requests.post(url, json=body)
        respj = response.json()
        print(respj.keys())
        samples.extend(respj["text"])
        
        if len(samples) >= config.save_every_n_samples:
            print("saving generations during sampling ... ")
            save_samples(item, config, samples, prompt, outpath)
            samples = [] # to not append duplicate answers

    save_samples(item, config, samples, prompt, outpath)


@pydra.main(GenerateScriptConfig)
def main(
    config: GenerateScriptConfig,
):
    if config.dataset == "math_train":
        test_dataset = list(load_dataset("hendrycks/competition_math", "main", split="train", trust_remote_code=True))

    elif config.dataset == "math128":
        test_dataset = list(
            load_dataset(
                "nishadsinghi/MATH128",
                split="test",
            )
        )
    elif config.dataset == 'aime24':
        test_dataset = list(
            load_dataset(
                "nishadsinghi/aime24", split='train'
            )
        )
    
    elif config.dataset == 'aime25':
        test_dataset = list(
            load_dataset(
                "yentinglin/aime_2025", split='train'
            )
        )
    
    elif config.dataset == 'amc23':
        test_dataset = list(
            load_dataset(
                "zwhe99/amc23", split='test'
            )
        )
    
    elif config.dataset == 'gpqa_diamond_64':
        test_dataset = list(
            load_dataset(
                "hbXNov/gpqa_diamond_64", split='train'
            )
        )
    
    else:
        raise Exception("Unknown dataset")

    print(f"Number of test items: {len(test_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(config.model)

    random.seed(config.seed)

    for i, data in enumerate(test_dataset):
        data["id"] = i

    # random.shuffle(test_dataset)
    shuffled_limit = test_dataset

    if config.limit is not None:
        limit = config.limit
    else:
        limit = len(shuffled_limit)

    if config.stride is not None:
        stride = config.stride
    else:
        stride = 1

    if config.offset is not None:
        offset = config.offset
    else:
        offset = 0

    shuffled_limit = shuffled_limit[offset:limit:stride]

    print(f"Total number of items to process: {len(shuffled_limit)}")

    with vllm_manager(config) as vllm_port:
        config.vllm_port = vllm_port

        go_func = partial(run_inference, config=config, tokenizer=tokenizer)

        if config.num_workers not in [0, None]:
            with multiprocessing.Pool(config.num_workers) as pool:
                predictions = list(
                    tqdm(
                        pool.imap_unordered(go_func, shuffled_limit),
                        total=len(shuffled_limit),
                    )
                )
        else:
            predictions = []
            for item in tqdm(shuffled_limit):
                predictions.append(go_func(item))


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("time taken = ", end_time - start_time)
