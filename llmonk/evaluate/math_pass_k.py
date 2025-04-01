import argparse
import itertools
import os
import re
import time
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import yaml

from eval_utils import hendrycks_extract_answer, get_is_correct_hendrycks
from lm_eval.tasks.hendrycks_math.utils import (
    is_equiv as is_equiv_hendrycks,
    last_boxed_only_string as last_boxed_only_string_hendrycks,
    remove_boxed as remove_boxed_hendrycks,
)

from lm_eval.tasks.minerva_math.utils import (
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)
from tqdm import tqdm


# Function to estimate pass@k
def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray], num_correct: Union[List[int], np.ndarray], k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    num_samples and num_correct should be lists of length num_problems
    num_samples[i] = number of samples of the ith problem
    num_correct[i] = number of correct samples of the ith problem
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def get_unnormalized_answer_pred(text: str) -> str:
    INVALID_ANSWER = "[invalidanswer]"
    match = re.search(
        r"(?<=The final answer is):?\s(.*)",
        text,
    )
    if match:
        return match.group(1).strip()
    else:
        return INVALID_ANSWER


def get_is_correct(answer, gt_answer):
    gt_answer = normalize_final_answer(remove_boxed(last_boxed_only_string(gt_answer)))
    predicted_ans = normalize_final_answer(get_unnormalized_answer_pred(answer))
    return predicted_ans == gt_answer or is_equiv(predicted_ans, gt_answer)


def get_is_correct_list(answers, gt_answer):
    return [get_is_correct(answer, gt_answer) for answer in answers]

def get_is_correct_list_hendrycks(answers, gt_answer):
    return [get_is_correct_hendrycks(answer, gt_answer) for answer in answers]

# Function to read YAML files and compute pass@k
def compute_pass_at_k_for_directory(directory_path: str, ks: List[int], plot_path: str, plot_title: str, math_type="minerva"):
    num_samples_list = []
    num_correct_list = []

    # Iterate through all files in the directory
    threshold = 100000
    for filename in tqdm(os.listdir(directory_path), desc="Loading YAML files"):
        if len(num_samples_list) > threshold:
            break
        if filename.endswith(".yaml"):
            file_path = os.path.join(directory_path, filename)
            t0 = time.time()
            with open(file_path, "r") as file:
                data = yaml.safe_load(file)
                answers = data.get("samples", [])
                gt_answer = data.get("gt_answer", "")
                print(f"Time taken to load: {time.time() - t0:.2f} seconds")
                if math_type == "minerva":
                    is_corrects = get_is_correct_list(answers, gt_answer)
                else:
                    is_corrects = get_is_correct_list_hendrycks(answers, gt_answer)
                # data.get("is_corrects", [])
                # if len(is_corrects) == 128:
                #     print(os.path.join(directory_path, filename))
                #     exit()
                print(f"Loaded {len(is_corrects)} samples from {file_path}")
                for i, is_correct in enumerate(is_corrects):
                    if is_correct is None:
                        is_corrects[i] = False
                        print(f"Sample {i} is None : {answers[i]}")
                try:
                    num_correct = sum(is_corrects)
                except:
                    print("file_path: ", file_path)
                    print(is_corrects)
                    exit()
                print(f"Number of correct samples: {num_correct}")
                if num_correct == 0:
                    print("No correct samples found in the file.")
                    if math_type == "minerva":
                        print(f"normalized gt:{normalize_final_answer(remove_boxed(last_boxed_only_string(gt_answer)))}")
                    else:
                        print(f"normalized gt:{gt_answer}")
                    for ans in answers[:5]:
                        if math_type == "minerva":
                            print(f"normalized ans: {normalize_final_answer(get_unnormalized_answer_pred(ans))}")
                        else:
                            print(f"normalized ans: {hendrycks_extract_answer(ans)}")
                    # print(f"some samples: " + "\n\n".join(answers[:2]))
                num_samples = len(is_corrects)
                num_samples_list.append(num_samples)
                num_correct_list.append(num_correct)
            
    if not num_samples_list:
        print("No YAML files found in the directory.")
        return

    # Compute pass@k for each k
    avg_pass_at_k = {}
    for k in ks:
        pass_at_k = estimate_pass_at_k(num_samples_list, num_correct_list, k)
        avg_pass_at_k[k] = np.mean(pass_at_k)

    # Plot the results and save to the specified path
    plot_pass_at_k(avg_pass_at_k, plot_path, plot_title)


# Function to plot pass@k and save the plot
def plot_pass_at_k(avg_pass_at_k: dict, plot_path: str, plot_title: str):
    ks = list(avg_pass_at_k.keys())
    values = list(avg_pass_at_k.values())

    plt.figure()
    plt.plot(ks, values, marker="o")
    plt.title(plot_title)
    plt.xlabel("k")
    plt.ylabel("Average pass@k")
    plt.xticks(ks)
    plt.grid(True)
    # plt.ylim(0, 1)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # Save the plot to the specified path
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


# Main function to handle command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Compute pass@k for YAML problem results")
    parser.add_argument("--directory", type=str, help="Path to the directory containing YAML files")
    parser.add_argument("--ks", type=int, nargs="+", help="List of k values to compute pass@k")
    parser.add_argument("--plot_path", type=str, required=True, help="Path to save the plot (e.g., /path/to/plot.png)")
    parser.add_argument("--plot_title", type=str, required=True, help="Title for the plot")
    parser.add_argument("--math_type", type=str, default="minerva", help="Type of math task (minerva or hendrycks)")

    args = parser.parse_args()

    compute_pass_at_k_for_directory(args.directory, args.ks, args.plot_path, args.plot_title, args.math_type)


if __name__ == "__main__":
    main()
