import os
import pickle
import random
import re
from collections import defaultdict


import scipy.special as scsp
import numpy as np
from lm_eval.tasks.hendrycks_math.utils import is_equiv as is_equiv_hendrycks
from lm_eval.tasks.minerva_math.utils import (
    get_unnormalized_answer,
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)

from llmonk.utils import load_yaml, save_yaml


def hendrycks_extract_answer(output: str) -> str:
    """Extract the final answer from a model-generated solution, which is expected to be in the format of \boxed{answer}.

    Uses the same logic as hendrycks_math.

    Args:
        output (str): Model-generated solution text

    Returns:
        str: Extracted final answer. Returns empty string if no answer found in \boxed.
    """

    try:
        answer = remove_boxed(last_boxed_only_string(output))
        return answer
    except:
        return ""


def get_is_correct_hendrycks(answer, gt_answer):
    """
    answer is raw
    gt_answer is final gt answer
    """
    if type(gt_answer) == str and "\\boxed" in gt_answer:
        gt_answer = normalize_final_answer(remove_boxed(last_boxed_only_string(gt_answer)))
    else:
        gt_answer = gt_answer

    predicted_ans = hendrycks_extract_answer(answer)
    raw_check = False
    try:
        raw_check = float(predicted_ans) == float(gt_answer)
    except:
        pass
    return is_equiv_hendrycks(predicted_ans, gt_answer) or raw_check


def count_verifications(verification_dir, samples_dir, no_verifier=False):
    """
    verification_dir has folders like problem_0, problem_1, ...; num_problems is the number of folders
    every folder (like problem_0) will have files like solution_0.yaml, solution_1.yaml, ...; num_solutions is the number of files in the folder problem_0
    every file like solution_0.yaml will have multiple verifications under 'verifications'. num_verifications is the length of verifiactions in problem_0/solution_0.yaml
    return num_problems, num_solutions, num_verifications
    """

    # compute number of solutions (regardless of whether we have verifications for them or not)
    num_samples_without_verifications = len(load_yaml(os.path.join(samples_dir, "0.yaml"))["samples"])

    if no_verifier:
        print(f"Skipping verification dir")
        num_problems = len([x for x in os.listdir(samples_dir) if x.endswith(".yaml")])
        num_solutions = num_samples_without_verifications
        num_verifications = 0
    else:
        problem_folders = [x for x in os.listdir(verification_dir) if ("problem_" in x)]

        num_problems = len(problem_folders)

        problem_0_path = os.path.join(verification_dir, "problem_0")
        problem_0_solutions = [x for x in os.listdir(problem_0_path) if "solution_" in x]
        num_solutions = len(problem_0_solutions)
    
        solution_0_path = os.path.join(problem_0_path, "solution_0.yaml")
        solution_0_yaml = load_yaml(solution_0_path)
        verifications = solution_0_yaml["verifications"]
        num_verifications = len(verifications)

    return num_problems, num_solutions, num_samples_without_verifications, num_verifications


def get_gt_solution(problem_id, samples_dir):
    path = os.path.join(samples_dir, f"{problem_id}.yaml")
    yaml = load_yaml(path)

    yaml["gt_answer"] = str(yaml["gt_answer"])

    if "boxed" in yaml["gt_answer"]:
        try:
            answer = normalize_final_answer(remove_boxed(last_boxed_only_string(yaml["gt_answer"])))
        except Exception as e:
            print(f"problem = {problem_id}")
            print(yaml["gt_answer"])
            print(last_boxed_only_string(yaml["gt_answer"]))
            print(e)
            exit()
    else:
        if yaml["gt_answer"].endswith(".0"):
            # happens for amc answers
            answer = yaml["gt_answer"][:-2]
            return answer
        return yaml["gt_answer"]
    return answer


def get_predicted_solutions(problem_id, solution_ids, samples_dir, model="llama", math_type="minerva"):
    if isinstance(solution_ids, int):   # in case the solution ids is not a list
        solution_ids = [solution_ids]

    path = os.path.join(samples_dir, f"{problem_id}.yaml")
    yaml = load_yaml(path)
    solutions = []
    for solution_id in solution_ids:
        try:
            solution = yaml["samples"][solution_id]
            if math_type == "minerva":
                if model == "llama":
                    predicted_answer = normalize_final_answer(get_unnormalized_answer(solution))
                elif model == "llama70b":
                    predicted_answer = normalize_final_answer(get_unnormalized_answer_llama70(solution))
                else:
                    print(f"WARNING: model {model} not recognized. Will use default")
                    predicted_answer = normalize_final_answer(get_unnormalized_answer(solution))
            elif math_type == "hendrycks":
                predicted_answer = hendrycks_extract_answer(solution)
            else:
                raise Exception("math_type not recognized")
            solutions.append(predicted_answer)
        
        except:
            print(f'COULD NOT FIND SOLUTION\tPROBLEM_ID: {problem_id}, SOLUTION_ID: {solution_id}')
            raise Exception("Couldn't find solution")

    if len(solutions) == 1:
        return solutions[0]
    
    return solutions


# this function is for llm as a judge when the output format is like
# solution status: [CORRECT]
def check_solution_status(text):
    """
    Check if a solution is marked as correct or incorrect in the given text.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        bool or None: True if correct, False if incorrect, None if no status found
    """
    # Convert text to lowercase for case-insensitive matching
    text = text.lower()
    
    # Define patterns for solution status indicators
    patterns = [
        r'solution\s+status\s*:?\s*[\[\(]?\s*incorrect\s*[\]\)]?',
        r'solution\s+status\s*:?\s*[\[\(]?\s*correct\s*[\]\)]?',
        r'status\s*:?\s*[\[\(]?\s*incorrect\s*[\]\)]?',
        r'status\s*:?\s*[\[\(]?\s*correct\s*[\]\)]?',
        r'assessment\s*:?\s*[\[\(]?\s*incorrect\s*[\]\)]?',
        r'assessment\s*:?\s*[\[\(]?\s*correct\s*[\]\)]?'
    ]
    
    # Check for incorrect first (as it's more specific)
    for pattern in patterns:
        if 'incorrect' in pattern and re.search(pattern, text):
            return False
            
    # Then check for correct
    for pattern in patterns:
        if 'correct' in pattern and re.search(pattern, text):
            return True
    
    # If neither is found, return None
    return None


def extract_final_verification(input_str):
    """
    Detects if a string contains a question about correctness and extracts the final "Yes" or "No" answer.

    Args:
        input_str (str): Input string to analyze.

    Returns:
        bool or None: True if the answer is "Yes", False if "No", None if pattern not found.
    """

    # Strip leading/trailing whitespaces
    text = input_str.strip()

    # Define the pattern to match the question and capture the answer part
    pattern = r"(?i)Is the (?:answer|solution) correct(?:\s*\(yes/no\))?\??\**\s*(.+)$"

    match = re.search(pattern, text)
    if match:
        answer_part = match.group(1)

        # Remove LaTeX commands like \boxed{...} or \text{...}
        # This pattern removes LaTeX commands and keeps the text inside
        answer_text = re.sub(r"\\[a-zA-Z]+\s*{([^{}]+)}", r"\1", answer_part)

        # Remove any remaining LaTeX commands and special characters
        answer_text = re.sub(r"\\[a-zA-Z]+|\{|\}|\(|\)|\*\*|[\^\$\\]", "", answer_text)

        # Convert to lowercase for case-insensitive comparison
        answer_text = answer_text.lower().strip()

        # Check for "yes" or "no" in the processed answer_text
        if "yes" in answer_text:
            return True
        elif "no" in answer_text:
            return False

    return check_solution_status(input_str)
    # return None


def extract_final_verification_from_logprobs(p_yes, p_no):
    if p_yes != -1:
        return p_yes

    elif p_no != -1:
        return 1 - p_no

    else:
        return None


def verification_extraction_wrapper(verification, use_hybrid, use_logprobs):
    if use_hybrid:
        p_yes, p_no = verification["p(yes)"], verification["p(no)"]
        logprob_verifications = extract_final_verification_from_logprobs(p_yes, p_no)
        if logprob_verifications:
            return logprob_verifications
        return extract_final_verification(verification["verification"])

    if use_logprobs:
        p_yes, p_no = verification["p(yes)"], verification["p(no)"]
        extract_yes_no = extract_final_verification_from_logprobs(p_yes, p_no)
        if use_hybrid:
            if extract_yes_no:
                return extract_yes_no
            else:
                return extract_final_verification(verification["verification"])
        else:
            return extract_yes_no
    else:
        return extract_final_verification(verification["verification"])
    

def get_verification_scores(problem_id, solution_id, verification_ids, verifications_dir, use_logprobs, use_hybrid):
    if isinstance(verification_ids, int):
        verification_ids = [verification_ids]
    
    path = os.path.join(verifications_dir, f"problem_{problem_id}", f"solution_{solution_id}.yaml")
    yaml = load_yaml(path)

    verification_scores = []

    for verification_id in verification_ids:
        try:
            verification = yaml["verifications"][verification_id]
            verification_extracted = verification_extraction_wrapper(verification, use_hybrid, use_logprobs)
            verification_scores.append(verification_extracted)
        except:
            print("could not find verification for: ", problem_id, solution_id, verification_id)
            raise Exception(
                f"Could not find verification. problem = {problem_id}, solution = {solution_id}, verification = {verification_id}"
            )

    return verification_scores

def get_solution_score(all_verifications, problem_id, solution_id, num_verifications_plot):
    if num_verifications_plot > all_verifications.shape[-1]:
        raise Exception("Not enough verifications!")
    verifications = all_verifications[problem_id, solution_id, :num_verifications_plot]
    try:
        verifications = [x for x in verifications if not np.isnan(x)]
    except:
        print('verifications:')
        print(verifications)
        print("type = ", [type(x) for x in verifications])
        raise Exception()
    if len(verifications) == 0:
        return 0
    else:
        return np.mean(verifications)


def get_solution_correctness(predicted_solution, gt_solution):
    return (
        predicted_solution == gt_solution
        or is_equiv(predicted_solution, gt_solution)
        or is_equiv_hendrycks(predicted_solution, gt_solution)
    )


def verifier_at_k(scores, k):
    EPS = 1e-9
    N = len(scores)  # Total number of samples
    # Pick ith example and (k-1) examples after it.
    numerators = [scores[i] * scsp.binom(N - i - 1, k - 1) for i in range(N - k + 1)]
    denominator = scsp.binom(N, k)  # - scsp.binom(N - C, k)
    fracs = [n / (denominator + EPS) for n in numerators]
    return sum(fracs)


def compute_best_of_k(solution_scores, predicted_solutions, gt_answer, k, set_invalid_to_zero=True):
    if len(solution_scores) != len(predicted_solutions):
        raise Exception
    assert len(solution_scores) == len(predicted_solutions)
    solution_correctnesses = [
        get_solution_correctness(predicted_solutions[i], gt_answer) for i in range(len(solution_scores))
    ]

    # set the score of invalidanswers to zero
    if set_invalid_to_zero:
        for i in range(len(solution_scores)):
            if is_invalid_answer(predicted_solutions[i]):
                solution_scores[i] = 0

    all_values = []
    for _ in range(100):
        z = list(zip(solution_correctnesses, solution_scores))
        random.shuffle(z)
        z = sorted(z, key=lambda x: x[1], reverse=True)
        correctnesses_sorted = list(zip(*z))[0]
        all_values.append(verifier_at_k(correctnesses_sorted, k))

    return np.mean(all_values)


def is_invalid_answer(answer):
    if "invalidanswer" in answer:
        return True
    elif answer == "":
        return True
    else:
        return False


def compute_weighted_sc(solution_scores, predicted_solutions, gt_answer, num_solutions, num_reps_sc, set_invalid_to_zero=True):
    # Convert inputs to numpy arrays for faster operations
    solution_scores = np.array(solution_scores)

    # Pre-process invalid answers once
    if set_invalid_to_zero:
        invalid_mask = np.array([is_invalid_answer(sol) for sol in predicted_solutions])
        solution_scores[invalid_mask] = 0

    # Pre-allocate array for successes
    num_reps = num_reps_sc
    if num_solutions == np.size(solution_scores):
        num_reps = 1
    successes = np.zeros(num_reps)
    total_num_solutions = len(solution_scores)

    # Generate all random samples at once
    try:
        all_samples = np.array([random.sample(range(total_num_solutions), num_solutions)
                           for _ in range(num_reps)])
    except:
        print("total_num_solutions: ", total_num_solutions, " num_solutions: ", num_solutions)
        raise Exception
    
    for i in range(num_reps):
        sampled_idxs = all_samples[i]
        sampled_predictions = [predicted_solutions[idx] for idx in sampled_idxs]
        sampled_scores = solution_scores[sampled_idxs]

        # Use a faster dictionary accumulation
        weighted_predictions = {}
        for pred, score in zip(sampled_predictions, sampled_scores):
            weighted_predictions[pred] = weighted_predictions.get(pred, 0) + score

        # Handle invalid answer case
        if set_invalid_to_zero:
            weighted_predictions["[invalidanswer]"] = -1
            weighted_predictions[""] = -1

        # Find prediction with highest rating
        predicted_solution = max(weighted_predictions, key=weighted_predictions.get)
        
        sorted_dict = dict(sorted(weighted_predictions.items(), key=lambda item: item[1], reverse=True))
        successes[i] = get_solution_correctness(predicted_solution, gt_answer)

    return float(np.mean(successes))


def load_pickle(path):
    print("loading pickle from: ", path)
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(data, path):
    print("saving pickle at: ", path)
    with open(path, "wb") as f:
        pickle.dump(data, f)


####### functions to help with math_datasets.py #######

ANS_RE_GSM8k = re.compile(r"#### (\-?[\$0-9\.\,]+)")
INVALID_ANS_GSM8k = "[invalid]"
GSM8K_IGNORE_REGEXES = [",", "\\$", "\\.$"]


def remove_trailing_parenthesis(s):
    return re.sub(r"\\?\)$", "", s)


def filter_ignores(st, regexes_to_ignore):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            st = re.sub(s, "", st)
    return st


def extract_answer_gsm8k(completion):
    match = ANS_RE_GSM8k.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = filter_ignores(
            match_str,
            GSM8K_IGNORE_REGEXES,
        )
        return match_str
    else:
        return INVALID_ANS_GSM8k


def get_unnormalized_answer_llama70(text: str) -> str:
    if ". I hope it is correct" in text:
        text = text.split(". I hope it is correct")[0]
    INVALID_ANSWER = "[invalidanswer]"
    match = re.search(
        # r"(?<=The final answer is(?:)\s)(.*)",
        r"(?<=The final answer is):?\s(.*)",
        text,
    )
    if match:
        return match.group(1).strip().rstrip(". I hope it is correct")
    else:
        # print(f"Invalid answer: {text[-100:]}")
        return INVALID_ANSWER


def is_correct_gsm8k(model_completion, gt_example):
    gt_answer = extract_answer_gsm8k(gt_example)
    assert gt_answer != INVALID_ANS_GSM8k
    model_answer = extract_answer_gsm8k(model_completion)
    return model_answer == gt_answer or is_equiv(model_answer, gt_answer)


def is_correct_minerva(og_pred, gt):
    pred = remove_trailing_parenthesis(normalize_final_answer(get_unnormalized_answer(og_pred)))
    # pred = remove_trailing_parenthesis(normalize_final_answer(get_unnormalized_answer_GPT(og_pred)))
    if "\\boxed" in gt:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = gt

    # string equality check needed because of https://github.com/EleutherAI/lm-evaluation-harness/issues/2212
    return pred == gt or is_equiv(pred, gt)


def is_correct_hendrycks(answer, gt_answer):
    if 'boxed' in gt_answer:
        gt_answer = normalize_final_answer(remove_boxed(last_boxed_only_string(gt_answer)))
    predicted_ans = hendrycks_extract_answer(answer)
    return is_equiv_hendrycks(predicted_ans, gt_answer)


def is_correct(sample: str, gt_answer: str, dset: str):
    if dset == "gsm8k":
        return is_correct_gsm8k(sample, gt_answer)
    elif dset == "math":
        return is_correct_minerva(sample, gt_answer)
    elif dset == "math_minerva":
        return is_correct_minerva(sample, gt_answer)
    elif dset == "math_hendrycks":
        return is_correct_hendrycks(sample, gt_answer)
    else:
        raise ValueError(f"Dataset {dset} not supported")
