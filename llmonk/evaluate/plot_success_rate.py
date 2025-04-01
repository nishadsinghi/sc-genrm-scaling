# import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import random

import numpy as np

from eval_utils import (
    compute_best_of_k,
    compute_weighted_sc,
    count_verifications,
    get_gt_solution,
    get_predicted_solutions,
    get_solution_correctness,
    get_solution_score,
    get_verification_scores,
    load_pickle,
    save_pickle,
)
from math_pass_k import estimate_pass_at_k
from matplotlib import pyplot as plt
from tqdm import tqdm
import os


def process_problem(problem_id, num_solutions, total_num_solution_without_verifications, num_verifications, samples_dir, verifications_dir, use_verification_logprobs, use_hybrid, math_type, model, seed):
    # Process a single problem: get ground truth solution, predicted solutions, and verification scores
    np.random.seed(seed)
    random.seed(seed)
    
    gt_solution = get_gt_solution(problem_id, samples_dir)

    # get all solutions
    solution_ids = list(range(total_num_solution_without_verifications))
    predicted_solutions_for_problem = get_predicted_solutions(problem_id, solution_ids, samples_dir, math_type=math_type, model=model)

    # get verifications
    verifications = []
    for solution_id in range(num_solutions):
        verification_ids = list(range(num_verifications))
        verifications_for_solution = get_verification_scores(
                    problem_id, solution_id, verification_ids, verifications_dir, use_verification_logprobs, use_hybrid
                )
        verifications.append(verifications_for_solution)
    
    verifications = np.array(verifications, dtype=float)

    return gt_solution, predicted_solutions_for_problem, verifications


def process_problem_wrapper(args):
    # Unpack arguments and call process_problem
    problem_id, num_solutions, total_num_solution_without_verifications, num_verifications, samples_dir, verifications_dir, use_verification_logprobs, use_hybrid, math_type, model, seed = args
    np.random.seed(seed)
    random.seed(seed)
    return process_problem(problem_id, num_solutions, total_num_solution_without_verifications, num_verifications, samples_dir, verifications_dir, use_verification_logprobs, use_hybrid, math_type, model, seed)


def compute_metrics_for_problem(
    problem_id,
    all_verifications,
    all_gt_solutions,
    all_predicted_solutions,
    num_solutions_list,
    all_num_solutions_plot_no_verif,
    num_verifications_list,
    best_of_n,
    weighted_sc,
    seed,
    num_reps_sc
):
    """
    Computes coverage, majority voting, best-of-k, and weighted SC for all pairs
    (s in num_solutions_list, v in num_verifications_list) for the given problem_id.
    Returns a dictionary:

    results = {
       "coverage": dict of form { s -> coverage_value },
       "majority_voting": dict of form { s -> majority_value },
       "best_of_k": dict of form { (s, v) -> best_of_k_value },
       "weighted_sc": dict of form { (s, v) -> weighted_sc_value }
    }
    """
    np.random.seed(seed)
    random.seed(seed)
    
    gt_solution = all_gt_solutions[problem_id]
    predicted_solutions = all_predicted_solutions[problem_id]

    # Calculate correctness flags for each predicted solution
    solution_correctnesses = [
        get_solution_correctness(predicted_solutions[i], gt_solution) for i in range(len(predicted_solutions))
    ]

    # results dictionary
    results = {
        "coverage": {},
        "majority_voting": {},
    }
    if weighted_sc:
        results["weighted_sc"] = {}
    if best_of_n:
        results["best_of_k"] = {}

    # coverage baseline
    num_samples = len(solution_correctnesses)
    num_corrects = np.sum(solution_correctnesses)
    for s in all_num_solutions_plot_no_verif:
        pass_at_k_for_problem = estimate_pass_at_k([num_samples], [num_corrects], s)
        results["coverage"][s] = pass_at_k_for_problem

    # majority voting baseline
    for s in all_num_solutions_plot_no_verif:
        majority_vote = compute_weighted_sc(
            np.ones(len(solution_correctnesses)),
            predicted_solutions,
            gt_solution,
            s,
            num_reps_sc
        )
        results["majority_voting"][s] = majority_vote

    num_solutions_with_verifications = np.shape(all_verifications)[1]

    # best_of_k and weighted_sc vary with both s and v
    for v in num_verifications_list:
        solution_scores = []
        for sol_id in range(num_solutions_with_verifications):
            score = get_solution_score(all_verifications, problem_id, sol_id, v)
            solution_scores.append(score)

        for s in num_solutions_list:
            predicted_solutions_with_verifications = predicted_solutions[:np.size(solution_scores)]
            
            if best_of_n:
                best_of_k_val = compute_best_of_k(solution_scores, predicted_solutions_with_verifications, gt_solution, s)
                results["best_of_k"][(s, v)] = best_of_k_val
            
            if weighted_sc:
                weighted_sc_val = compute_weighted_sc(solution_scores, predicted_solutions_with_verifications, gt_solution, s, num_reps_sc)
                results["weighted_sc"][(s, v)] = weighted_sc_val

    return results


# Wrapper to call compute_metrics_for_problem with correct arguments
def wrapper(arg):
    pid, av, ag, ap, ns_list, all_num_solutions_plot_no_verif, nv_list, best_of_n, weighted_sc, seed, num_reps_sc = arg
    np.random.seed(seed)
    random.seed(seed)

    return compute_metrics_for_problem(
        pid, av, ag, ap, ns_list, all_num_solutions_plot_no_verif, nv_list, best_of_n, weighted_sc, seed, num_reps_sc
    )


def main(samples_dir,
         verifications_dir,
         num_problems,
         max_num_solutions_with_verifs,
         max_num_solutions_without_verifs,
         max_num_verifications,
         all_num_solutions_plot,
         all_num_solutions_plot_no_verif,
         all_num_verifications_plot,
         plot_save_dir,
         plot_title,
         use_verification_logprobs,
         recompute,
         best_of_n=False,
         weighted_sc=False,
         majority_voting=False,
         coverage=False,
         fix_num_solutions=False,
         fix_num_verifications=True,
         x_axis_num_solutions=False,
         use_hybrid=False,
         math_type='minerva',
         model="llama",
         seed=33,
         num_reps_sc=25,
         lambda_=1):

    np.random.seed(seed)
    random.seed(seed)
    
    # Create the directory to save plots if it doesn't already exist
    Path(plot_save_dir).mkdir(parents=True, exist_ok=True)

    # Create a single figure
    plt.figure(figsize=(12, 8))

    if use_hybrid:
        pickle_filename = "plot_points_success_rate_vs_budget_logprobs=hybrid.pickle"
    else:
        pickle_filename = f"plot_points_success_rate_vs_budget_logprobs={use_verification_logprobs}.pickle"
    plot_points_path = os.path.join(plot_save_dir, pickle_filename)

    if os.path.exists(plot_points_path) and not recompute:
        print("FOUND SAVED PLOT POINTS, LOADING")
        plot_points = load_pickle(plot_points_path)
    else:
        print("PLOT POINTS DID NOT EXIST OR RECOMPUTE FLAG SET, COMPUTING")

        if use_hybrid:
            raw_data_pickle_path = os.path.join(plot_save_dir, "raw_data_logprobs=hybrid.pickle")
        else:
            raw_data_pickle_path = os.path.join(plot_save_dir, f"raw_data_logprobs={use_verification_logprobs}.pickle")

        if os.path.exists(raw_data_pickle_path) and not recompute:
            print("RAW DATA EXISTS, LOADING")
            raw_data = load_pickle(raw_data_pickle_path)
            all_gt_solutions = raw_data["all_gt_solutions"]
            all_predicted_solutions = raw_data["all_predicted_solutions"]
            all_verifications = raw_data["all_verifications"]
            num_problems = len(all_gt_solutions)
            
        else:
            print("RAW DATA DID NOT EXIST OR RECOMPUTE FLAG SET, COMPUTING FROM SCRATCH")
            num_problems_, num_solutions_, total_num_solution_without_verifications_, num_verifications_ = count_verifications(
                verification_dir=verifications_dir, samples_dir=samples_dir
            )

            if num_problems is None:
                num_problems = num_problems_
            
            if max_num_solutions_with_verifs is None:
                num_solutions = num_solutions_
            else:
                num_solutions = max_num_solutions_with_verifs
            
            if max_num_solutions_without_verifs is None:
                total_num_solution_without_verifications = total_num_solution_without_verifications_
            else:
                total_num_solution_without_verifications = max_num_solutions_without_verifs

            if max_num_verifications is None:
                num_verifications = num_verifications_
            else:
                num_verifications = max_num_verifications

            # num_solutions = 128
            # total_num_solution_without_verifications = 128
            # num_verifications = 32
            
            print(f"num problems = {num_problems}, num solutions with verifications = {num_solutions}, total num solutions without verifications = {total_num_solution_without_verifications}, num verifications = {num_verifications}")

            if max(all_num_verifications_plot) > num_verifications:
                raise Exception("You're trying to plot more verifications than are present in the data")

            # Use multiprocessing to parallelize data loading for all problems
            with Pool() as pool:
                results = list(
                    tqdm(
                        pool.imap(
                            process_problem_wrapper,
                            [
                                (problem_id, num_solutions, total_num_solution_without_verifications, num_verifications,
                                 samples_dir, verifications_dir, use_verification_logprobs, use_hybrid, math_type, model, seed)
                                for problem_id in range(num_problems)
                            ],
                        ),
                        total=num_problems,
                    )
                )

            # Filter out None values
            print(results[:5])
            # print(f"Filtering out None values from {len(results)} results")
            results = [r for r in results if r is not None]
            # print(f"resulted in {len(results)}")

            # Unpack results
            all_gt_solutions, all_predicted_solutions, all_verifications = zip(*results)
            all_verifications = np.array(all_verifications)

            # Save the computed raw data
            raw_data = {
                "all_gt_solutions": list(all_gt_solutions),
                "all_predicted_solutions": list(all_predicted_solutions),
                "all_verifications": all_verifications,
            }
            save_pickle(raw_data, raw_data_pickle_path)

        # Initialize plot points structure
        plot_points = {"coverage": {}, "majority_voting": {}, "best_of_k": {}, "weighted_sc": {}}

        # We will gather partial results from each problem and aggregate
        coverage_values = defaultdict(list)
        majority_values = defaultdict(list)
        best_of_k_values = defaultdict(lambda: defaultdict(list))
        weighted_sc_values = defaultdict(lambda: defaultdict(list))

        # Prepare arguments for parallel processing
        args_for_pool = [
            (pid, all_verifications, all_gt_solutions, all_predicted_solutions,
             all_num_solutions_plot, all_num_solutions_plot_no_verif, all_num_verifications_plot, best_of_n, weighted_sc, seed, num_reps_sc)
            for pid in range(num_problems)
        ]

        # Process all problems in parallel
        with Pool() as pool:
            all_results = list(tqdm(pool.imap(wrapper, args_for_pool), total=len(args_for_pool)))

        # Aggregate across problems
        for rdict in all_results:
            for s in rdict["coverage"]:
                coverage_values[s].append(rdict["coverage"][s])
            for s in rdict["majority_voting"]:
                majority_values[s].append(rdict["majority_voting"][s])

            if best_of_n:
                for (s, v), val in rdict["best_of_k"].items():
                    best_of_k_values[s][v].append(val)
            if weighted_sc:
                for (s, v), val in rdict["weighted_sc"].items():
                    weighted_sc_values[s][v].append(val)

        # Compute averages
        for s in coverage_values:
            plot_points["coverage"][s] = np.mean(coverage_values[s])
        for s in majority_values:
            plot_points["majority_voting"][s] = np.mean(majority_values[s])

        if best_of_n:
            for s in best_of_k_values:
                for v in best_of_k_values[s]:
                    plot_points["best_of_k"][(s, v)] = np.mean(best_of_k_values[s][v])
        if weighted_sc:
            for s in weighted_sc_values:
                for v in weighted_sc_values[s]:
                    plot_points["weighted_sc"][(s, v)] = np.mean(weighted_sc_values[s][v])

        # Save plot_points for future use
        save_pickle(plot_points, plot_points_path)

    # ----------------- PLOTTING -----------------
    print("START PLOTTING")
    plt.rcParams.update(
        {
            "font.size": 18,  # Base font size
            "axes.titlesize": 18,  # Title font size
            "axes.labelsize": 18,  # Axis label font size
            "xtick.labelsize": 18,  # X-axis tick label size
            "ytick.labelsize": 18,  # Y-axis tick label size
            "legend.fontsize": 18,  # Legend font size
        }
    )

    plt.grid(True)
    plt.xscale("log", base=2)
    if x_axis_num_solutions:
        plt.xlabel("Num Solutions")
    else:
        plt.xlabel("Inference Compute Per Problem (FLOPs)")
    plt.ylabel("Success Rate")

    # coverage (only valid for v=0, but depends on s)
    if coverage:
        s_vals = sorted(list(plot_points["coverage"].keys()))
        x_vals = [s * (1 + 0) if not x_axis_num_solutions else s for s in s_vals]
        y_vals = [plot_points["coverage"][s] for s in s_vals]
        plt.plot(x_vals, y_vals, label="Coverage", marker='o', color='red')

    # majority voting (only valid for v=0, but depends on s)
    if majority_voting:
        s_vals = sorted(list(plot_points["majority_voting"].keys()))
        x_vals = [s * (1 + 0) if not x_axis_num_solutions else s for s in s_vals]
        y_vals = [plot_points["majority_voting"][s] for s in s_vals]
        plt.plot(x_vals, y_vals, label="Majority Voting", marker="o", color="black", linestyle="-")

    # best of n and weighted_sc
    if best_of_n or weighted_sc:
        all_s = list(set([pair[0] for pair in plot_points["best_of_k"].keys()]))
        all_v = list(set([pair[1] for pair in plot_points["best_of_k"].keys()]))
        all_v.sort()
        all_s.sort()

        if fix_num_solutions:
            for s in all_s:
                x_vals = [s * (1 + lambda_ * v) if not x_axis_num_solutions else s for v in all_v]

                if best_of_n:
                    y_vals_best = [plot_points["best_of_k"].get((s, v), 0.0) for v in all_v]
                    line_best = plt.plot(x_vals, y_vals_best, label=f"S={s} (Best-of-N)", marker="o")
                    best_color = line_best[0].get_color()

                if weighted_sc:
                    y_vals_wsc = [plot_points["weighted_sc"].get((s, v), 0.0) for v in all_v]
                    if best_of_n:
                        plt.plot(
                            x_vals,
                            y_vals_wsc,
                            label=f"S={s} (Weighted SC)",
                            marker="o",
                            linestyle="--",
                            color=best_color,
                        )
                    else:
                        plt.plot(
                            x_vals,
                            y_vals_wsc,
                            label=f"S={s} (Weighted SC)",
                            marker="o",
                        )

        elif fix_num_verifications:
            for v in all_v:
                x_vals = [s * (1 + lambda_ * v) if not x_axis_num_solutions else s for s in all_s]

                if best_of_n:
                    y_vals_best = [plot_points["best_of_k"].get((s, v), 0.0) for s in all_s]
                    line_best = plt.plot(x_vals, y_vals_best, label=f"V={v} (Best-of-N)", marker="o")
                    best_color = line_best[0].get_color()

                if weighted_sc:
                    if best_of_n:
                        y_vals_wsc = [plot_points["weighted_sc"].get((s, v), 0.0) for s in all_s]
                        if best_of_n:
                            plt.plot(
                                x_vals,
                                y_vals_wsc,
                                label=f"V={v} (Weighted SC)",
                                marker="o",
                                linestyle="--",
                                color=best_color,
                            )
                    else:
                        y_vals_wsc = [plot_points["weighted_sc"].get((s, v), 0.0) for s in all_s]
                        plt.plot(
                            x_vals,
                            y_vals_wsc,
                            label=f"V={v} (Weighted SC)",
                            marker="o",
                        )

    # Add overall title and adjust layout
    plt.title(plot_title, fontsize=16)
    plt.tight_layout()

    # Save the figure
    filename = "plot_success_rate_vs_budget_"
    if use_hybrid:
        filename += "logprobs=hybrid"
    else:
        filename += f"logprobs={use_verification_logprobs}"
    if fix_num_solutions:
        filename += "_fix_num_sols"
    elif fix_num_verifications:
        filename += "_fix_num_verifs"
    filename += ".png"

    plot_save_path = os.path.join(plot_save_dir, filename)
    plt.legend()
    plt.savefig(plot_save_path, bbox_inches="tight")
    print("plot saved at:", plot_save_path)


if __name__ == "__main__":
    import argparse
    import yaml
    
    # Create an argument parser for just the config file path
    parser = argparse.ArgumentParser(description="Compute success rate at different k values.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    
    # Parse the single argument
    args = parser.parse_args()
    
    # Load configuration from YAML file
    try:
        with open(args.config, 'r') as config_file:
            config = yaml.safe_load(config_file)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        exit(1)
    
    # Check for required parameters
    required_params = ["samples_dir", "verifications_dir", "all_num_solutions_plot", 
                      "all_num_solutions_plot_no_verif", "all_num_verifications_plot", 
                      "plot_save_dir"]
    
    for param in required_params:
        if param not in config:
            raise ValueError(f"Required parameter '{param}' not found in configuration file")
    
    # Set default values for optional parameters
    defaults = {
        "num_problems": None,
        "max_num_solutions_with_verifs": None,
        "max_num_solutions_without_verifs": None,
        "max_num_verifications": None,
        "math_type": "minerva",
        "plot_title": "Success Rate Plot",
        "use_verification_logprobs": False,
        "use_hybrid": False,
        "recompute": False,
        "lamb": 1,
        "best_of_n": False,
        "weighted_sc": False,
        "majority_voting": False,
        "coverage": False,
        "fix_num_solutions": False,
        "fix_num_verifications": False,
        "x_axis_num_solutions": False,
        "model": "llama",
        "seed": 33,
        "num_reps_sc": 25,
    }
    
    # Update config with defaults for missing parameters
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    # Check for conflicting parameters
    if config["fix_num_solutions"] == config["fix_num_verifications"] and config["fix_num_solutions"]:
        raise Exception("You can only plot either fix num solutions or fix num verifications")
    
    # If user does not specify any flag among best_of_n, weighted_sc, majority_voting, coverage, plot everything
    if not any([config["best_of_n"], config["weighted_sc"], config["majority_voting"], config["coverage"]]):
        config["best_of_n"] = True
        config["weighted_sc"] = True
        config["majority_voting"] = True
        config["coverage"] = True
    
    # Call the main function with the configuration
    main(
        samples_dir=config["samples_dir"],
        verifications_dir=config["verifications_dir"],
        num_problems=config["num_problems"],
        max_num_solutions_with_verifs=config["max_num_solutions_with_verifs"],
        max_num_solutions_without_verifs=config["max_num_solutions_without_verifs"],
        max_num_verifications=config["max_num_verifications"],
        all_num_solutions_plot=config["all_num_solutions_plot"],
        all_num_solutions_plot_no_verif=config["all_num_solutions_plot_no_verif"],
        all_num_verifications_plot=config["all_num_verifications_plot"],
        plot_save_dir=config["plot_save_dir"],
        plot_title=config["plot_title"],
        use_verification_logprobs=config["use_verification_logprobs"],
        recompute=config["recompute"],
        best_of_n=config["best_of_n"],
        weighted_sc=config["weighted_sc"],
        majority_voting=config["majority_voting"],
        coverage=config["coverage"],
        fix_num_solutions=config["fix_num_solutions"],
        fix_num_verifications=config["fix_num_verifications"],
        x_axis_num_solutions=config["x_axis_num_solutions"],
        use_hybrid=config["use_hybrid"],
        math_type=config["math_type"],
        model=config["model"],
        seed=config["seed"],
        num_reps_sc=config["num_reps_sc"],
        lambda_=config["lamb"]
    )