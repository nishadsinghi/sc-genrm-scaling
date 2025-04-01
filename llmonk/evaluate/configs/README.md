Parameters for the plotting script:

- `samples_dir`: path to folder where solutions are saved
- `verifications_dir`: path to folder where verifications are saved
- `all_num_solutions_plot [list]`: list of all the number of solutions that will be plotted (Best-of-N curves)
- `all_num_verifications_plot`: list of all the number of verifications that will be plotted (Best-of-N curves)
- `all_num_solutions_plot_no_verif`: list of all the number of solutions that will be plotted for the verification-free baselines (Self-consistency and coverage)
- `plot_save_dir`: path to folder where the output plot will be saved
- `math_type`: can be either:
  - `minerva`: if the solutions are of the form: 'The final answer is X. I hope it is correct.'
  - `hendrycks`: if the final answer is enclosed in `\boxed{}`
- `lamb`: the ratio of tokens per verification and tokens per solution (e.g., `1` for GenRM-Base and `2` for `GenRM-FT`)
- `recompute`: if set to `False`, the script will re-use cached values and just plot them again. If set to `True`, it will re-do the entire calculation.
- `use_verification_logprobs`: if this is `True`, then the token probabilities obtained from logprobs are used as the final score of a verification. Otherwise, the score is `1` if the final verdict is `Yes` and `0` if the verdict is `No`
- `use_hybrid`: if this is set to `True`, the script tries to use the logprobs scores whenever possible, and uses binary scores otherwise
- `best_of_n`: f this is set to `True`, the script will plot Best-of-N
- `majority_voting`: if this is set to `True`, the script will plot Self-consistency
- `fix_num_verifications`: if this is set to `True`, each curve will correspond to a fixed number of verifications
- `fix_num_solutions`: if this is set to `True`, each curve will correspond to a fixed number of solutions
- `x_axis_num_solutions`: if this is set to `True`, the x-axis will show the number of solutions and not the total compute
- `model`: if the model is Llama-3.3-70B-Instruct, set this to `llama70b`. Not needed otherwise.

Optional, but recommended:
- `max_num_solutions_with_verifs`: the maximum number of solutions for which verifications are available
- `max_num_solutions_without_verifs`: the maximum number of solutions (regardless of whether verifications are available or not). This will always be greater than or equal to `max_num_solutions_with_verifs`.
- `max_num_verifications`: number of verifications available per solution

(The script can automatically infer these values from the solution and verification files, but sometimes that logic can fail so specifying these values is safer)