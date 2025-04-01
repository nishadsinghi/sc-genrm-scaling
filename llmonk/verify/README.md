## GenRM-FT on MATH/AIME24

```
python3 llmonk/verify/generate_verifications.py \
model=hbXNov/llama3.1-8b_train_gpt_4o_verifications_e3_lr5e-7-31389-merged \
verification_template=prompts/llama3.1_8b_instruct/finetuned.txt \
output_dir=llmonk/outputs/verifications/<verification_folder_name> \
samples_dir=llmonk/outputs/<folder_where_solutions_are_saved> \
num_verifications=32 \
--list vllm_args --disable-log-requests list-- 
batch_size=32 \
temperature=0.7 \
logprobs=1 \
max_tokens=2048 \
num_workers=32 \
num_problems=128 \  # 30 for AIME24
num_solutions=256 \
```

Where `model` can be any GenRM-FT fine-tuned for this task.

## GenRM-Base on MATH

```
python3 llmonk/verify/generate_verifications.py \
model=meta-llama/Llama-3.3-70B-Instruct \
verification_template=prompts/llama3.3_70b/genrm_base.txt \
output_dir=llmonk/outputs/verifications/<verification_folder_name> \
samples_dir=llmonk/outputs/<folder_where_solutions_are_saved> \
num_verifications=32 \
--list vllm_args --disable-log-requests list-- 
batch_size=32 \
temperature=0.7 \
logprobs=1 \
max_tokens=1024 \
num_workers=32 \
num_problems=128 \
num_solutions=256 \
```

## GenRM-Base on GPQA

```
python3 llmonk/verify/generate_verifications.py \
model=meta-llama/Llama-3.3-70B-Instruct \
verification_template=prompts/gpqa/zero_shot.txt \
output_dir=llmonk/outputs/verifications/<verification_folder_name> \
samples_dir=llmonk/outputs/<folder_where_solutions_are_saved> \
num_verifications=32 \
--list vllm_args --disable-log-requests list-- 
batch_size=32 \
temperature=0.7 \
logprobs=1 \
max_tokens=1024 \
num_workers=32 \
num_problems=64 \
num_solutions=64 \
```

## QwQ-32B

```
python3 llmonk/verify/generate_verifications.py \
model=Qwen/QwQ-32B \
verification_template=prompts/QwQ32B/genrm_base.txt \
output_dir=llmonk/outputs/verifications/<verification_folder_name> \
samples_dir=llmonk/outputs/<folder_where_solutions_are_saved> \
num_verifications=64 \
--list vllm_args --disable-log-requests list-- 
batch_size=32 \
temperature=0.7 \
logprobs=1 \
max_tokens=32768 \
num_workers=32 \
num_problems=30 \
num_solutions=32 \
```