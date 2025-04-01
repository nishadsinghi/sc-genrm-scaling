# Example Commands

## MATH and AIME24

```
python llmonk/generate/generate_solutions.py \
model=meta-llama/Llama-3.1-8B-Instruct \
save_dir=llmonk/outputs/<output_folder_name> \
--list vllm_args --disable-log-requests list-- --list stop_strings Problem: list-- \
temperature=0.7 \
num_samples=256 \
batch_size=64 \
num_workers=32 \
zero_shot=False \
dataset=math128 \
max_tokens=1024 \
apply_chat_template=False \
save_every_n_samples=64 \
gpus=0  # 0,1,2,3 for Llama 70B
```

- Replace the `model` name for a different model
- Replace `math128` with `aime24` for AIME24
- Use `dataset=math_train` for MATH train split
- We use `apply_chat_template=False` for these datasets as the in-context examples enable the model to format its response correctly

## AIME25

```
python llmonk/generate/generate_solutions.py \
model=Qwen/QwQ-32B \
save_dir=llmonk/outputs/<output_folder_name> \
--list vllm_args --disable-log-requests list-- --list stop_strings Problem: list-- \
temperature=0.7 \
num_samples=256 \
batch_size=64 \
num_workers=32 \
zero_shot=True \
dataset=aime25 \
max_tokens=32768 \
apply_chat_template=True \
save_every_n_samples=64 \
gpus=0,1
```

## GPQA-Diamond

```
python llmonk/generate/generate_solutions.py \
model=meta-llama/Llama-3.3-70B-Instruct \
save_dir=llmonk/outputs/<output_folder_name> \
--list vllm_args --disable-log-requests list-- --list stop_strings Problem: list-- \
temperature=0.7 \
num_samples=256 \
batch_size=64 \
num_workers=32 \
dataset=gpqa_diamond_64 \
max_tokens=1024 \
apply_chat_template=True \
save_every_n_samples=64 \
gpus=0,1,2,3
```
