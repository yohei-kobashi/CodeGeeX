#!/bin/bash
#PBS -q short-c
#PBS -l select=1
#PBS -W group_list=go25
#PBS -j oe

module load singularity
cd CodeGeeX
source env/bin/activate
bash scripts/evaluate_humanevalx_all.sh evaluation_qwen2.5_grpo_reward7b codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_qwen2.5_grpo_reward7b.csv
bash scripts/evaluate_humanevalx_all.sh evaluation_qwen2.5_grpo_reward30b codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_qwen2.5_grpo_reward30b.csv
bash scripts/evaluate_humanevalx_all.sh evaluation_qwen2.5_grpo_reward80b codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_qwen2.5_grpo_reward80b.csv
bash scripts/evaluate_humanevalx_all.sh evaluation_qwen3.5_grpo_reword7b codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_qwen3.5_grpo_reward7b.csv
bash scripts/evaluate_humanevalx_all.sh evaluation_qwen3.5_grpo_reword30b codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_qwen3.5_grpo_reward30b.csv
bash scripts/evaluate_humanevalx_all.sh evaluation_qwen3.5_grpo_reward80b codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_qwen3.5_grpo_reward80b.csv