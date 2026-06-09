#!/bin/bash
#PBS -q short-c
#PBS -l select=1
#PBS -W group_list=go25
#PBS -j oe

module load singularity
cd CodeGeeX
source env/bin/activate
bash scripts/evaluate_humanevalx_all.sh evaluation_qwen2.5 codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_qwen2.5.csv
bash scripts/evaluate_humanevalx_all.sh evaluation_qwen3.5 codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_qwen3.5.csv
bash scripts/evaluate_humanevalx_all.sh evaluation_qwen2.5_grpo_reward7b codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_qwen2.5_grpo_reward7b.csv
bash scripts/evaluate_humanevalx_all.sh evaluation_qwen2.5_grpo_reward30b codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_qwen2.5_grpo_reward30b.csv
bash scripts/evaluate_humanevalx_all.sh evaluation_qwen2.5_grpo_reward80b codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_qwen2.5_grpo_reward80b.csv
bash scripts/evaluate_humanevalx_all.sh evaluation_qwen3.5_grpo_reward7b codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_qwen3.5_grpo_reward7b.csv
bash scripts/evaluate_humanevalx_all.sh evaluation_qwen3.5_grpo_reward30b codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_qwen3.5_grpo_reward30b.csv
bash scripts/evaluate_humanevalx_all.sh evaluation_qwen3.5_grpo_reward80b codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_qwen3.5_grpo_reward80b.csv
bash scripts/evaluate_humanevalx_all.sh evaluation_qwen2.5_grpo_reward30b_v2_100 codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_qwen2.5_grpo_reward30b_v2_100.csv
bash scripts/evaluate_humanevalx_all.sh evaluation_qwen2.5_grpo_reward30b_v2_194 codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_qwen2.5_grpo_reward30b_v2_194.csv
bash scripts/evaluate_humanevalx_all.sh evaluation_qwen3.5_grpo_reward80b_v2_100 codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_qwen3.5_grpo_reward80b_v2_100.csv
bash scripts/evaluate_humanevalx_all.sh evaluation_qwen3.5_grpo_reward80b_v2_194 codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_qwen3.5_grpo_reward80b_v2_194.csv
