#!/bin/bash
#PBS -q short-c
#PBS -l select=1
#PBS -W group_list=go25
#PBS -j oe

module load singularity
cd CodeGeeX
source env/bin/activate
# bash scripts/evaluate_humanevalx_all.sh grpo_0116 codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_grpo_0116.csv
bash scripts/evaluate_humanevalx_all.sh drgrpo_0116 codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_drgrpo_0116.csv
bash scripts/evaluate_humanevalx_all.sh dapo_0116 codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_dapo_0116.csv
bash scripts/evaluate_humanevalx_all.sh grpo_0123_code codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_grpo_0123_code.csv
bash scripts/evaluate_humanevalx_all.sh grpo_0123_md codegeex/benchmark/humaneval-x/humanevalx.sif 64 30 evaluation_grpo_0123_md.csv