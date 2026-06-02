#!/bin/bash
#PBS -q short-g
#PBS -l select=1
#PBS -W group_list=go25
#PBS -j oe
module purge
# module load cuda/12.8
# module load cudnn/9.10.1.4
# module load nvidia/25.3
# module load nv-hpcx/25.3
# source /work/go25/b20048/miniconda3/etc/profile.d/conda.sh
# conda activate inference_env
# export PATH="$CONDA_PREFIX/bin:/opt/rh/gcc-toolset-14/root/usr/bin:$PATH"

# export CC=/opt/rh/gcc-toolset-14/root/usr/bin/gcc
# export CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++
# export TRITON_CC="$CC"
# export TRITON_CXX="$CXX"
# export CUDAHOSTCXX="$CXX"

# export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
module load nvidia/25.9
module load singularity/4.2.1
singularity shell --nv --bind /work/go25:/work/go25 /work/gj26/share/sif/vllm_v0.21.0.sif
cd CodeGeeX

PYTHON_MODULE="codegeex.benchmark.humaneval-x.translate_humaneval_x_vllm"
BENCHMARK_DIR="codegeex/benchmark/humaneval-x"
BATCH_SIZE=128
SAMPLES_PER_PROBLEM=1

LANGUAGES=(cpp go java js python rust)

MODEL_NAMES=(
  "Qwen/Qwen3-Coder-30B-A3B-Instruct"
  "ByteDance-Seed/Seed-Coder-8B-Instruct"
  "Qwen/Qwen2.5-Coder-7B-Instruct"
  "Qwen/Qwen3-Coder-30B-A3B-Instruct"
  "/work/go25/share/model/Qwen2.5_Coder_7B_grpo_0116/checkpoint-68"
  "/work/go25/share/model/Qwen2.5_Coder_7B_drgrpo_0116/checkpoint-68"
  "/work/go25/share/model/Qwen2.5_Coder_7B_dapo_0116/checkpoint-68"
  "/work/go25/share/model/Qwen2.5_Coder_7B_grpo_0123_code/checkpoint-68"
  "/work/go25/share/model/Qwen2.5_Coder_7B_grpo_0123_md/checkpoint-68"
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen2.5_Coder_7B_grpo_reward30b/global_step_194"
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen2.5_Coder_7B_grpo_reward7b/global_step_194"
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen3.5_4B_grpo_reward30b/global_step_194"
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen3.5_4B_grpo_reward7b/global_step_194"
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen2.5_Coder_7B_grpo_reward80b/global_step_194"
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen3.5_4B_grpo_reward80b/global_step_194"
)

OUTPUT_DIRS=(
  "evaluation_base2"
  "evaluation_seed_base2"
  "evaluation_qwen2.5_base2"
  "evaluation"
  "grpo_0116"
  "drgrpo_0116"
  "dapo_0116"
  "grpo_0123_code"
  "grpo_0123_md"
  "evaluation_qwen2.5_grpo_reward80b"
  "evaluation_qwen3.5_grpo_reward80b"
)

for model_index in "${!MODEL_NAMES[@]}"; do
  model_name="${MODEL_NAMES[$model_index]}"
  output_dir="${OUTPUT_DIRS[$model_index]}"

  echo "Running translations with ${model_name}; output directory: ${output_dir}"

  for src_lang in "${LANGUAGES[@]}"; do
    for tgt_lang in "${LANGUAGES[@]}"; do
      if [[ "$src_lang" == "$tgt_lang" ]]; then
        continue
      fi

      src_path="${BENCHMARK_DIR}/${src_lang}/data/humaneval_${src_lang}.jsonl.gz"
      tgt_path="${BENCHMARK_DIR}/${tgt_lang}/data/humaneval_${tgt_lang}.jsonl.gz"
      output_file="${BENCHMARK_DIR}/${src_lang}/${output_dir}/humaneval_${src_lang}_to_${tgt_lang}.jsonl"

      python3 -m "$PYTHON_MODULE" \
        --model-name-or-path "$model_name" \
        --src-path "$src_path" \
        --tgt-path "$tgt_path" \
        --language-src-type "$src_lang" \
        --language-tgt-type "$tgt_lang" \
        --batch-size "$BATCH_SIZE" \
        --output-file "$output_file" \
        --samples-per-problem "$SAMPLES_PER_PROBLEM"
    done
  done
done
