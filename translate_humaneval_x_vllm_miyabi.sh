#!/bin/bash
#PBS -q short-g
#PBS -l select=1
#PBS -W group_list=go25
#PBS -j oe
module purge
export CUDA_VISIBLE_DEVICES=0
module load nvidia/25.9
module load singularity/4.2.1
singularity shell --nv --bind /work/go25:/work/go25 /work/gj26/share/sif/vllm_v0.21.0.sif
cd CodeGeeX

PYTHON_MODULE="codegeex.benchmark.humaneval-x.translate_humaneval_x_vllm"
BENCHMARK_DIR="codegeex/benchmark/humaneval-x"
BATCH_SIZE=64
MAX_TOKENS=4096
SAMPLES_PER_PROBLEM=3

LANGUAGES=(cpp go java js python rust)

MODEL_NAMES=(
  "Qwen/Qwen3.5-4B"
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen2.5_Coder_7B_grpo_reward7b/global_step_194"
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen2.5_Coder_7B_grpo_reward30b/global_step_194"
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen2.5_Coder_7B_grpo_reward80b/global_step_194"
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen3.5_4B_grpo_reward7b/global_step_194"
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen3.5_4B_grpo_reward30b/global_step_194"
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen3.5_4B_grpo_reward80b/global_step_194"
)

OUTPUT_DIRS=(
  "evaluation_qwen3.5"
  "evaluation_qwen2.5_grpo_reward7b"
  "evaluation_qwen2.5_grpo_reward30b"
  "evaluation_qwen2.5_grpo_reward80b"
  "evaluation_qwen3.5_grpo_reward7b"
  "evaluation_qwen3.5_grpo_reward30b"
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
