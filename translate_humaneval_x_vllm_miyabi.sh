#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -W group_list=go25
#PBS -j oe
module purge
export CUDA_VISIBLE_DEVICES=0
module load nvidia/25.9
module load singularity/4.2.1
singularity exec --nv --bind /work/go25:/work/go25 /work/gj26/share/sif/vllm_v0.21.0.sif bash <<'EOF'
cd CodeGeeX

PYTHON_MODULE="codegeex.benchmark.humaneval-x.translate_humaneval_x_vllm"
BENCHMARK_DIR="codegeex/benchmark/humaneval-x"
BATCH_SIZE=128
MAX_TOKENS=4096
SAMPLES_PER_PROBLEM=5

LANGUAGES=(cpp go java js python rust)

MODEL_NAMES=(
  "Qwen/Qwen2.5-Coder-7B-Instruct"
  "Qwen/Qwen3.5-4B"
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen2.5_Coder_7B_grpo_reward7b/global_step_194"
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen2.5_Coder_7B_grpo_reward30b/global_step_194"
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen2.5_Coder_7B_grpo_reward80b/global_step_194"
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen3.5_4B_grpo_reward7b/global_step_194"
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen3.5_4B_grpo_reward30b/global_step_194"
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen3.5_4B_grpo_reward80b/global_step_194"
)

OUTPUT_DIRS=(
  "evaluation_qwen2.5"
  "evaluation_qwen3.5"
  "evaluation_qwen2.5_grpo_reward7b"
  "evaluation_qwen2.5_grpo_reward30b"
  "evaluation_qwen2.5_grpo_reward80b"
  "evaluation_qwen3.5_grpo_reward7b"
  "evaluation_qwen3.5_grpo_reward30b"
  "evaluation_qwen3.5_grpo_reward80b"
)

sampling_args_for_model() {
  local model_name="$1"

  case "$model_name" in
    *Qwen3.5*|*qwen3.5*)
      echo "--do-sample true --temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repetition-penalty 1.0"
      ;;
    *Qwen2.5-Coder-7B-Instruct*|*Qwen2.5_Coder_7B*|*qwen2.5_coder_7b*)
      echo "--do-sample true --temperature 0.7 --top-p 0.8 --top-k 20 --repetition-penalty 1.1"
      ;;
    *)
      echo "--do-sample true"
      ;;
  esac
}

for model_index in "${!MODEL_NAMES[@]}"; do
  model_name="${MODEL_NAMES[$model_index]}"
  output_dir="${OUTPUT_DIRS[$model_index]}"
  read -r -a sampling_args <<< "$(sampling_args_for_model "$model_name")"

  echo "Running translations with ${model_name}; output directory: ${output_dir}; sampling args: ${sampling_args[*]}"

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
        --max-tokens "$MAX_TOKENS" \
        --batch-size "$BATCH_SIZE" \
        --output-file "$output_file" \
        --samples-per-problem "$SAMPLES_PER_PROBLEM" \
        "${sampling_args[@]}"
    done
  done
done

EOF
