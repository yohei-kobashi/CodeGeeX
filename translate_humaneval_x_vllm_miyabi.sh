#!/bin/bash
#PBS -q regular-g
#PBS -l select=1:mpiprocs=1
#PBS -W group_list=go25
#PBS -j oe
#PBS -o translate_humaneval_x_vllm_miyabi.log

set -euo pipefail

module purge
module load nvidia/25.9
module load singularity/4.2.1

export CC=gcc
export CXX=g++
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="9.0"

cd "${PBS_O_WORKDIR}"

export MASTER_ADDR=$(head -1 "${PBS_NODEFILE}")
echo "Server will start on ${MASTER_ADDR}:18000+"

singularity exec --nv --bind /work/go25:/work/go25 /work/gj26/share/sif/vllm_v0.21.0.sif bash <<'EOF'
set -euo pipefail

PYTHON_MODULE="codegeex.benchmark.humaneval-x.translate_humaneval_x_vllm"
BENCHMARK_DIR="codegeex/benchmark/humaneval-x"
BATCH_SIZE=205
MAX_MODEL_LEN=8192
MAX_TOKENS=4096
SAMPLES_PER_PROBLEM=5
REQUEST_TIMEOUT=900
SERVER_REQUEST_RETRIES=3
SKIP_EXISTING=1
SERVER_HOST="0.0.0.0"
CLIENT_HOST="127.0.0.1"
BASE_PORT=18000
LOG_DIR="logs/humaneval_x_vllm_miyabi"

mkdir -p "$LOG_DIR"

echo "[$(date)] Work directory: $(pwd)"
echo "[$(date)] Benchmark directory: $(realpath "$BENCHMARK_DIR")"
echo "[$(date)] SKIP_EXISTING=${SKIP_EXISTING}"

skip_existing_args=(--skip-existing)
if [[ "$SKIP_EXISTING" == "0" || "$SKIP_EXISTING" == "false" || "$SKIP_EXISTING" == "no" ]]; then
  skip_existing_args=()
  echo "[$(date)] Existing output files will be overwritten."
fi

LANGUAGES=(cpp go java js python rust)

MODEL_NAMES=(
  # "Qwen/Qwen2.5-Coder-7B-Instruct"
  # "Qwen/Qwen3.5-4B"
  # "/work/go25/share/model/code_trans_grpo_model_0409/Qwen2.5_Coder_7B_grpo_reward7b/global_step_194"
  # "/work/go25/share/model/code_trans_grpo_model_0409/Qwen2.5_Coder_7B_grpo_reward30b/global_step_194"
  # "/work/go25/share/model/code_trans_grpo_model_0409/Qwen2.5_Coder_7B_grpo_reward80b/global_step_194"
  # "/work/go25/share/model/code_trans_grpo_model_0409/Qwen3.5_4B_grpo_reward7b/global_step_194"
  # "/work/go25/share/model/code_trans_grpo_model_0409/Qwen3.5_4B_grpo_reward30b/global_step_194"
  # "/work/go25/share/model/code_trans_grpo_model_0409/Qwen3.5_4B_grpo_reward80b/global_step_194"
  # "/work/go25/share/model/code_trans_grpo_model_0409/Qwen2.5_Coder_7B_grpo_reward30b_v2/global_step_100"
  # "/work/go25/share/model/code_trans_grpo_model_0409/Qwen2.5_Coder_7B_grpo_reward30b_v2/global_step_194"
  # "/work/go25/share/model/code_trans_grpo_model_0409/Qwen3.5_4B_grpo_reward30b_v2/global_step_100"
  # "/work/go25/share/model/code_trans_grpo_model_0409/Qwen3.5_4B_grpo_reward30b_v2/global_step_194"
  # "/work/go25/share/model/code_trans_grpo_model_0409/Qwen3.5_4B_drgrpo_reward30b/global_step_100"
  # "/work/go25/share/model/code_trans_grpo_model_0409/Qwen3.5_4B_drgrpo_reward30b/global_step_194"
  # "/work/go25/share/model/code_trans_grpo_model_0409/Qwen3.5_4B_dapo_reward30b/global_step_100"
  # /work/go25/share/model/code_trans_grpo_model_0409/Qwen3.5_4B_dapo_reward30b_ors/global_step_100
  # /work/go25/share/model/code_trans_grpo_model_0702/Qwen3.5_4B_dapo_0702_reward35b_think/global_step_100
  # /work/go25/share/model/code_trans_grpo_model_0702/Qwen3.5_4B_dapo_0702_reward35b_nothink/global_step_100
  /work/go25/share/model/code_trans_grpo_model_0702/Qwen3.5_9B_dapo_0702_reward35b_think/global_step_100
  /work/go25/share/model/code_trans_grpo_model_0702/Qwen3.5_9B_dapo_0702_reward35b_nothink/global_step_100
)

OUTPUT_DIRS=(
  # "evaluation_qwen2.5"
  # "evaluation_qwen3.5"
  # "evaluation_qwen2.5_grpo_reward7b"
  # "evaluation_qwen2.5_grpo_reward30b"
  # "evaluation_qwen2.5_grpo_reward80b"
  # "evaluation_qwen3.5_grpo_reward7b"
  # "evaluation_qwen3.5_grpo_reward30b"
  # "evaluation_qwen3.5_grpo_reward80b"
  # "evaluation_qwen2.5_grpo_reward30b_v2_100"
  # "evaluation_qwen2.5_grpo_reward30b_v2_194"
  # "evaluation_qwen3.5_grpo_reward80b_v2_100"
  # "evaluation_qwen3.5_grpo_reward80b_v2_194"
  # "evaluation_qwen3.5_drgrpo_reward30b_100"
  # "evaluation_qwen3.5_drgrpo_reward30b_194"
  # "evaluation_qwen3.5_dapo_reward30b_100"
  # "evaluation_qwen3.5_dapo_reward30b_ors_100"
  # "evaluation_qwen3.5_dapo_0702_reward35b_think_100"
  # "evaluation_qwen3.5_dapo_0702_reward35b_nothink_100"
  "evaluation_qwen3.5_9B_dapo_0702_reward35b_think_100"
  "evaluation_qwen3.5_9B_dapo_0702_reward35b_nothink_100"
)

if [[ "${#MODEL_NAMES[@]}" -ne "${#OUTPUT_DIRS[@]}" ]]; then
  echo "MODEL_NAMES and OUTPUT_DIRS must have the same length." >&2
  echo "MODEL_NAMES: ${#MODEL_NAMES[@]}, OUTPUT_DIRS: ${#OUTPUT_DIRS[@]}" >&2
  exit 1
fi

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

sanitize_name() {
  echo "$1" | tr '/:.' '___' | tr -c 'A-Za-z0-9_-' '_'
}

wait_for_server() {
  local url="$1"
  local timeout_seconds="$2"
  local start_time
  start_time=$(date +%s)

  while true; do
    if python3 - "$url" <<'PY'
import sys
import urllib.request

url = sys.argv[1].rstrip("/") + "/models"
try:
    with urllib.request.urlopen(url, timeout=5) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
    then
      return 0
    fi

    if (( $(date +%s) - start_time >= timeout_seconds )); then
      echo "Timed out waiting for vLLM server at ${url}" >&2
      return 1
    fi

    sleep 10
  done
}

stop_server() {
  if [[ -n "${VLLM_SERVER_PID:-}" ]]; then
    if kill -0 "$VLLM_SERVER_PID" 2>/dev/null; then
      echo "Stopping vLLM server pid=${VLLM_SERVER_PID}"
      kill "$VLLM_SERVER_PID" 2>/dev/null || true
      wait "$VLLM_SERVER_PID" 2>/dev/null || true
    fi
    unset VLLM_SERVER_PID
  fi
}

trap stop_server EXIT

for model_index in "${!MODEL_NAMES[@]}"; do
  model_name="${MODEL_NAMES[$model_index]}"
  output_dir="${OUTPUT_DIRS[$model_index]}"
  served_model_name="$model_name"
  read -r -a sampling_args <<< "$(sampling_args_for_model "$model_name")"
  port=$((BASE_PORT + model_index))
  server_url="http://${CLIENT_HOST}:${port}/v1"
  safe_model_name="$(sanitize_name "$output_dir")"
  server_log="${LOG_DIR}/vllm_${model_index}_${safe_model_name}.log"

  echo "[$(date)] Starting vLLM server for ${model_name} on ${MASTER_ADDR:-localhost}:${port}; output directory: ${output_dir}; sampling args: ${sampling_args[*]}"
  vllm serve "$model_name" \
    --host "$SERVER_HOST" \
    --port "$port" \
    --served-model-name "$served_model_name" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --kv-cache-dtype fp8 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 131072 \
    --max-num-seqs "$BATCH_SIZE" \
    --enforce-eager \
    > "$server_log" 2>&1 &
  VLLM_SERVER_PID=$!

  wait_for_server "$server_url" 1800
  echo "[$(date)] vLLM server is ready: ${server_url}; pid=${VLLM_SERVER_PID}; log=${server_log}"

  for src_lang in "${LANGUAGES[@]}"; do
    for tgt_lang in "${LANGUAGES[@]}"; do
      if [[ "$src_lang" == "$tgt_lang" ]]; then
        continue
      fi

      src_path="${BENCHMARK_DIR}/${src_lang}/data/humaneval_${src_lang}.jsonl.gz"
      tgt_path="${BENCHMARK_DIR}/${tgt_lang}/data/humaneval_${tgt_lang}.jsonl.gz"
      output_file="${BENCHMARK_DIR}/${src_lang}/${output_dir}/humaneval_${src_lang}_to_${tgt_lang}.jsonl"
      output_abs_path="$(realpath -m "$output_file")"

      echo "[$(date)] Running ${model_name}: ${src_lang} -> ${tgt_lang}; output=${output_file}; abs=${output_abs_path}"
      python3 -u -m "$PYTHON_MODULE" \
        --model-name-or-path "$served_model_name" \
        --tokenizer-name-or-path "$model_name" \
        --src-path "$src_path" \
        --tgt-path "$tgt_path" \
        --language-src-type "$src_lang" \
        --language-tgt-type "$tgt_lang" \
        --max-tokens "$MAX_TOKENS" \
        --batch-size "$BATCH_SIZE" \
        --output-file "$output_file" \
        --samples-per-problem "$SAMPLES_PER_PROBLEM" \
        --server-url "$server_url" \
        --request-timeout "$REQUEST_TIMEOUT" \
        --server-request-retries "$SERVER_REQUEST_RETRIES" \
        "${skip_existing_args[@]}" \
        "${sampling_args[@]}"
      echo "[$(date)] Finished ${model_name}: ${src_lang} -> ${tgt_lang}"
    done
  done

  echo "[$(date)] Finished all translations for ${model_name}"
  stop_server
done

EOF
