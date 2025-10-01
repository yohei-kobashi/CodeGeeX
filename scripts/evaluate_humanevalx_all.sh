#!/usr/bin/env bash

set -euo pipefail

# Evaluate HumanEval-X results for all languages inside a Singularity/Apptainer container.
#
# Usage:
#   scripts/evaluate_humanevalx_all.sh <input_dir> [sif_path] [n_workers] [timeout] [csv_out] [target_langs]
#
# Arguments:
#   input_dir  : Directory name that exists under each language dir, e.g.
#                codegeex/benchmark/humaneval-x/{cpp,go,java,js,python,rust}/<input_dir>
#                and contains *.jsonl files to evaluate.
#   sif_path   : Path to the built SIF image (default: ./humanevalx.sif)
#   n_workers  : Number of parallel workers (default: 64)
#   timeout    : Per-test timeout in seconds (default: 5)
#   csv_out    : Path to aggregate CSV output (default: <repo>/humanevalx_results.csv)
#   target_langs : Optional comma-separated target languages to evaluate only
#                  (e.g., "python" or "java,cpp"). If omitted, evaluate all.
#
# Notes:
# - This script assumes the repository is checked out and that the SIF is already built.
# - It binds the following into the container:
#     /workspace -> repository root (for Python imports and runscript expectations)

usage() {
  echo "Usage: $0 <input_dir> [sif_path] [n_workers] [timeout] [csv_out] [target_langs]" 1>&2
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

INPUT_DIR="$1"
SIF_PATH="${2:-./humanevalx.sif}"
N_WORKERS="${3:-64}"
TIMEOUT="${4:-5}"

# Resolve repo root from this script location (scripts/..)
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
HUMX_DIR="$REPO_ROOT/codegeex/benchmark/humaneval-x"
CSV_OUT="${5:-$REPO_ROOT/humanevalx_results.csv}"
TARGET_LANGS_RAW="${6:-}"

if [[ ! -d "$HUMX_DIR" ]]; then
  echo "Error: humaneval-x dir not found: $HUMX_DIR" 1>&2
  exit 2
fi

if [[ ! -f "$SIF_PATH" ]]; then
  echo "Error: SIF image not found: $SIF_PATH" 1>&2
  exit 3
fi

# Choose container runner
if command -v singularity >/dev/null 2>&1; then
  CTR="singularity"
elif command -v apptainer >/dev/null 2>&1; then
  CTR="apptainer"
else
  echo "Error: neither 'singularity' nor 'apptainer' found in PATH" 1>&2
  exit 4
fi

langs=(cpp go java js python rust)

# Normalize language token to one of: cpp, go, java, js, python, rust
norm_lang() {
  local l=${1,,}
  case "$l" in
    js|javascript) echo js ;;
    py|python) echo python ;;
    c++|cplusplus|cpp) echo cpp ;;
    golang|go) echo go ;;
    java) echo java ;;
    rust) echo rust ;;
    *) echo "$l" ;;
  esac
}

# Detect translation src/target from filename
# Supports patterns like: foo-python-to-java-bar.jsonl, foo_python_to_java.jsonl, etc.
detect_src_tgt() {
  local base="$1"
  # Strip directory
  base=$(basename -- "$base")
  # Try hyphen or underscore variants
  if [[ "$base" =~ ([A-Za-z0-9\+]+)[-_]to[-_]+([A-Za-z0-9\+]+) ]]; then
    local src="$(norm_lang "${BASH_REMATCH[1]}")"
    local tgt="$(norm_lang "${BASH_REMATCH[2]}")"
    echo "$src $tgt"
    return 0
  fi
  # Not a translation filename
  echo " "
  return 1
}

echo "Evaluating input_dir='$INPUT_DIR' across languages: ${langs[*]}"

# Build target language filter (if provided)
declare -A TARGET_FILTER=()
if [[ -n "$TARGET_LANGS_RAW" ]]; then
  TARGET_LANGS_WS=${TARGET_LANGS_RAW//,/ }
  read -r -a _targets <<< "$TARGET_LANGS_WS"
  for t in "${_targets[@]}"; do
    tl=$(norm_lang "$t")
    [[ -n "$tl" ]] && TARGET_FILTER["$tl"]=1
  done
  if ((${#TARGET_FILTER[@]} > 0)); then
    echo "Filtering to target languages: ${!TARGET_FILTER[*]}"
  fi
fi

# Prepare CSV header (truncate if exists)
echo "language,input_file,pass@1,pass@10,pass@100" > "$CSV_OUT"

# Optional: bind host node_modules as global in container if present (for js-md5, etc.)
HOST_NODE_MODULES_DIR="$REPO_ROOT/node_modules"
EXTRA_BINDS=()
ENV_ARGS=(--env NODE_PATH=/usr/local/lib/node_modules:/usr/lib/node_modules:/workspace/node_modules)
if [[ -d "$HOST_NODE_MODULES_DIR/js-md5" || -d "$HOST_NODE_MODULES_DIR/node_modules/js-md5" ]]; then
  # Support either repo/node_modules/js-md5 or repo/node_modules/node_modules/js-md5
  if [[ -d "$HOST_NODE_MODULES_DIR/js-md5" ]]; then
    EXTRA_BINDS+=( -B "$HOST_NODE_MODULES_DIR:/usr/local/lib/node_modules" )
  else
    EXTRA_BINDS+=( -B "$HOST_NODE_MODULES_DIR/node_modules:/usr/local/lib/node_modules" )
  fi
fi


for lang in "${langs[@]}"; do
  DATA_FILE="$HUMX_DIR/$lang/data/humaneval_${lang}.jsonl.gz"
  IN_DIR="$HUMX_DIR/$lang/$INPUT_DIR"

  if [[ ! -d "$IN_DIR" ]]; then
    echo "[skip] $lang: input dir not found: $IN_DIR"
    continue
  fi
  if [[ ! -f "$DATA_FILE" ]]; then
    echo "[skip] $lang: data file not found: $DATA_FILE"
    continue
  fi

  shopt -s nullglob
  files=("$IN_DIR"/*.jsonl)
  shopt -u nullglob
  # Exclude any already-produced results files (e.g., *_results.jsonl)
  if [[ ${#files[@]} -gt 0 ]]; then
    filtered=()
    for f in "${files[@]}"; do
      case "$f" in
        *results.jsonl) continue ;;
      esac
      filtered+=("$f")
    done
    files=("${filtered[@]}")
  fi
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "[skip] $lang: no .jsonl files under $IN_DIR"
    continue
  fi

  echo "[lang=$lang] Found ${#files[@]} files under $IN_DIR"

  for host_input in "${files[@]}"; do
    base=$(basename -- "$host_input")
    container_input="/workspace/codegeex/benchmark/humaneval-x/$lang/$INPUT_DIR/$base"
    # Determine target language from filename if it's a translation; otherwise default to dir language
    read -r src_lang tgt_lang < <(detect_src_tgt "$base" || true)
    if [[ -n "$tgt_lang" ]]; then
      problem_lang="$tgt_lang"
      # Optional sanity check: warn if src_lang differs from directory name
      if [[ -n "$src_lang" && "$src_lang" != "$lang" ]]; then
        echo "[warn] filename src='$src_lang' differs from dir lang='$lang' for $host_input" 1>&2
      fi
    else
      problem_lang="$lang"
    fi

    container_problem="/workspace/codegeex/benchmark/humaneval-x/$problem_lang/data/humaneval_${problem_lang}.jsonl.gz"

    # If a target filter is specified, skip non-matching target languages
    if ((${#TARGET_FILTER[@]} > 0)) && [[ -z "${TARGET_FILTER[$problem_lang]:-}" ]]; then
      echo "  [skip] target '$problem_lang' not in filter"
      continue
    fi

    if [[ ! -f "$HUMX_DIR/$problem_lang/data/humaneval_${problem_lang}.jsonl.gz" ]]; then
      echo "[skip] $lang: problem file for target '$problem_lang' not found: $HUMX_DIR/$problem_lang/data/humaneval_${problem_lang}.jsonl.gz" 1>&2
      continue
    fi

    echo "  -> Evaluating: $host_input"
    log_file="${host_input%.jsonl}.log"
    # Capture evaluator CSV row from stdout; stream logs (stderr) to both console and log file
    set -x
    row=$("$CTR" run \
      "${ENV_ARGS[@]}" \
      -B "$REPO_ROOT":/workspace \
      "${EXTRA_BINDS[@]}" \
      "$SIF_PATH" \
      --input_file "$container_input" \
      --problem_file "$container_problem" \
      --tmp_dir /workspace/codegeex/benchmark/humaneval-x \
      --n_workers "$N_WORKERS" \
      --timeout "$TIMEOUT" \
      2> >(tee "$log_file" >&2))
    set +x
    # Append CSV row (fallback to empty metrics if nothing captured)
    if [[ -n "$row" ]]; then
      echo "$row" >> "$CSV_OUT"
    else
      echo "$problem_lang,$container_input,,," >> "$CSV_OUT"
    fi
  done
done

echo "CSV written to: $CSV_OUT"
