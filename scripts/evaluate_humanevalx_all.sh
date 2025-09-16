#!/usr/bin/env bash

set -euo pipefail

# Evaluate HumanEval-X results for all languages inside a Singularity/Apptainer container.
#
# Usage:
#   scripts/evaluate_humanevalx_all.sh <input_dir> [sif_path] [n_workers] [timeout] [csv_out]
#
# Arguments:
#   input_dir  : Directory name that exists under each language dir, e.g.
#                codegeex/benchmark/humaneval-x/{cpp,go,java,js,python,rust}/<input_dir>
#                and contains *.jsonl files to evaluate.
#   sif_path   : Path to the built SIF image (default: ./humanevalx.sif)
#   n_workers  : Number of parallel workers (default: 64)
#   timeout    : Per-test timeout in seconds (default: 5)
#   csv_out    : Path to aggregate CSV output (default: <repo>/humanevalx_results.csv)
#
# Notes:
# - This script assumes the repository is checked out and that the SIF is already built.
# - It binds the following into the container:
#     /workspace -> repository root (for Python imports and runscript expectations)

usage() {
  echo "Usage: $0 <input_dir> [sif_path] [n_workers] [timeout] [csv_out]" 1>&2
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

    if [[ ! -f "$HUMX_DIR/$problem_lang/data/humaneval_${problem_lang}.jsonl.gz" ]]; then
      echo "[skip] $lang: problem file for target '$problem_lang' not found: $HUMX_DIR/$problem_lang/data/humaneval_${problem_lang}.jsonl.gz" 1>&2
      continue
    fi

    echo "  -> Evaluating: $host_input"
    log_file="${host_input%.jsonl}.log"
    set -x
    "$CTR" run \
      -B "$REPO_ROOT":/workspace \
      "$SIF_PATH" \
      --input_file "$container_input" \
      --problem_file "$container_problem" \
      --tmp_dir /workspace/codegeex/benchmark/humaneval-x \
      --n_workers "$N_WORKERS" \
      --timeout "$TIMEOUT" \
      2>&1 | tee "$log_file"
    set +x

    # Record expected results file path on host
  done
done

echo "All evaluations finished. Building CSV from evaluator stdout..."

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is required on host to build CSV '$CSV_OUT'" 1>&2
  exit 5
fi

# Header for the CSV
echo "language,input_file,pass@1,pass@10,pass@100" > "$CSV_OUT"

# Re-run detections to append rows for entries we processed in this session.
# We tracked evaluated files implicitly; reconstruct by scanning per-language dir.
for lang in "${langs[@]}"; do
  IN_DIR="$HUMX_DIR/$lang/$INPUT_DIR"
  shopt -s nullglob
  files=("$IN_DIR"/*.jsonl)
  shopt -u nullglob
  if [[ ${#files[@]} -eq 0 ]]; then
    continue
  fi
  for host_input in "${files[@]}"; do
    base=$(basename -- "$host_input")
    # For each input, find the latest corresponding log captured beside results if present
    # Our execution didn't persist logs; so extract pass@k by re-parsing the most recent run output if available.
    # As a fallback, parse the results file's sibling stdout if user used tee externally.
    # If not found, attempt a heuristic: look into a cached log under /tmp if created earlier.
    # Since reliable logs may not exist, we instead parse the evaluator-produced *_results.jsonl's directory
    # marker to decide row presence, but leave metrics blank if stdout unavailable.
    # Try to locate a saved run log next to results file with .log extension
    run_log="${host_input%.jsonl}.log"
    pass1=""; pass10=""; pass100=""
    if [[ -f "$run_log" ]]; then
      # Parse pass@k dict from evaluator stdout log
      csv_vals=$(python3 - "$run_log" << 'PY'
import sys, ast, re
text = open(sys.argv[1], 'r', encoding='utf-8', errors='ignore').read().splitlines()
cand=None
for line in reversed(text):
    if 'pass@' in line and '{' in line and '}' in line:
        m=re.search(r'\{.*\}', line)
        cand=m.group(0) if m else line.strip()
        break
if cand is None:
    print(',,')
    sys.exit(0)
try:
    d=ast.literal_eval(cand)
except Exception:
    print(',,')
    sys.exit(0)
def g(k):
    v=d.get(k, '')
    return f"{v:.6f}" if isinstance(v,float) else ('' if v is None else str(v))
print(f"{g('pass@1')},{g('pass@10')},{g('pass@100')}")
PY
)
      echo "$lang,$host_input,$csv_vals" >> "$CSV_OUT"
    else
      # No log found; write row with empty metrics
      echo "$lang,$host_input,,," >> "$CSV_OUT"
    fi
  done
done

echo "CSV written to: $CSV_OUT"
