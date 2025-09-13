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

echo "Evaluating input_dir='$INPUT_DIR' across languages: ${langs[*]}"

# Collect produced result files to aggregate later
declare -a RESULTS_FILES=()

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
    container_problem="/workspace/codegeex/benchmark/humaneval-x/$lang/data/humaneval_${lang}.jsonl.gz"

    echo "  -> Evaluating: $host_input"
    set -x
    "$CTR" run \
      -B "$REPO_ROOT":/workspace \
      "$SIF_PATH" \
      --input_file "$container_input" \
      --problem_file "$container_problem" \
      --tmp_dir /workspace/codegeex/benchmark/humaneval-x \
      --n_workers "$N_WORKERS" \
      --timeout "$TIMEOUT"
    set +x

    # Record expected results file path on host
    host_results_file="${host_input%.jsonl}_results.jsonl"
    if [[ -f "$host_results_file" ]]; then
      RESULTS_FILES+=("$host_results_file")
    else
      # Also check for gz in case inputs were gzipped
      if [[ -f "${host_results_file}.gz" ]]; then
        RESULTS_FILES+=("${host_results_file}.gz")
      fi
    fi
  done
done

echo "All evaluations finished. Aggregating to CSV..."

if [[ ${#RESULTS_FILES[@]} -eq 0 ]]; then
  echo "Warning: no *_results.jsonl files found to aggregate." 1>&2
  exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is required on host to build CSV '$CSV_OUT'" 1>&2
  exit 5
fi

# Build a single CSV from all results files
python3 - "$CSV_OUT" "${RESULTS_FILES[@]}" << 'PY'
import sys, os, csv, json, gzip

def iter_jsonl(path):
    if path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if line:
                    yield json.loads(line)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if line:
                    yield json.loads(line)

def detect_lang(path):
    # try to get segment after 'humaneval-x'
    parts = os.path.normpath(path).split(os.sep)
    if 'humaneval-x' in parts:
        idx = parts.index('humaneval-x')
        if idx + 1 < len(parts):
            return parts[idx+1]
    for cand in ('cpp','go','java','js','python','rust'):
        if f"{os.sep}{cand}{os.sep}" in path:
            return cand
    return ''

out_csv = sys.argv[1]
in_files = sys.argv[2:]

fields = [
    'language','task_id','completion_id','passed','result',
    'input_file','results_file'
]

with open(out_csv, 'w', newline='', encoding='utf-8') as wf:
    writer = csv.DictWriter(wf, fieldnames=fields)
    writer.writeheader()
    for rf in in_files:
        lang = detect_lang(rf)
        for rec in iter_jsonl(rf):
            writer.writerow({
                'language': lang,
                'task_id': rec.get('task_id',''),
                'completion_id': rec.get('completion_id',''),
                'passed': rec.get('passed',''),
                'result': rec.get('result',''),
                'input_file': rf.replace('_results.jsonl','').replace('.gz',''),
                'results_file': rf,
            })

print(f"Wrote CSV: {out_csv}")
PY

echo "CSV written to: $CSV_OUT"
