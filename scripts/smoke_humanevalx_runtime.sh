#!/usr/bin/env bash

# Wrapper to run smoke_humanevalx_runtime.py inside the SIF with proper binds/env.
# Usage:
#   scripts/smoke_humanevalx_runtime.sh [sif_path]

set -euo pipefail

SIF_PATH="${1:-codegeex/benchmark/humaneval-x/humanevalx.sif}"

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

if command -v singularity >/dev/null 2>&1; then
  CTR="singularity"
elif command -v apptainer >/dev/null 2>&1; then
  CTR="apptainer"
else
  echo "Error: neither singularity nor apptainer found" 1>&2
  exit 2
fi

ENV_ARGS=(--env NODE_PATH=/usr/local/lib/node_modules:/usr/lib/node_modules:/workspace/node_modules)

set -x
"$CTR" exec \
  "${ENV_ARGS[@]}" \
  -B "$REPO_ROOT":/workspace \
  "$SIF_PATH" \
  python3 /workspace/scripts/smoke_humanevalx_runtime.py
set +x

