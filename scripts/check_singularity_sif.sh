#!/usr/bin/env bash

# Quick smoke test for humaneval-x Singularity/Apptainer image.
# Verifies core toolchain availability and Python import path inside the container.
#
# Usage:
#   scripts/check_singularity_sif.sh [sif_path]
#
# Examples:
#   scripts/check_singularity_sif.sh ./humanevalx.sif

set -euo pipefail

SIF_PATH="${1:-./humanevalx.sif}"

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

if [[ ! -f "$SIF_PATH" ]]; then
  echo "[FAIL] SIF not found: $SIF_PATH" 1>&2
  exit 2
fi

if command -v singularity >/dev/null 2>&1; then
  CTR="singularity"
elif command -v apptainer >/dev/null 2>&1; then
  CTR="apptainer"
else
  echo "[FAIL] neither 'singularity' nor 'apptainer' found in PATH" 1>&2
  exit 3
fi

pass() { echo "[PASS] $*"; }
fail() { echo "[FAIL] $*"; EXIT_CODE=1; }

EXIT_CODE=0

echo "Running smoke checks on: $SIF_PATH"
echo "Repo bound into container as /workspace: $REPO_ROOT"

# 1) Python imports and codegeex availability
if "$CTR" exec -B "$REPO_ROOT":/workspace "$SIF_PATH" bash -lc '
  set -e
  export PYTHONPATH=/workspace:$PYTHONPATH
  python3 - <<PY
import sys
print("python_version:", sys.version.split()[0])
import numpy, regex, fire
import codegeex
print("codegeex_path:", codegeex.__file__)
PY
' >/dev/null; then
  pass "Python env and codegeex import"
else
  fail "Python env and codegeex import"
fi

# 2) Node.js and js-md5 presence (with diagnostics)
if "$CTR" exec -B "$REPO_ROOT":/workspace "$SIF_PATH" bash -lc '
  set -e
  node -v >/dev/null
  NPM_ROOT=$(npm root -g 2>/dev/null || true)
  echo "[node] NODE_PATH(before)=${NODE_PATH:-}"
  echo "[node] npm root -g=${NPM_ROOT}"
  if [ -n "$NPM_ROOT" ] && [ -d "$NPM_ROOT" ]; then
    export NODE_PATH="$NPM_ROOT:${NODE_PATH}"
  else
    export NODE_PATH="/usr/local/lib/node_modules:${NODE_PATH}"
  fi
  echo "[node] NODE_PATH(after)=${NODE_PATH}"
  if [ -d "$NPM_ROOT/js-md5" ] || [ -d "/usr/local/lib/node_modules/js-md5" ]; then
    echo "[node] js-md5 directory found"
  else
    echo "[node] js-md5 directory NOT found" >&2
    exit 10
  fi
  node - <<"NODE" >/dev/null
require("js-md5");
console.log("ok");
NODE
' >/dev/null; then
  pass "Node.js and js-md5 module"
else
  fail "Node.js and js-md5 module"
fi

# 3) Go toolchain
if "$CTR" exec -B "$REPO_ROOT":/workspace "$SIF_PATH" bash -lc 'go version' >/dev/null 2>&1; then
  pass "Go toolchain"
else
  fail "Go toolchain"
fi

# 4) Java toolchain
if "$CTR" exec -B "$REPO_ROOT":/workspace "$SIF_PATH" bash -lc 'javac -version >/dev/null 2>&1 && java -version >/dev/null 2>&1'; then
  pass "Java toolchain"
else
  fail "Java toolchain"
fi

# 5) C++ compiler
if "$CTR" exec -B "$REPO_ROOT":/workspace "$SIF_PATH" bash -lc 'g++ --version' >/dev/null 2>&1; then
  pass "G++ compiler"
else
  fail "G++ compiler"
fi

# 6) Rust toolchain
if "$CTR" exec -B "$REPO_ROOT":/workspace "$SIF_PATH" bash -lc 'rustc --version >/dev/null 2>&1 && cargo --version >/dev/null 2>&1'; then
  pass "Rust toolchain"
else
  fail "Rust toolchain"
fi

# 7) Evaluator runscript help (via `run`)
if "$CTR" run -B "$REPO_ROOT":/workspace "$SIF_PATH" --help >/dev/null 2>&1; then
  pass "Evaluator runscript reachable (--help)"
else
  fail "Evaluator runscript reachable (--help)"
fi

if [[ $EXIT_CODE -eq 0 ]]; then
  echo "All smoke checks passed."
else
  echo "One or more checks failed. See messages above." 1>&2
fi

exit $EXIT_CODE
