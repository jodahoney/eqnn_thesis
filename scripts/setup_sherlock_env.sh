#!/bin/bash
set -euo pipefail

# Usage:
#   bash scripts/setup_sherlock_env.sh
#
# Optional overrides:
#   PROJ=/scratch/users/jdehoney/eqnn_thesis \
#   VENV_DIR=/scratch/users/jdehoney/venvs/eqnn_py312 \
#   MODULE_PYTHON=python/3.12.1 \
#   MODULE_OPENBLAS=openblas/0.3.28 \
#   bash scripts/setup_sherlock_env.sh

PROJ="${PROJ:-/scratch/users/jdehoney/eqnn_thesis}"
VENV_DIR="${VENV_DIR:-/scratch/users/jdehoney/venvs/eqnn_py312}"
MODULE_PYTHON="${MODULE_PYTHON:-python/3.12.1}"
MODULE_OPENBLAS="${MODULE_OPENBLAS:-openblas/0.3.28}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==> Project root: $PROJ"
echo "==> Venv dir:     $VENV_DIR"
echo "==> Python mod:   $MODULE_PYTHON"
echo "==> OpenBLAS mod: $MODULE_OPENBLAS"

cd "$PROJ"

if command -v ml >/dev/null 2>&1; then
    ml reset
    ml "$MODULE_PYTHON"
    ml "$MODULE_OPENBLAS"
elif command -v module >/dev/null 2>&1; then
    module reset
    module load "$MODULE_PYTHON"
    module load "$MODULE_OPENBLAS"
else
    echo "ERROR: neither 'ml' nor 'module' command is available"
    exit 1
fi

mkdir -p "$(dirname "$VENV_DIR")"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "==> Creating venv"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "==> Reusing existing venv"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .

echo "==> Running sanity checks"
python - <<'PY'
import sys
print("sys.executable =", sys.executable)

import numpy, scipy
print("numpy =", numpy.__version__)
print("scipy =", scipy.__version__)

import eqnn
print("eqnn import ok")
PY

echo "==> CLI check"
eqnn --help >/dev/null

echo "==> Done"