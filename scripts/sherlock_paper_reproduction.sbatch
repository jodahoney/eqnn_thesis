#!/bin/bash
#SBATCH --job-name=eqnn-paper
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=6-13
#SBATCH --output=/scratch/users/jdehoney/eqnn_thesis/logs/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/users/jdehoney/eqnn_thesis/logs/slurm/%x_%A_%a.err

set -euo pipefail

# Usage:
#   mkdir -p /scratch/users/jdehoney/eqnn_thesis/logs/slurm
#   cd /scratch/users/jdehoney/eqnn_thesis
#   sbatch scripts/sherlock_paper_reproduction.sbatch
#
# Optional overrides:
#   PROJ=/scratch/users/jdehoney/eqnn_thesis \
#   VENV_DIR=/scratch/users/jdehoney/venvs/eqnn_py312 \
#   MODULE_PYTHON=python/3.12.1 \
#   MODULE_OPENBLAS=openblas/0.3.28 \
#   sbatch --array=6-8 scripts/sherlock_paper_reproduction.sbatch

PROJ="${PROJ:-/scratch/users/jdehoney/eqnn_thesis}"
PROJECT_ROOT="${PROJECT_ROOT:-$PROJ}"

MODULE_PYTHON="${MODULE_PYTHON:-python/3.12.1}"
MODULE_OPENBLAS="${MODULE_OPENBLAS:-openblas/0.3.28}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-/scratch/users/jdehoney/venvs/eqnn_py312}"

TRAIN_SIZES="${TRAIN_SIZES:-2 4 6 8 10 12}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-750}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
GRADIENT_BACKEND="${GRADIENT_BACKEND:-exact}"
INITIALIZATION_STRATEGY="${INITIALIZATION_STRATEGY:-noisy_current}"
INITIALIZATION_NOISE_SCALE="${INITIALIZATION_NOISE_SCALE:-0.05}"
EIGENSOLVER="${EIGENSOLVER:-auto}"

NUM_QUBITS="${NUM_QUBITS:-${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}}"
RUN_ID="${RUN_ID:-paper_reproduction_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/reproduction/$RUN_ID}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-$OUTPUT_ROOT/paper_reproduction_n${NUM_QUBITS}}"

mkdir -p "$PROJECT_ROOT/logs/slurm" "$OUTPUT_ROOT" "$RUN_OUTPUT_DIR"
cd "$PROJECT_ROOT"

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

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: expected venv at $VENV_DIR but did not find $VENV_DIR/bin/activate"
    exit 1
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "[$(date)] Starting paper reproduction"
echo "Host: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Working dir: $(pwd)"
echo "Python executable: $(command -v "$PYTHON_BIN")"
echo "Python version: $("$PYTHON_BIN" --version 2>&1)"
echo "Num qubits: $NUM_QUBITS"
echo "Train sizes: $TRAIN_SIZES"
echo "Random seeds: $RANDOM_SEEDS"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Gradient backend: $GRADIENT_BACKEND"
echo "Initialization strategy: $INITIALIZATION_STRATEGY"
echo "Initialization noise scale: $INITIALIZATION_NOISE_SCALE"
echo "Eigensolver: $EIGENSOLVER"
echo "Output dir: $RUN_OUTPUT_DIR"
echo "SLURM job: ${SLURM_JOB_ID:-unknown}"
echo "SLURM array task: ${SLURM_ARRAY_TASK_ID:-none}"
echo "SLURM cpus-per-task: ${SLURM_CPUS_PER_TASK:-unknown}"

python - <<'PY'
import sys
print("sys.executable =", sys.executable)
import numpy, scipy
print("numpy =", numpy.__version__)
print("scipy =", scipy.__version__)
import eqnn
print("eqnn import ok")
PY

srun "$PYTHON_BIN" -m eqnn run-paper-reproduction \
  --num-qubits "$NUM_QUBITS" \
  --train-sizes $TRAIN_SIZES \
  --random-seeds $RANDOM_SEEDS \
  --epochs "$EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --gradient-backend "$GRADIENT_BACKEND" \
  --initialization-strategy "$INITIALIZATION_STRATEGY" \
  --initialization-noise-scale "$INITIALIZATION_NOISE_SCALE" \
  --eigensolver "$EIGENSOLVER" \
  --output-dir "$RUN_OUTPUT_DIR"

echo "[$(date)] Completed paper reproduction for N=$NUM_QUBITS"