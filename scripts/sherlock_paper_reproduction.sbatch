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
# Common overrides:
#   PROJ=/scratch/users/jdehoney/eqnn_thesis \
#   MODULE_PYTHON=python/3.11.9 \
#   sbatch --array=6-8 scripts/sherlock_paper_reproduction.sbatch
#   VENV_DIR="$HOME/venvs/eqnn" sbatch scripts/sherlock_paper_reproduction.sbatch
#
# Assumptions:
# - The repo is already checked out on Sherlock.
# - Python and dependencies are available through a loaded module and/or venv.
# - This script is intended to run from the repo checkout itself; no editable
#   install is required because `python -m eqnn` is executed from the repo root.

PROJ="${PROJ:-/scratch/users/jdehoney/eqnn_thesis}"
PROJECT_ROOT="${PROJECT_ROOT:-$PROJ}"
MODULE_PYTHON="${MODULE_PYTHON:-python/3.11.9}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-}"
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

mkdir -p "$PROJECT_ROOT/logs/slurm" "$OUTPUT_ROOT"
cd "$PROJECT_ROOT"

if command -v module >/dev/null 2>&1; then
  module purge
  module load "$MODULE_PYTHON"
fi

if [[ -n "${VENV_DIR:-}" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
fi

# Keep BLAS/OpenMP from oversubscribing the allocated cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "[$(date)] Starting paper reproduction"
echo "Project root: $PROJECT_ROOT"
echo "Python: $("$PYTHON_BIN" --version 2>&1)"
echo "Num qubits: $NUM_QUBITS"
echo "Train sizes: $TRAIN_SIZES"
echo "Random seeds: $RANDOM_SEEDS"
echo "Epochs: $EPOCHS"
echo "Output dir: $RUN_OUTPUT_DIR"
echo "SLURM job: ${SLURM_JOB_ID:-unknown}, array task: ${SLURM_ARRAY_TASK_ID:-none}"

"$PYTHON_BIN" -m eqnn run-paper-reproduction \
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
