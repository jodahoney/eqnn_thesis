#!/bin/bash
#SBATCH --job-name=eqnn-paper
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=6-13
#SBATCH --output=logs/slurm/eqnn-paper-%A_%a.out
#SBATCH --error=logs/slurm/eqnn-paper-%A_%a.err

set -euo pipefail

# Usage:
#   sbatch scripts/sherlock_paper_reproduction.sbatch
#
# Common overrides:
#   MODULE_PYTHON=python/3.11.9 sbatch --array=6-8 scripts/sherlock_paper_reproduction.sbatch
#   VENV_DIR="$HOME/venvs/eqnn" sbatch scripts/sherlock_paper_reproduction.sbatch
#   INSTALL_EDITABLE=1 sbatch scripts/sherlock_paper_reproduction.sbatch
#
# Assumptions:
# - The repo is already checked out on Sherlock.
# - Dependencies are available either through a loaded module + venv,
#   or by setting INSTALL_EDITABLE=1 for a per-job editable install.

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SIZES="${TRAIN_SIZES:-2 4 6 8 10 12}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-750}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
GRADIENT_BACKEND="${GRADIENT_BACKEND:-exact}"
INITIALIZATION_STRATEGY="${INITIALIZATION_STRATEGY:-noisy_current}"
INITIALIZATION_NOISE_SCALE="${INITIALIZATION_NOISE_SCALE:-0.05}"
EIGENSOLVER="${EIGENSOLVER:-auto}"
NUM_QUBITS="${NUM_QUBITS:-${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/reproduction/sherlock}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-$OUTPUT_ROOT/paper_reproduction_n${NUM_QUBITS}}"

mkdir -p "$PROJECT_ROOT/logs/slurm" "$OUTPUT_ROOT"
cd "$PROJECT_ROOT"

if [[ -n "${MODULE_PYTHON:-}" ]]; then
  module load "$MODULE_PYTHON"
fi

if [[ -n "${VENV_DIR:-}" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
fi

if [[ "${INSTALL_EDITABLE:-0}" == "1" ]]; then
  "$PYTHON_BIN" -m pip install -e "$PROJECT_ROOT"
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
