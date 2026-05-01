#!/bin/bash
#SBATCH --job-name=eqnn-depol-n7-qubits
#SBATCH --partition=normal
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-3
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/users/jdehoney/eqnn_thesis}"
VENV_DIR="${VENV_DIR:-/scratch/users/jdehoney/venvs/eqnn_py312}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODULE_PYTHON="${MODULE_PYTHON:-python/3.12.1}"
MODULE_OPENBLAS="${MODULE_OPENBLAS:-openblas/0.3.28}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/noisy_comparisons/depolarizing_n7_selected_qubits_array_${SLURM_ARRAY_JOB_ID}}"

mkdir -p "$PROJECT_ROOT/logs/slurm"
mkdir -p "$OUTPUT_ROOT"
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
    echo "ERROR: neither 'ml' nor 'module' is available"
    exit 1
fi

source "$VENV_DIR/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# ---------------------------------------------------------------------
# Array split:
#   MODEL_FAMILIES = su2_qcnn, hea_qcnn
#   TRAIN_SIZES = 12, 16
#
# Task mapping:
#   task 0: su2_qcnn, train_size=12
#   task 1: su2_qcnn, train_size=16
#   task 2: hea_qcnn, train_size=12
#   task 3: hea_qcnn, train_size=16
# ---------------------------------------------------------------------

MODEL_VALUES=(su2_qcnn hea_qcnn)
TRAIN_SIZE_VALUES=(12 16)

NUM_TRAIN_SIZES=${#TRAIN_SIZE_VALUES[@]}

TASK_ID="${SLURM_ARRAY_TASK_ID}"
MODEL_INDEX=$(( TASK_ID / NUM_TRAIN_SIZES ))
TRAIN_INDEX=$(( TASK_ID % NUM_TRAIN_SIZES ))

MODEL_FAMILY="${MODEL_VALUES[$MODEL_INDEX]}"
TRAIN_SIZE="${TRAIN_SIZE_VALUES[$TRAIN_INDEX]}"

NUM_QUBITS="${NUM_QUBITS:-7}"
EPOCHS_VALUES="${EPOCHS_VALUES:-10}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2}"

# Focused strengths around the observed depolarizing failure region.
NOISE_STRENGTH_VALUES="${NOISE_STRENGTH_VALUES:-0.01 0.03 0.05 0.08 0.1}"

NOISY_QUBIT_INDICES="${NOISY_QUBIT_INDICES:-0 1 2 3 4 5 6}"

RUN_TAG="selected_qubit_depolarizing_n${NUM_QUBITS}_${MODEL_FAMILY}_train${TRAIN_SIZE}"
RUN_OUTPUT_DIR="$OUTPUT_ROOT/$RUN_TAG"

mkdir -p "$RUN_OUTPUT_DIR"

echo "[$(date)] Starting n=7 selected-qubit depolarizing diagnostic"
echo "Host: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Working dir: $(pwd)"
echo "Python executable: $(which python3)"
echo "Python version: $(python3 -V)"
echo "SLURM job: ${SLURM_JOB_ID:-}"
echo "SLURM array job: ${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM array task: ${SLURM_ARRAY_TASK_ID:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "Output root: $OUTPUT_ROOT"
echo "Run output dir: $RUN_OUTPUT_DIR"
echo "Model family: $MODEL_FAMILY"
echo "Num qubits: $NUM_QUBITS"
echo "Train size: $TRAIN_SIZE"
echo "Epochs: $EPOCHS_VALUES"
echo "Seeds: $RANDOM_SEEDS"
echo "Noise strengths: $NOISE_STRENGTH_VALUES"
echo "Noisy qubit indices: $NOISY_QUBIT_INDICES"

"$PYTHON_BIN" - <<'PY'
import sys
print("sys.executable =", sys.executable)
import qiskit
import qiskit_aer
print("qiskit =", qiskit.__version__)
print("qiskit_aer =", qiskit_aer.__version__)
PY

"$PYTHON_BIN" -m eqnn run-noisy-comparison \
  --model-families "$MODEL_FAMILY" \
  --num-qubits-values "$NUM_QUBITS" \
  --train-sizes "$TRAIN_SIZE" \
  --epochs-values "$EPOCHS_VALUES" \
  --random-seeds $RANDOM_SEEDS \
  --backend-name qiskit_mixed \
  --odd-qubits-only \
  --noise-model-name depolarizing \
  --noise-strength-values $NOISE_STRENGTH_VALUES \
  --noise-application-scope selected_qubits \
  --noisy-qubit-indices $NOISY_QUBIT_INDICES \
  --output-dir "$RUN_OUTPUT_DIR"

"$PYTHON_BIN" -m eqnn summarize-noisy-comparison \
  --input-dir "$RUN_OUTPUT_DIR" || true

echo "[$(date)] Finished n=7 selected-qubit depolarizing diagnostic task ${SLURM_ARRAY_TASK_ID}"