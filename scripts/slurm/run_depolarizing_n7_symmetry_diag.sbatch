#!/bin/bash
#SBATCH --job-name=eqnn-depol-n7-sym
#SBATCH --partition=normal
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/users/jdehoney/eqnn_thesis}"
VENV_DIR="${VENV_DIR:-/scratch/users/jdehoney/venvs/eqnn_py312}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODULE_PYTHON="${MODULE_PYTHON:-python/3.12.1}"
MODULE_OPENBLAS="${MODULE_OPENBLAS:-openblas/0.3.28}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/noisy_comparisons/depolarizing_n7_symmetry_diag_${SLURM_JOB_ID}}"
RUN_OUTPUT_DIR="$OUTPUT_ROOT/symmetry_diagnostic_depolarizing_n7"

mkdir -p "$PROJECT_ROOT/logs/slurm"
mkdir -p "$RUN_OUTPUT_DIR"
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

MODEL_FAMILIES="${MODEL_FAMILIES:-su2_qcnn hea_qcnn}"
NUM_QUBITS="${NUM_QUBITS:-7}"
TRAIN_SIZES="${TRAIN_SIZES:-16}"
EPOCHS_VALUES="${EPOCHS_VALUES:-10}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2}"
NOISE_STRENGTH_VALUES="${NOISE_STRENGTH_VALUES:-0.0 0.01 0.05 0.1}"

echo "[$(date)] Starting n=7 depolarizing symmetry diagnostic"
echo "Host: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Run output dir: $RUN_OUTPUT_DIR"
echo "Model families: $MODEL_FAMILIES"
echo "Num qubits: $NUM_QUBITS"
echo "Train sizes: $TRAIN_SIZES"
echo "Epochs: $EPOCHS_VALUES"
echo "Seeds: $RANDOM_SEEDS"
echo "Noise strengths: $NOISE_STRENGTH_VALUES"

"$PYTHON_BIN" - <<'PY'
import sys
print("sys.executable =", sys.executable)
import qiskit
import qiskit_aer
print("qiskit =", qiskit.__version__)
print("qiskit_aer =", qiskit_aer.__version__)
PY

"$PYTHON_BIN" -m eqnn run-noisy-comparison \
  --model-families $MODEL_FAMILIES \
  --num-qubits-values "$NUM_QUBITS" \
  --train-sizes $TRAIN_SIZES \
  --epochs-values $EPOCHS_VALUES \
  --random-seeds $RANDOM_SEEDS \
  --backend-name qiskit_mixed \
  --odd-qubits-only \
  --noise-model-name depolarizing \
  --noise-strength-values $NOISE_STRENGTH_VALUES \
  --compute-symmetry-diagnostics \
  --num-symmetry-samples 8 \
  --num-state-samples-for-diagnostic 8 \
  --output-dir "$RUN_OUTPUT_DIR"

"$PYTHON_BIN" -m eqnn summarize-noisy-comparison \
  --input-dir "$RUN_OUTPUT_DIR"

echo "[$(date)] Finished n=7 depolarizing symmetry diagnostic"