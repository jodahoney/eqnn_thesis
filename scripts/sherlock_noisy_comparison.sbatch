#!/bin/bash
#SBATCH --job-name=eqnn-noisy-odd
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-5
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/users/jdehoney/eqnn_thesis}"
VENV_DIR="${VENV_DIR:-/scratch/users/jdehoney/venvs/eqnn_py312}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODULE_PYTHON="${MODULE_PYTHON:-python/3.12.1}"
MODULE_OPENBLAS="${MODULE_OPENBLAS:-openblas/0.3.28}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/noisy_comparisons/qiskit_mixed_odd_${SLURM_ARRAY_JOB_ID}}"

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

NOISE_MODELS=("depolarizing" "coherent_overrotation")
N_VALUES=(5 7 9)

TASK_ID=${SLURM_ARRAY_TASK_ID}
NOISE_INDEX=$(( TASK_ID / 3 ))
N_INDEX=$(( TASK_ID % 3 ))

NOISE_MODEL="${NOISE_MODELS[$NOISE_INDEX]}"
NUM_QUBITS="${N_VALUES[$N_INDEX]}"

MODEL_FAMILIES="${MODEL_FAMILIES:-su2_qcnn hea_qcnn}"
TRAIN_SIZES="${TRAIN_SIZES:-4 8 12 16}"
EPOCHS_VALUES="${EPOCHS_VALUES:-10}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2}"
NOISE_STRENGTH_VALUES="${NOISE_STRENGTH_VALUES:-0.0 0.001 0.005 0.01 0.05 0.1}"

RUN_TAG="noise_${NOISE_MODEL}_n${NUM_QUBITS}"
RUN_OUTPUT_DIR="$OUTPUT_ROOT/$RUN_TAG"

mkdir -p "$RUN_OUTPUT_DIR"

echo "[$(date)] Starting noisy comparison task"
echo "Host: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Working dir: $(pwd)"
echo "Python executable: $(which python3)"
echo "Python version: $(python3 -V)"
echo "SLURM job: ${SLURM_JOB_ID:-}"
echo "SLURM array task: ${SLURM_ARRAY_TASK_ID:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "Noise model: $NOISE_MODEL"
echo "Num qubits: $NUM_QUBITS"
echo "Train sizes: $TRAIN_SIZES"
echo "Epoch values: $EPOCHS_VALUES"
echo "Random seeds: $RANDOM_SEEDS"
echo "Noise strengths: $NOISE_STRENGTH_VALUES"
echo "Output dir: $RUN_OUTPUT_DIR"

python3 - <<'PY'
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
  --noise-model-name "$NOISE_MODEL" \
  --noise-strength-values $NOISE_STRENGTH_VALUES \
  --output-dir "$RUN_OUTPUT_DIR"

"$PYTHON_BIN" -m eqnn summarize-noisy-comparison \
  --input-dir "$RUN_OUTPUT_DIR" || true

echo "[$(date)] Finished noisy comparison task ${SLURM_ARRAY_TASK_ID}"