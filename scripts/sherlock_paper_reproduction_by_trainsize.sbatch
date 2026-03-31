#!/bin/bash
#SBATCH --job-name=eqnn-paper-nt
#SBATCH --partition=normal
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-47

set -euo pipefail

PROJ="${PROJ:-/scratch/users/jdehoney/eqnn_thesis}"
PROJECT_ROOT="${PROJECT_ROOT:-$PROJ}"

MODULE_PYTHON="${MODULE_PYTHON:-python/3.12.1}"
MODULE_OPENBLAS="${MODULE_OPENBLAS:-openblas/0.3.28}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-/scratch/users/jdehoney/venvs/eqnn_py312}"

NUM_QUBITS_LIST=(${NUM_QUBITS_LIST:-13 12 11 10 9 8 7 6})
TRAIN_SIZES_LIST=(${TRAIN_SIZES_LIST:-12 10 8 6 4 2})
RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2}"

EPOCHS="${EPOCHS:-750}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
GRADIENT_BACKEND="${GRADIENT_BACKEND:-exact}"
INITIALIZATION_STRATEGY="${INITIALIZATION_STRATEGY:-noisy_current}"
INITIALIZATION_NOISE_SCALE="${INITIALIZATION_NOISE_SCALE:-0.05}"
EIGENSOLVER="${EIGENSOLVER:-auto}"

FORCE_REGENERATE="${FORCE_REGENERATE:-0}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

N_COUNT="${#NUM_QUBITS_LIST[@]}"
T_COUNT="${#TRAIN_SIZES_LIST[@]}"
TOTAL=$((N_COUNT * T_COUNT))

if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
    echo "ERROR: task id $TASK_ID out of range 0..$((TOTAL-1))"
    exit 1
fi

N_INDEX=$(( TASK_ID / T_COUNT ))
T_INDEX=$(( TASK_ID % T_COUNT ))

NUM_QUBITS="${NUM_QUBITS_LIST[$N_INDEX]}"
TRAIN_SIZE="${TRAIN_SIZES_LIST[$T_INDEX]}"

RUN_ID="${RUN_ID:-paper_reproduction_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/reproduction/$RUN_ID}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-$OUTPUT_ROOT/n${NUM_QUBITS}/train_size_${TRAIN_SIZE}}"

CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/data/cache/paper_reproduction}"
PROFILE_JSON="${PROFILE_JSON:-$RUN_OUTPUT_DIR/runtime_profile.json}"

mkdir -p "$PROJECT_ROOT/logs/slurm" "$RUN_OUTPUT_DIR" "$CACHE_DIR"
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

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

echo "[$(date)] Starting paper reproduction task"
echo "Host: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Working dir: $(pwd)"
echo "Python executable: $(command -v "$PYTHON_BIN")"
echo "Python version: $("$PYTHON_BIN" --version 2>&1)"
echo "SLURM array job: ${SLURM_ARRAY_JOB_ID:-none}"
echo "SLURM job: ${SLURM_JOB_ID:-unknown}"
echo "SLURM array task: ${SLURM_ARRAY_TASK_ID:-none}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "Decoded num qubits: $NUM_QUBITS"
echo "Decoded train size: $TRAIN_SIZE"
echo "Seeds: $RANDOM_SEEDS"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Gradient backend: $GRADIENT_BACKEND"
echo "Initialization strategy: $INITIALIZATION_STRATEGY"
echo "Initialization noise scale: $INITIALIZATION_NOISE_SCALE"
echo "Eigensolver: $EIGENSOLVER"
echo "Cache dir: $CACHE_DIR"
echo "Profile json: $PROFILE_JSON"
echo "Force regenerate: $FORCE_REGENERATE"
echo "Output dir: $RUN_OUTPUT_DIR"

python - <<'PY'
import sys
import numpy, scipy, eqnn
print("sys.executable =", sys.executable)
print("numpy =", numpy.__version__)
print("scipy =", scipy.__version__)
print("eqnn import ok")
PY

echo "[$(date)] Launching reproduction for N=$NUM_QUBITS, train_size=$TRAIN_SIZE"

if [[ "$FORCE_REGENERATE" == "1" ]]; then
    "$PYTHON_BIN" -m eqnn run-paper-reproduction \
      --num-qubits "$NUM_QUBITS" \
      --train-sizes "$TRAIN_SIZE" \
      --random-seeds $RANDOM_SEEDS \
      --epochs "$EPOCHS" \
      --learning-rate "$LEARNING_RATE" \
      --gradient-backend "$GRADIENT_BACKEND" \
      --initialization-strategy "$INITIALIZATION_STRATEGY" \
      --initialization-noise-scale "$INITIALIZATION_NOISE_SCALE" \
      --eigensolver "$EIGENSOLVER" \
      --cache-dir "$CACHE_DIR" \
      --force-regenerate \
      --profile-json "$PROFILE_JSON" \
      --output-dir "$RUN_OUTPUT_DIR"
else
    "$PYTHON_BIN" -m eqnn run-paper-reproduction \
      --num-qubits "$NUM_QUBITS" \
      --train-sizes "$TRAIN_SIZE" \
      --random-seeds $RANDOM_SEEDS \
      --epochs "$EPOCHS" \
      --learning-rate "$LEARNING_RATE" \
      --gradient-backend "$GRADIENT_BACKEND" \
      --initialization-strategy "$INITIALIZATION_STRATEGY" \
      --initialization-noise-scale "$INITIALIZATION_NOISE_SCALE" \
      --eigensolver "$EIGENSOLVER" \
      --cache-dir "$CACHE_DIR" \
      --profile-json "$PROFILE_JSON" \
      --output-dir "$RUN_OUTPUT_DIR"
fi

echo "[$(date)] Completed paper reproduction for N=$NUM_QUBITS, train_size=$TRAIN_SIZE"
echo "Contents of $RUN_OUTPUT_DIR:"
ls -lah "$RUN_OUTPUT_DIR"