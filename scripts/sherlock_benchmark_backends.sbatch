#!/bin/bash
#SBATCH --job-name=eqnn-bench
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-3
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/users/jdehoney/eqnn_thesis}"
VENV_DIR="${VENV_DIR:-/scratch/users/jdehoney/venvs/eqnn_py312}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODULE_PYTHON="${MODULE_PYTHON:-python/3.12.1}"
MODULE_OPENBLAS="${MODULE_OPENBLAS:-openblas/0.3.28}"

MODEL_FAMILIES=(${MODEL_FAMILIES:-su2_qcnn su2_qcnn hea_qcnn hea_qcnn})
NUM_QUBITS_LIST=(${NUM_QUBITS_LIST:-6 10 6 10})

TASK_ID="${SLURM_ARRAY_TASK_ID:?missing array task id}"
TOTAL="${#MODEL_FAMILIES[@]}"

if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
    echo "ERROR: task id $TASK_ID out of range 0..$((TOTAL-1))"
    exit 1
fi

MODEL_FAMILY="${MODEL_FAMILIES[$TASK_ID]}"
NUM_QUBITS="${NUM_QUBITS_LIST[$TASK_ID]}"

RUN_ID="${RUN_ID:-backend_benchmark_${SLURM_ARRAY_JOB_ID:-manual}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/backend_benchmarks/$RUN_ID}"
OUTPUT_DIR="$OUTPUT_ROOT/${MODEL_FAMILY}/n${NUM_QUBITS}"

mkdir -p "$PROJECT_ROOT/logs/slurm" "$OUTPUT_DIR"
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

echo "[$(date)] Starting backend benchmark"
echo "Host: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Model family: $MODEL_FAMILY"
echo "Num qubits: $NUM_QUBITS"
echo "Output dir: $OUTPUT_DIR"

"$PYTHON_BIN" -m eqnn benchmark-backends \
  --backends numpy_pure torch_pure \
  --num-qubits "$NUM_QUBITS" \
  --model-family "$MODEL_FAMILY" \
  --num-points 9 \
  --epochs 50 \
  --output-dir "$OUTPUT_DIR"

echo "[$(date)] Finished backend benchmark"