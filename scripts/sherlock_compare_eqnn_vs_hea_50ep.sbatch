#!/bin/bash
#SBATCH --job-name=eqnn-vs-hea-50
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-287
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/users/jdehoney/eqnn_thesis}"
VENV_DIR="${VENV_DIR:-/scratch/users/jdehoney/venvs/eqnn_py312}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODULE_PYTHON="${MODULE_PYTHON:-python/3.12.1}"
MODULE_OPENBLAS="${MODULE_OPENBLAS:-openblas/0.3.28}"

RUN_ID="${RUN_ID:-eqnn_vs_hea_50ep_${SLURM_ARRAY_JOB_ID:-manual}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/comparisons/$RUN_ID}"

mkdir -p "$PROJECT_ROOT/logs/slurm" "$OUTPUT_ROOT"
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

echo "[$(date)] Starting comparison task ${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Output root: $OUTPUT_ROOT"

"$PYTHON_BIN" -m eqnn run-calibration-sweep \
  --model-families su2_qcnn hea_qcnn \
  --num-qubits-values 6 7 8 9 10 11 12 13 \
  --train-sizes 2 4 6 8 10 12 \
  --epochs-values 50 \
  --random-seeds 0 1 2 \
  --job-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir "$OUTPUT_ROOT"

echo "[$(date)] Finished comparison task ${SLURM_ARRAY_TASK_ID}"