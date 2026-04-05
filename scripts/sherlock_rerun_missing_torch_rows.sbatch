#!/bin/bash
#SBATCH --job-name=eqnn-torch-rerun
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-35
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/users/jdehoney/eqnn_thesis}"
VENV_DIR="${VENV_DIR:-/scratch/users/jdehoney/venvs/eqnn_py312}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODULE_PYTHON="${MODULE_PYTHON:-python/3.12.1}"
MODULE_OPENBLAS="${MODULE_OPENBLAS:-openblas/0.3.28}"

RUN_ID="${RUN_ID:-eqnn_vs_hea_purestate_50ep_torch_20529347}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/comparisons/$RUN_ID}"
MANIFEST="${MANIFEST:-$OUTPUT_ROOT/missing_rows.csv}"

mkdir -p "$PROJECT_ROOT/logs/slurm"
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

TASK_ID="${SLURM_ARRAY_TASK_ID:?missing array task id}"
export TASK_ID
export MANIFEST

eval "$(
python3 - <<'PY'
import csv
import os
from pathlib import Path

task_id = int(os.environ["TASK_ID"])
manifest = Path(os.environ["MANIFEST"])

with manifest.open() as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)

if task_id < 0 or task_id >= len(rows):
    raise SystemExit(f"task_id {task_id} out of range for {len(rows)} rows")

row = rows[task_id]
print(f'MODEL_FAMILY="{row["model_family"]}"')
print(f'NUM_QUBITS="{row["num_qubits"]}"')
print(f'TRAIN_SIZE="{row["train_size"]}"')
print(f'EPOCHS="{row["epochs"]}"')
PY
)"

echo "[$(date)] Starting rerun"
echo "Host: $(hostname)"
echo "Model family: $MODEL_FAMILY"
echo "Num qubits: $NUM_QUBITS"
echo "Train size: $TRAIN_SIZE"
echo "Epochs: $EPOCHS"
echo "Output root: $OUTPUT_ROOT"

python3 - <<'PY'
import sys
import torch
print("sys.executable =", sys.executable)
print("torch =", torch.__version__)
print("cuda available =", torch.cuda.is_available())
PY

"$PYTHON_BIN" -m eqnn run-calibration-sweep \
  --model-families "$MODEL_FAMILY" \
  --num-qubits-values "$NUM_QUBITS" \
  --train-sizes "$TRAIN_SIZE" \
  --epochs-values "$EPOCHS" \
  --random-seeds 0 1 2 \
  --backend-name torch_pure \
  --job-index 0 \
  --output-dir "$OUTPUT_ROOT"

echo "[$(date)] Finished rerun"