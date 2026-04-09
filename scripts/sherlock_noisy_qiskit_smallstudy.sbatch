#!/bin/bash
#SBATCH --job-name=eqnn-noisy-qiskit
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-63
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/users/jdehoney/eqnn_thesis}"
VENV_DIR="${VENV_DIR:-/scratch/users/jdehoney/venvs/eqnn_py312}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODULE_PYTHON="${MODULE_PYTHON:-python/3.12.1}"
MODULE_OPENBLAS="${MODULE_OPENBLAS:-openblas/0.3.28}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/noisy_comparisons/qiskit_mixed_smallstudy_${SLURM_ARRAY_JOB_ID}}"

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

echo "[$(date)] Starting noisy comparison task"
echo "Host: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Working dir: $(pwd)"
echo "Python executable: $(which python3)"
echo "Python version: $(python3 -V)"
echo "SLURM job: ${SLURM_JOB_ID:-}"
echo "SLURM array task: ${SLURM_ARRAY_TASK_ID:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"

python3 - <<'PY'
import sys
print("sys.executable =", sys.executable)
try:
    import qiskit
    import qiskit_aer
    print("qiskit =", qiskit.__version__)
    print("qiskit_aer =", qiskit_aer.__version__)
except Exception as exc:
    print("Qiskit import failed:", exc)
    raise
PY

"$PYTHON_BIN" -m eqnn run-noisy-comparison \
  --model-families su2_qcnn hea_qcnn \
  --num-qubits-values 4 6 \
  --train-sizes 4 8 \
  --epochs-values 10 \
  --random-seeds 0 1 \
  --backend-name qiskit_mixed \
  --noise-model-name depolarizing \
  --noise-strength-values 0.0 0.001 0.005 0.01 \
  --job-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir "$OUTPUT_ROOT"

echo "[$(date)] Finished noisy comparison task ${SLURM_ARRAY_TASK_ID}"