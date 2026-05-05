#!/bin/bash
#SBATCH --job-name=eqnn-smoke-mitig
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/users/jdehoney/eqnn_thesis}"
VENV_DIR="${VENV_DIR:-/scratch/users/jdehoney/venvs/eqnn_py312}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODULE_PYTHON="${MODULE_PYTHON:-python/3.12.1}"
MODULE_OPENBLAS="${MODULE_OPENBLAS:-openblas/0.3.28}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/noisy_comparisons/smoke_mitigation_validation_${SLURM_ARRAY_JOB_ID}}"

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

TASK_ID="${SLURM_ARRAY_TASK_ID}"

echo "[$(date)] Starting mitigation smoke task $TASK_ID"
echo "Host: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Python executable: $(which python3)"
echo "Python version: $(python3 -V)"
echo "Output root: $OUTPUT_ROOT"

"$PYTHON_BIN" - <<'PY'
import sys
print("sys.executable =", sys.executable)
import qiskit
import qiskit_aer
print("qiskit =", qiskit.__version__)
print("qiskit_aer =", qiskit_aer.__version__)
PY

case "$TASK_ID" in
    0)
        RUN_NAME="symmetry_twirled_eval"
        RUN_OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"

        "$PYTHON_BIN" -m eqnn run-noisy-comparison \
          --model-families su2_qcnn hea_qcnn \
          --num-qubits-values 5 \
          --train-sizes 4 \
          --epochs-values 1 \
          --random-seeds 0 \
          --backend-name qiskit_mixed \
          --odd-qubits-only \
          --noise-model-name depolarizing \
          --noise-strength-values 0.01 \
          --compute-symmetry-twirled-evaluation \
          --num-symmetry-twirl-samples 2 \
          --num-state-samples-for-twirled-evaluation 4 \
          --output-dir "$RUN_OUTPUT_DIR"
        ;;

    1)
        RUN_NAME="noise_aware_training"
        RUN_OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"

        "$PYTHON_BIN" -m eqnn run-noisy-comparison \
          --model-families su2_qcnn \
          --num-qubits-values 5 \
          --train-sizes 4 \
          --epochs-values 2 \
          --random-seeds 0 \
          --backend-name qiskit_mixed \
          --odd-qubits-only \
          --noise-model-name depolarizing \
          --noise-strength-values 0.0 0.03 \
          --noise-aware-training \
          --training-noise-strengths 0.0 0.01 0.03 \
          --training-noise-sampling per_epoch \
          --training-noise-seed 0 \
          --output-dir "$RUN_OUTPUT_DIR"
        ;;

    2)
        RUN_NAME="symmetry_regularized_training"
        RUN_OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"

        "$PYTHON_BIN" -m eqnn run-noisy-comparison \
          --model-families su2_qcnn \
          --num-qubits-values 5 \
          --train-sizes 4 \
          --epochs-values 2 \
          --random-seeds 0 \
          --backend-name qiskit_mixed \
          --odd-qubits-only \
          --noise-model-name depolarizing \
          --noise-strength-values 0.0 0.03 \
          --symmetry-regularization \
          --symmetry-regularization-weight 0.01 \
          --num-symmetry-regularization-samples 2 \
          --symmetry-regularization-state-samples 4 \
          --symmetry-regularization-seed 0 \
          --output-dir "$RUN_OUTPUT_DIR"
        ;;

    *)
        echo "ERROR: unknown task id $TASK_ID"
        exit 1
        ;;
esac

"$PYTHON_BIN" -m eqnn summarize-noisy-comparison \
  --input-dir "$RUN_OUTPUT_DIR" || true

echo "[$(date)] Finished mitigation smoke task $TASK_ID"