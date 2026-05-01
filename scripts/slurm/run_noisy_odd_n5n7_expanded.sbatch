#!/bin/bash
#SBATCH --job-name=eqnn-noisy-odd-n5n7
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-11
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/users/jdehoney/eqnn_thesis}"
VENV_DIR="${VENV_DIR:-/scratch/users/jdehoney/venvs/eqnn_py312}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODULE_PYTHON="${MODULE_PYTHON:-python/3.12.1}"
MODULE_OPENBLAS="${MODULE_OPENBLAS:-openblas/0.3.28}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/noisy_comparisons/qiskit_mixed_odd_n5n7_expanded_${SLURM_ARRAY_JOB_ID}}"

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
# Sweep definition:
#   6 noise configurations × 2 odd qubit sizes = 12 array tasks.
#
# Noise configs:
#   0: depolarizing, active/default scope
#   1: amplitude_damping, selected-qubit scope
#   2: phase_damping, selected-qubit scope
#   3: coherent_overrotation, fixed
#   4: coherent_overrotation, stochastic
#   5: coherent_overrotation, random_angle
#
# Qubit sizes:
#   n = 5, 7
# ---------------------------------------------------------------------

N_VALUES=(5 7)
NUM_N_VALUES=${#N_VALUES[@]}

TASK_ID="${SLURM_ARRAY_TASK_ID}"
NOISE_CONFIG_INDEX=$(( TASK_ID / NUM_N_VALUES ))
N_INDEX=$(( TASK_ID % NUM_N_VALUES ))

NUM_QUBITS="${N_VALUES[$N_INDEX]}"

MODEL_FAMILIES="${MODEL_FAMILIES:-su2_qcnn hea_qcnn}"
TRAIN_SIZES="${TRAIN_SIZES:-4 8 12 16}"
EPOCHS_VALUES="${EPOCHS_VALUES:-10}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2}"

# Multiple noise strengths, including the high-noise point from prior sweeps.
NOISE_STRENGTH_VALUES="${NOISE_STRENGTH_VALUES:-0.0 0.001 0.005 0.01 0.05 0.1}"

NOISY_QUBIT_INDICES=""
NOISE_MODEL=""
RUN_TAG=""
COHERENT_MODE="<none>"

case "$NOISE_CONFIG_INDEX" in
    0)
        NOISE_MODEL="depolarizing"
        RUN_TAG="noise_depolarizing_active_n${NUM_QUBITS}"
        ;;

    1)
        NOISE_MODEL="amplitude_damping"
        RUN_TAG="noise_amplitude_damping_selected_qubits_n${NUM_QUBITS}"
        NOISY_QUBIT_INDICES="$(seq 0 $((NUM_QUBITS - 1)) | tr '\n' ' ')"
        ;;

    2)
        NOISE_MODEL="phase_damping"
        RUN_TAG="noise_phase_damping_selected_qubits_n${NUM_QUBITS}"
        NOISY_QUBIT_INDICES="$(seq 0 $((NUM_QUBITS - 1)) | tr '\n' ' ')"
        ;;

    3)
        NOISE_MODEL="coherent_overrotation"
        RUN_TAG="noise_coherent_overrotation_fixed_n${NUM_QUBITS}"
        COHERENT_MODE="fixed"
        ;;

    4)
        NOISE_MODEL="coherent_overrotation"
        RUN_TAG="noise_coherent_overrotation_stochastic_n${NUM_QUBITS}"
        COHERENT_MODE="stochastic"
        ;;

    5)
        NOISE_MODEL="coherent_overrotation"
        RUN_TAG="noise_coherent_overrotation_random_angle_n${NUM_QUBITS}"
        COHERENT_MODE="random_angle"
        ;;

    *)
        echo "ERROR: invalid NOISE_CONFIG_INDEX=$NOISE_CONFIG_INDEX"
        exit 1
        ;;
esac

RUN_OUTPUT_DIR="$OUTPUT_ROOT/$RUN_TAG"
mkdir -p "$RUN_OUTPUT_DIR"

echo "[$(date)] Starting expanded n=5,7 noisy comparison task"
echo "Host: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Working dir: $(pwd)"
echo "Python executable: $(which python3)"
echo "Python version: $(python3 -V)"
echo "SLURM job: ${SLURM_JOB_ID:-}"
echo "SLURM array task: ${SLURM_ARRAY_TASK_ID:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "Noise config index: $NOISE_CONFIG_INDEX"
echo "Noise model: $NOISE_MODEL"
echo "Num qubits: $NUM_QUBITS"
echo "Model families: $MODEL_FAMILIES"
echo "Train sizes: $TRAIN_SIZES"
echo "Epoch values: $EPOCHS_VALUES"
echo "Random seeds: $RANDOM_SEEDS"
echo "Noise strengths: $NOISE_STRENGTH_VALUES"
echo "Noisy qubit indices: ${NOISY_QUBIT_INDICES:-<none>}"
echo "Coherent mode: $COHERENT_MODE"
echo "Output dir: $RUN_OUTPUT_DIR"

"$PYTHON_BIN" - <<'PY'
import sys
print("sys.executable =", sys.executable)

import qiskit
import qiskit_aer
print("qiskit =", qiskit.__version__)
print("qiskit_aer =", qiskit_aer.__version__)
PY

CMD=(
  "$PYTHON_BIN" -m eqnn run-noisy-comparison
  --model-families $MODEL_FAMILIES
  --num-qubits-values "$NUM_QUBITS"
  --train-sizes $TRAIN_SIZES
  --epochs-values $EPOCHS_VALUES
  --random-seeds $RANDOM_SEEDS
  --backend-name qiskit_mixed
  --odd-qubits-only
  --noise-model-name "$NOISE_MODEL"
  --noise-strength-values $NOISE_STRENGTH_VALUES
  --output-dir "$RUN_OUTPUT_DIR"
)

if [[ -n "$NOISY_QUBIT_INDICES" ]]; then
  CMD+=(
    --noise-application-scope selected_qubits
    --noisy-qubit-indices $NOISY_QUBIT_INDICES
  )
fi

case "$COHERENT_MODE" in
  fixed)
    CMD+=(
      --coherent-overrotation-mode fixed
    )
    ;;

  stochastic)
    CMD+=(
      --coherent-overrotation-mode stochastic
      --coherent-overrotation-probability "${COHERENT_OVERROTATION_PROBABILITY:-0.5}"
      --coherent-overrotation-seed "${COHERENT_OVERROTATION_SEED:-0}"
    )
    ;;

  random_angle)
    CMD+=(
      --coherent-overrotation-mode random_angle
      --coherent-overrotation-angle-std "${COHERENT_OVERROTATION_ANGLE_STD:-0.005}"
      --coherent-overrotation-seed "${COHERENT_OVERROTATION_SEED:-0}"
    )
    ;;

  "<none>")
    ;;

  *)
    echo "ERROR: invalid COHERENT_MODE=$COHERENT_MODE"
    exit 1
    ;;
esac

echo "Command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

"$PYTHON_BIN" -m eqnn summarize-noisy-comparison \
  --input-dir "$RUN_OUTPUT_DIR" || true

"$PYTHON_BIN" -m eqnn summarize-zne \
  --input-dir "$RUN_OUTPUT_DIR" \
  --metric test_accuracy || true

echo "[$(date)] Finished expanded n=5,7 noisy comparison task ${SLURM_ARRAY_TASK_ID}"