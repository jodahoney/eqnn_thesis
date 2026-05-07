#!/bin/bash
#SBATCH --job-name=smoke_s3_beta
#SBATCH --partition=normal
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/smoke_stage3_beta_%j.out
#SBATCH --error=logs/smoke_stage3_beta_%j.err

set -euo pipefail

cd /scratch/users/jdehoney/eqnn_thesis
mkdir -p logs

module load python/3.12.1
module load openblas/0.3.28

source /scratch/users/jdehoney/venvs/eqnn_py312/bin/activate

echo "===== Python / package versions ====="
python3 - <<'PY'
import sys
import qiskit
import qiskit_aer
print("python =", sys.version)
print("qiskit =", qiskit.__version__)
print("qiskit_aer =", qiskit_aer.__version__)
PY

echo
echo "===== Compile checks ====="
python3 -m py_compile \
  eqnn/experiments/noisy_comparison.py \
  eqnn/experiments/zero_noise_extrapolation.py \
  eqnn/cli.py \
  tests/test_noisy_comparison.py \
  tests/test_zero_noise_extrapolation.py

echo
echo "===== Unit tests ====="
python3 -m unittest tests.test_noisy_comparison
python3 -m unittest tests.test_zero_noise_extrapolation

echo
echo "===== CLI help check ====="
python3 -m eqnn run-noisy-comparison --help | grep -E "symmetry-regularization-beta-values|symmetry-regularization-beta"

echo
echo "===== Stage 3 beta sweep smoke run ====="
SMOKE_ROOT="data/noisy_comparisons/smoke_stage3_beta_sweep"
rm -rf "${SMOKE_ROOT}"

python3 -m eqnn run-noisy-comparison \
  --backend-name qiskit_mixed \
  --model-families su2_qcnn \
  --num-qubits-values 4 \
  --train-sizes 4 \
  --epochs-values 1 \
  --random-seeds 0 \
  --noise-model-name depolarizing \
  --noise-strength-values 0.0 0.01 \
  --symmetry-regularization \
  --symmetry-regularization-beta-values 0.0 0.01 0.1 \
  --output-root "${SMOKE_ROOT}"

echo
echo "===== Inspect noisy_run.json outputs ====="
python3 - <<'PY'
import json
from pathlib import Path

root = Path("data/noisy_comparisons/smoke_stage3_beta_sweep")
paths = sorted(root.rglob("noisy_run.json"))

print(f"found {len(paths)} noisy_run.json files")
expected = 3 * 2  # 3 beta values x 2 eval noise strengths
if len(paths) != expected:
    raise SystemExit(f"Expected {expected} noisy_run.json files, found {len(paths)}")

seen = set()

for path in paths:
    data = json.loads(path.read_text())
    beta = data.get("symmetry_regularization_beta")
    weight = data.get("symmetry_regularization_weight")
    noise_strength = data.get("noise_strength")

    print()
    print(path)
    print("  beta:", beta)
    print("  weight:", weight)
    print("  noise_strength:", noise_strength)
    print("  symmetry_regularization:", data.get("symmetry_regularization"))
    print("  symmetry_regularization_enabled:", data.get("symmetry_regularization_enabled"))
    print("  final_symmetry_penalty:", data.get("final_symmetry_penalty"))
    print("  final_equivariance_error_mean:", data.get("final_equivariance_error_mean"))
    print("  final_equivariance_error_max:", data.get("final_equivariance_error_max"))

    if beta not in [0.0, 0.01, 0.1]:
        raise SystemExit(f"Unexpected beta value: {beta}")
    if weight != beta:
        raise SystemExit(f"Expected weight == beta, got weight={weight}, beta={beta}")
    if data.get("symmetry_regularization_enabled") is not True:
        raise SystemExit("Expected symmetry_regularization_enabled=true")

    seen.add((beta, noise_strength))

expected_pairs = {
    (0.0, 0.0),
    (0.0, 0.01),
    (0.01, 0.0),
    (0.01, 0.01),
    (0.1, 0.0),
    (0.1, 0.01),
}
missing = expected_pairs.difference(seen)
if missing:
    raise SystemExit(f"Missing beta/noise pairs: {missing}")
PY

echo
echo "===== Inspect summary.csv ====="
test -f "${SMOKE_ROOT}/summary.csv"

python3 - <<'PY'
import pandas as pd

path = "data/noisy_comparisons/smoke_stage3_beta_sweep/summary.csv"
df = pd.read_csv(path)

print("summary rows:", len(df))
print("columns include required Stage 3 fields:")
required = [
    "symmetry_regularization",
    "symmetry_regularization_enabled",
    "symmetry_regularization_beta",
    "symmetry_regularization_weight",
    "mean_symmetry_penalty_history",
    "mean_test_accuracy",
    "noise_strength",
]
for col in required:
    print(f"  {col}: {col in df.columns}")
    if col not in df.columns:
        raise SystemExit(f"Missing required column: {col}")

print()
print(df[
    [
        "model_family",
        "noise_strength",
        "symmetry_regularization",
        "symmetry_regularization_enabled",
        "symmetry_regularization_beta",
        "symmetry_regularization_weight",
        "mean_test_accuracy",
        "mean_symmetry_penalty_history",
    ]
].to_string(index=False))

pairs = set(zip(df["symmetry_regularization_beta"], df["noise_strength"]))
expected = {
    (0.0, 0.0),
    (0.0, 0.01),
    (0.01, 0.0),
    (0.01, 0.01),
    (0.1, 0.0),
    (0.1, 0.01),
}
missing = expected.difference(pairs)
if missing:
    raise SystemExit(f"Missing beta/noise pairs in summary.csv: {missing}")
PY

echo
echo "===== Stage 3 ZNE compatibility check ====="
python3 -m eqnn summarize-zne \
  --input-dir "${SMOKE_ROOT}" \
  --metric mean_test_accuracy \
  --fit-types linear log_margin_linear \
  --fit-ranges full \
  --chance-accuracy 0.5

python3 - <<'PY'
import pandas as pd
from pathlib import Path

path = Path("data/noisy_comparisons/smoke_stage3_beta_sweep/zne_summary.csv")
df = pd.read_csv(path)

print("ZNE rows:", len(df))
print(df[
    [
        "fit_type",
        "fit_range",
        "symmetry_regularization_beta",
        "symmetry_regularization_weight",
        "num_points_used",
        "fit_valid",
        "zero_noise_estimate",
        "fit_warning",
    ]
].to_string(index=False))

betas = set(df["symmetry_regularization_beta"])
expected_betas = {0.0, 0.01, 0.1}
if not expected_betas.issubset(betas):
    raise SystemExit(f"ZNE did not preserve beta grouping. Found betas: {betas}")
PY

echo
echo "===== Stage 3 beta sweep smoke completed successfully ====="
