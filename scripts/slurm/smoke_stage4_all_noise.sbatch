#!/bin/bash
#SBATCH --job-name=smoke_s4_all_noise
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/smoke_stage4_all_noise_%j.out
#SBATCH --error=logs/smoke_stage4_all_noise_%j.err

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
echo "===== CLI help checks ====="
python3 -m eqnn run-all-noise-mitigation-comparison --help | grep -E "noise-model-names|mitigation-methods|train-noise-strength-values|symmetry-regularization-beta-values"
python3 -m eqnn run-noisy-comparison --help | grep -E "train-noise-strength-values|symmetry-regularization-beta-values"

echo
echo "===== Stage 4 all-noise mitigation smoke run ====="
SMOKE_ROOT="data/noisy_comparisons/smoke_all_noise_mitigation"
rm -rf "${SMOKE_ROOT}"

python3 -m eqnn run-all-noise-mitigation-comparison \
  --backend-name qiskit_mixed \
  --model-families su2_qcnn \
  --num-qubits-values 4 \
  --train-sizes 4 \
  --epochs-values 1 \
  --random-seeds 0 \
  --noise-model-names depolarizing phase_damping \
  --noise-strength-values 0.0 0.01 \
  --mitigation-methods none symmetry_regularized \
  --symmetry-regularization-beta-values 0.01 0.1 \
  --output-root "${SMOKE_ROOT}"

echo
echo "===== Inspect noisy_run.json outputs ====="
python3 - <<'PY'
import json
from pathlib import Path
from collections import Counter

root = Path("data/noisy_comparisons/smoke_all_noise_mitigation")
paths = sorted(root.rglob("noisy_run.json"))

print(f"found {len(paths)} noisy_run.json files")
expected = 12
if len(paths) != expected:
    raise SystemExit(f"Expected {expected} noisy_run.json files, found {len(paths)}")

counts = Counter()
symbreak = {}

for path in paths:
    data = json.loads(path.read_text())

    noise_model = data.get("noise_model_name")
    mitigation = data.get("mitigation_method")
    noise_strength = data.get("noise_strength")
    beta = data.get("symmetry_regularization_beta")
    expected_breaking = data.get("expected_symmetry_breaking")
    expected_note = data.get("expected_symmetry_breaking_note")

    print()
    print(path)
    print("  noise_model_name:", noise_model)
    print("  mitigation_method:", mitigation)
    print("  noise_strength:", noise_strength)
    print("  symmetry_regularization_beta:", beta)
    print("  expected_symmetry_breaking:", expected_breaking)
    print("  expected_symmetry_breaking_note:", expected_note)

    if noise_model not in {"depolarizing", "phase_damping"}:
        raise SystemExit(f"Unexpected noise model: {noise_model}")

    if mitigation not in {"none", "symmetry_regularized"}:
        raise SystemExit(f"Unexpected mitigation method: {mitigation}")

    if noise_model == "depolarizing":
        if expected_breaking != "unknown":
            raise SystemExit(f"Expected depolarizing expected_symmetry_breaking=unknown, got {expected_breaking}")
    if noise_model == "phase_damping":
        if expected_breaking is not True and expected_breaking != "true":
            raise SystemExit(f"Expected phase_damping expected_symmetry_breaking=true, got {expected_breaking}")

    if mitigation == "none":
        if beta not in (None, "", "nan"):
            raise SystemExit(f"Expected no beta for mitigation none, got {beta}")
    if mitigation == "symmetry_regularized":
        if beta not in [0.01, 0.1]:
            raise SystemExit(f"Unexpected beta for symmetry_regularized: {beta}")

    counts[(noise_model, mitigation, beta)] += 1
    symbreak[noise_model] = expected_breaking

print()
print("Counts by noise_model, mitigation, beta:")
for key, value in sorted(counts.items(), key=lambda kv: str(kv[0])):
    print(" ", key, value)

expected_counts = {
    ("depolarizing", "none", None): 2,
    ("phase_damping", "none", None): 2,
    ("depolarizing", "symmetry_regularized", 0.01): 2,
    ("depolarizing", "symmetry_regularized", 0.1): 2,
    ("phase_damping", "symmetry_regularized", 0.01): 2,
    ("phase_damping", "symmetry_regularized", 0.1): 2,
}

for key, expected_value in expected_counts.items():
    actual = counts.get(key, 0)
    if actual != expected_value:
        raise SystemExit(f"Count mismatch for {key}: expected {expected_value}, got {actual}")
PY

echo
echo "===== Inspect summary.csv files ====="
python3 - <<'PY'
import pandas as pd
from pathlib import Path

root = Path("data/noisy_comparisons/smoke_all_noise_mitigation")
summary_paths = sorted(root.rglob("summary.csv"))

print(f"found {len(summary_paths)} summary.csv files")
if not summary_paths:
    raise SystemExit("No summary.csv files found")

all_dfs = []
for path in summary_paths:
    print("\nSummary:", path)
    df = pd.read_csv(path)
    print("rows:", len(df))
    print("columns:", len(df.columns))
    all_dfs.append(df)

df = pd.concat(all_dfs, ignore_index=True)

required = [
    "noise_model_name",
    "mitigation_method",
    "expected_symmetry_breaking",
    "expected_symmetry_breaking_note",
    "noise_strength",
    "eval_noise_strength",
    "symmetry_regularization_beta",
    "symmetry_regularization_weight",
    "mean_test_accuracy",
]
print()
print("Required summary columns:")
for col in required:
    print(f"  {col}: {col in df.columns}")
    if col not in df.columns:
        raise SystemExit(f"Missing required summary column: {col}")

print()
print(df[
    [
        "noise_model_name",
        "mitigation_method",
        "expected_symmetry_breaking",
        "noise_strength",
        "symmetry_regularization_beta",
        "mean_test_accuracy",
    ]
].to_string(index=False))

group_cols = [
    "noise_model_name",
    "mitigation_method",
    "noise_strength",
    "symmetry_regularization_beta",
]
print()
print("Grouped row counts:")
print(df.groupby(group_cols, dropna=False).size().to_string())
PY

echo
echo "===== Check ZNE post-processing metadata behavior ====="
POST_JSON="${SMOKE_ROOT}/post_processing_mitigation_methods.json"
if [ -f "${POST_JSON}" ]; then
  cat "${POST_JSON}"
else
  echo "No post_processing_mitigation_methods.json expected because zne was not requested in this smoke."
fi

echo
echo "===== Stage 4 all-noise mitigation smoke completed successfully ====="