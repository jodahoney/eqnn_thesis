#!/bin/bash
#SBATCH --job-name=smoke_s1_s2
#SBATCH --partition=normal
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/smoke_stage1_stage2_%j.out
#SBATCH --error=logs/smoke_stage1_stage2_%j.err

set -euo pipefail

cd /scratch/users/jdehoney/eqnn_thesis
mkdir -p logs

module load python/3.12.1
module load openblas/0.3.28

source /scratch/users/jdehoney/venvs/eqnn_py312/bin/activate

echo "===== Python / package versions ====="
python3 - <<'PY'
import sys
print("python =", sys.version)

import qiskit
import qiskit_aer
print("qiskit =", qiskit.__version__)
print("qiskit_aer =", qiskit_aer.__version__)
PY

echo
echo "===== Compile checks: Stage 1 + Stage 2 files ====="
python3 -m py_compile \
  eqnn/experiments/zero_noise_extrapolation.py \
  eqnn/experiments/noisy_comparison.py \
  eqnn/cli.py \
  tests/test_zero_noise_extrapolation.py \
  tests/test_noisy_comparison.py

echo
echo "===== Unit tests: Stage 1 ZNE ====="
python3 -m unittest tests.test_zero_noise_extrapolation

echo
echo "===== Unit tests: Stage 2 noisy comparison ====="
python3 -m unittest tests.test_noisy_comparison

echo
echo "===== CLI help checks ====="
python3 -m eqnn summarize-zne --help | grep -E "fit-types|fit-ranges|low-noise-max|chance-accuracy"
python3 -m eqnn run-noisy-comparison --help | grep -E "train-noise-strength-values|train-noise-sampling-mode|output-root|noise-aware-training"

echo
echo "===== Stage 2 Qiskit smoke run ====="
SMOKE_ROOT="data/noisy_comparisons/smoke_stage1_stage2_noise_aware_nozero"
rm -rf "${SMOKE_ROOT}"

python3 -m eqnn run-noisy-comparison \
  --backend-name qiskit_mixed \
  --model-families su2_qcnn \
  --num-qubits-values 4 \
  --train-sizes 4 \
  --epochs-values 1 \
  --random-seeds 0 \
  --noise-model-name depolarizing \
  --noise-strength-values 0.0 0.01 0.03 \
  --noise-aware-training \
  --train-noise-strength-values 0.01 0.03 \
  --output-root "${SMOKE_ROOT}"

echo
echo "===== Inspect Stage 2 noisy_run.json metadata ====="
python3 - <<'PY'
import json
from pathlib import Path

root = Path("data/noisy_comparisons/smoke_stage1_stage2_noise_aware_nozero")
paths = sorted(root.rglob("noisy_run.json"))

print(f"found {len(paths)} noisy_run.json files")
if len(paths) != 3:
    raise SystemExit(f"Expected 3 noisy_run.json files, found {len(paths)}")

for path in paths:
    print("\n" + str(path))
    data = json.loads(path.read_text())

    for key in [
        "noise_aware_training",
        "training_noise_strengths",
        "train_noise_strength_values",
        "train_noise_sampling_mode",
        "training_noise_sampling",
        "train_noise_includes_zero",
        "training_noise_defaulted_from_evaluation_grid",
        "training_noise_schedule",
        "training_noise_strength_counts",
        "noise_strength",
        "eval_noise_strength",
    ]:
        if key in data:
            print(f"{key}: {data[key]}")

    assert data.get("noise_aware_training") is True
    assert data.get("train_noise_includes_zero") is False
    assert data.get("training_noise_defaulted_from_evaluation_grid") is False
    assert data.get("train_noise_strength_values") == [0.01, 0.03]
    assert data.get("training_noise_strengths") == [0.01, 0.03]
    assert data.get("eval_noise_strength") in [0.0, 0.01, 0.03]
PY

echo
echo "===== Check Stage 2 summary.csv ====="
test -f "${SMOKE_ROOT}/summary.csv"

python3 - <<'PY'
import pandas as pd

path = "data/noisy_comparisons/smoke_stage1_stage2_noise_aware_nozero/summary.csv"
df = pd.read_csv(path)

print("summary rows:", len(df))
print("summary columns include required Stage 2 fields:")
required = [
    "noise_aware_training",
    "training_noise_strengths",
    "train_noise_strength_values",
    "training_noise_sampling",
    "train_noise_sampling_mode",
    "train_noise_includes_zero",
    "training_noise_defaulted_from_evaluation_grid",
    "noise_strength",
    "eval_noise_strength",
    "mean_test_accuracy",
]
for col in required:
    print(f"  {col}: {col in df.columns}")
    if col not in df.columns:
        raise SystemExit(f"Missing required summary column: {col}")

print()
print(df[
    [
        "model_family",
        "noise_strength",
        "eval_noise_strength",
        "noise_aware_training",
        "train_noise_strength_values",
        "train_noise_includes_zero",
        "mean_test_accuracy",
    ]
].to_string(index=False))
PY

echo
echo "===== Stage 1 ZNE on Stage 2 smoke output ====="
python3 -m eqnn summarize-zne \
  --input-dir "${SMOKE_ROOT}" \
  --metric mean_test_accuracy \
  --fit-types linear quadratic log_margin_linear \
  --fit-ranges low_noise full \
  --low-noise-max 0.05 \
  --chance-accuracy 0.5

echo
echo "===== Inspect Stage 1 ZNE output ====="
test -f "${SMOKE_ROOT}/zne_summary.csv"
test -f "${SMOKE_ROOT}/zne_summary.json"

python3 - <<'PY'
import json
from pathlib import Path
import pandas as pd

root = Path("data/noisy_comparisons/smoke_stage1_stage2_noise_aware_nozero")
json_path = root / "zne_summary.json"
rows = json.loads(json_path.read_text())

print(f"ZNE rows: {len(rows)}")
if not rows:
    raise SystemExit("No ZNE rows produced")

fit_pairs = sorted({(row.get("fit_type"), row.get("fit_range")) for row in rows})
print("fit_type x fit_range pairs:")
for pair in fit_pairs:
    print("  ", pair)

expected = {
    ("linear", "low_noise"),
    ("linear", "full"),
    ("quadratic", "low_noise"),
    ("quadratic", "full"),
    ("log_margin_linear", "low_noise"),
    ("log_margin_linear", "full"),
}
missing = expected.difference(fit_pairs)
if missing:
    raise SystemExit(f"Missing expected ZNE fit pairs: {missing}")

df = pd.read_csv(root / "zne_summary.csv")
required_cols = [
    "metric_name",
    "fit_type",
    "fit_range",
    "low_noise_max",
    "chance_accuracy",
    "num_points_total",
    "num_points_used",
    "zero_noise_estimate",
    "fit_valid",
    "fit_warning",
    "train_noise_strength_values",
    "train_noise_includes_zero",
]
print()
print("ZNE CSV required fields:")
for col in required_cols:
    print(f"  {col}: {col in df.columns}")
    if col not in df.columns:
        raise SystemExit(f"Missing required ZNE column: {col}")

print()
print(df[
    [
        "fit_type",
        "fit_range",
        "num_points_used",
        "fit_valid",
        "zero_noise_estimate",
        "fit_warning",
    ]
].to_string(index=False))
PY

echo
echo "===== Smoke Stage 1 + Stage 2 completed successfully ====="