#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path("data/comparisons")

RUNS = {
    "numpy_older": ROOT / "eqnn_vs_hea_50ep_20247287" / "summary.csv",
    "torch_main": ROOT / "eqnn_vs_hea_purestate_50ep_torch_20529347" / "summary.csv",
}

OUTDIR = ROOT / "meta_summary"
OUTDIR.mkdir(parents=True, exist_ok=True)

frames = []
for source_run, csv_path in RUNS.items():
    if not csv_path.exists():
        print(f"Skipping missing file: {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    df["source_run"] = source_run
    if "backend_name" not in df.columns:
        df["backend_name"] = "unknown"
    frames.append(df)

if not frames:
    raise SystemExit("No summary.csv files found.")

combined = pd.concat(frames, ignore_index=True)
combined.to_csv(OUTDIR / "combined_runs.csv", index=False)

# Main baseline filter: Torch run, tractable n range
main = combined[
    (combined["source_run"] == "torch_main")
    & (combined["backend_name"] == "torch_pure")
    & (combined["num_qubits"].between(6, 11))
].copy()

main = main.sort_values(["model_family", "num_qubits", "train_size"])
main.to_csv(OUTDIR / "filtered_main_baseline.csv", index=False)

# Side-by-side EQNN vs HEA table
value_cols = [
    "mean_test_accuracy",
    "mean_test_loss",
    "mean_train_accuracy",
    "mean_train_loss",
    "mean_runtime_seconds",
    "num_runs",
]

wide = main.pivot_table(
    index=["backend_name", "num_qubits", "train_size", "epochs"],
    columns="model_family",
    values=value_cols,
)

wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
wide = wide.reset_index()

# Convenience differences
if "mean_test_accuracy_su2_qcnn" in wide.columns and "mean_test_accuracy_hea_qcnn" in wide.columns:
    wide["delta_test_accuracy_su2_minus_hea"] = (
        wide["mean_test_accuracy_su2_qcnn"] - wide["mean_test_accuracy_hea_qcnn"]
    )

if "mean_test_loss_su2_qcnn" in wide.columns and "mean_test_loss_hea_qcnn" in wide.columns:
    wide["delta_test_loss_su2_minus_hea"] = (
        wide["mean_test_loss_su2_qcnn"] - wide["mean_test_loss_hea_qcnn"]
    )

if "mean_runtime_seconds_su2_qcnn" in wide.columns and "mean_runtime_seconds_hea_qcnn" in wide.columns:
    wide["runtime_ratio_hea_over_su2"] = (
        wide["mean_runtime_seconds_hea_qcnn"] / wide["mean_runtime_seconds_su2_qcnn"]
    )

wide = wide.sort_values(["num_qubits", "train_size"])
wide.to_csv(OUTDIR / "model_comparison_wide.csv", index=False)

# Compact human-readable summary by n
summary_by_n = main.groupby(["num_qubits", "model_family"], as_index=False).agg(
    mean_of_mean_test_accuracy=("mean_test_accuracy", "mean"),
    mean_of_mean_test_loss=("mean_test_loss", "mean"),
    mean_of_mean_runtime_seconds=("mean_runtime_seconds", "mean"),
)
summary_by_n.to_csv(OUTDIR / "summary_by_num_qubits.csv", index=False)

print(f"Wrote:")
print(f"  {OUTDIR / 'combined_runs.csv'}")
print(f"  {OUTDIR / 'filtered_main_baseline.csv'}")
print(f"  {OUTDIR / 'model_comparison_wide.csv'}")
print(f"  {OUTDIR / 'summary_by_num_qubits.csv'}")