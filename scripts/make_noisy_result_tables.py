#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build coverage and results tables from qiskit_mixed_odd aggregated summary output."
    )
    ap.add_argument("--aggregated-csv", required=True, help="Path to qiskit_mixed_odd_aggregated_summary.csv")
    ap.add_argument("--output-dir", required=True, help="Directory to write derived tables")
    args = ap.parse_args()

    aggregated_csv = Path(args.aggregated_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(aggregated_csv)

    # Normalize numeric columns where present.
    numeric_cols = [
        "num_qubits", "train_size", "epochs", "noise_strength", "seeds_present",
        "parameter_count_mean", "classification_threshold_mean",
        "train_loss_mean", "train_loss_std", "test_loss_mean", "test_loss_std",
        "train_accuracy_mean", "train_accuracy_std", "test_accuracy_mean", "test_accuracy_std",
        "history_final_loss_mean", "history_final_loss_std",
        "history_final_accuracy_mean", "history_final_accuracy_std",
        "history_best_loss_mean", "history_best_loss_std",
        "history_best_accuracy_mean", "history_best_accuracy_std",
        "build_time_sec_mean", "forward_time_sec_mean", "gradient_time_sec_mean",
        "train_time_sec_mean", "total_time_sec_mean",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Coverage table.
    coverage_cols = [
        "num_qubits",
        "model_family",
        "noise_model_name",
        "train_size",
        "noise_strength",
        "seeds_present",
        "seeds",
    ]
    coverage = df[coverage_cols].copy().sort_values(
        by=["num_qubits", "model_family", "noise_model_name", "train_size", "noise_strength"]
    )

    # Complete conditions = all 3 seeds present.
    complete = df[df["seeds_present"] >= 3].copy().sort_values(
        by=["num_qubits", "model_family", "noise_model_name", "train_size", "noise_strength"]
    )

    # Partial conditions, mainly useful for n=9.
    partial = df[df["seeds_present"] < 3].copy().sort_values(
        by=["num_qubits", "model_family", "noise_model_name", "train_size", "noise_strength"]
    )

    n9 = df[df["num_qubits"] == 9].copy().sort_values(
        by=["model_family", "noise_model_name", "train_size", "noise_strength"]
    )
    n9_complete = n9[n9["seeds_present"] >= 3].copy()
    n9_partial = n9[n9["seeds_present"] < 3].copy()

    # Advisor-facing "best available" table:
    # keep full 3-seed conditions first; include partial n=9 separately.
    best_cols = [
        "num_qubits",
        "model_family",
        "noise_model_name",
        "train_size",
        "noise_strength",
        "seeds_present",
        "test_accuracy_mean",
        "test_accuracy_std",
        "test_loss_mean",
        "test_loss_std",
        "history_best_accuracy_mean",
        "history_best_accuracy_std",
        "history_best_loss_mean",
        "history_best_loss_std",
    ]
    best_available = df[best_cols].copy().sort_values(
        by=["num_qubits", "model_family", "noise_model_name", "train_size", "noise_strength"]
    )

    coverage.to_csv(output_dir / "coverage_table.csv", index=False)
    complete.to_csv(output_dir / "complete_conditions.csv", index=False)
    partial.to_csv(output_dir / "partial_conditions.csv", index=False)
    n9.to_csv(output_dir / "n9_all_conditions.csv", index=False)
    n9_complete.to_csv(output_dir / "n9_complete_conditions.csv", index=False)
    n9_partial.to_csv(output_dir / "n9_partial_conditions.csv", index=False)
    best_available.to_csv(output_dir / "best_available_results.csv", index=False)

    # Small LaTeX table for the strongest complete n=9 conditions.
    n9_latex = n9_complete[
        [
            "model_family", "noise_model_name", "train_size", "noise_strength",
            "seeds_present", "test_accuracy_mean", "test_accuracy_std",
            "test_loss_mean", "test_loss_std",
        ]
    ].copy()

    if not n9_latex.empty:
        n9_latex["test_accuracy"] = n9_latex.apply(
            lambda r: f'{r["test_accuracy_mean"]:.3f} $\\pm$ {r["test_accuracy_std"]:.3f}'
            if pd.notna(r["test_accuracy_std"])
            else f'{r["test_accuracy_mean"]:.3f}',
            axis=1,
        )
        n9_latex["test_loss"] = n9_latex.apply(
            lambda r: f'{r["test_loss_mean"]:.3f} $\\pm$ {r["test_loss_std"]:.3f}'
            if pd.notna(r["test_loss_std"])
            else f'{r["test_loss_mean"]:.3f}',
            axis=1,
        )
        n9_latex = n9_latex[
            ["model_family", "noise_model_name", "train_size", "noise_strength", "seeds_present", "test_accuracy", "test_loss"]
        ]
        latex_path = output_dir / "n9_complete_results_table.tex"
        with latex_path.open("w", encoding="utf-8") as f:
            f.write(n9_latex.to_latex(index=False, escape=False))
    else:
        latex_path = output_dir / "n9_complete_results_table.tex"
        latex_path.write_text("% No complete n=9 conditions found.\n", encoding="utf-8")

    print(f"Wrote derived tables to {output_dir}")


if __name__ == "__main__":
    main()
