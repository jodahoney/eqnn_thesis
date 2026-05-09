#!/usr/bin/env python3
"""Analyze all-noise mitigation comparison results.

Example:
    python3 scripts/analysis/analyze_all_noise_mitigation.py \
      --summary-csv data/noisy_comparisons/all_noise_mitigation_n7_array/summary.csv \
      --output-dir data/noisy_comparisons/all_noise_mitigation_n7_array/analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_SUMMARY_CSV = (
    "data/noisy_comparisons/all_noise_mitigation_n7_array/summary.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create core tables and figures for all-noise mitigation results."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path(DEFAULT_SUMMARY_CSV),
        help="Path to aggregated summary.csv from summarize-noisy-comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for tables and figures. Defaults to <summary parent>/analysis.",
    )
    parser.add_argument(
        "--metric",
        default="mean_test_accuracy",
        help="Metric column to analyze.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI.",
    )
    return parser.parse_args()


def safe_beta_label(value: object) -> str:
    if pd.isna(value):
        return "none"
    value_f = float(value)
    return f"{value_f:g}"


def ensure_required_columns(df: pd.DataFrame, metric: str) -> None:
    required = [
        "model_family",
        "train_size",
        "noise_model_name",
        "noise_strength",
        "mitigation_method",
        metric,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns from summary.csv: {missing}")


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Make sure these fields exist even if older summaries omit them.
    if "symmetry_regularization_beta" not in df.columns:
        df["symmetry_regularization_beta"] = pd.NA
    if "expected_symmetry_breaking" not in df.columns:
        df["expected_symmetry_breaking"] = "unknown"
    if "eval_noise_strength" not in df.columns:
        df["eval_noise_strength"] = df["noise_strength"]

    # Keep a cleaner plotting label.
    model_labels = {
        "su2_qcnn": "SU2-QCNN",
        "hea_qcnn": "HEA-QCNN",
    }
    df["model_label"] = df["model_family"].map(model_labels).fillna(df["model_family"])

    noise_labels = {
        "depolarizing": "Depolarizing",
        "phase_damping": "Phase damping",
        "amplitude_damping": "Amplitude damping",
        "coherent_overrotation": "Coherent over-rotation",
    }
    df["noise_label"] = (
        df["noise_model_name"].map(noise_labels).fillna(df["noise_model_name"])
    )

    mitigation_labels = {
        "none": "None",
        "noise_aware_training": "Noise-aware training",
        "symmetry_regularized": "Symmetry regularized",
        "noise_aware_symmetry_regularized": "Noise-aware + symmetry regularized",
    }
    df["mitigation_label"] = (
        df["mitigation_method"].map(mitigation_labels).fillna(df["mitigation_method"])
    )

    df["beta_label"] = df["symmetry_regularization_beta"].map(safe_beta_label)

    return df


def aggregate_for_core_tables(df: pd.DataFrame, metric: str) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    # 1. Noise effect under no mitigation.
    baseline = df[df["mitigation_method"] == "none"].copy()
    noise_effect = (
        baseline.groupby(
            [
                "model_family",
                "model_label",
                "train_size",
                "noise_model_name",
                "noise_label",
                "noise_strength",
            ],
            dropna=False,
        )[metric]
        .mean()
        .reset_index()
        .sort_values(
            [
                "model_family",
                "train_size",
                "noise_model_name",
                "noise_strength",
            ]
        )
    )

    clean = (
        noise_effect[noise_effect["noise_strength"] == 0.0]
        .rename(columns={metric: "clean_accuracy"})
        [
            [
                "model_family",
                "train_size",
                "noise_model_name",
                "clean_accuracy",
            ]
        ]
    )
    noise_effect = noise_effect.merge(
        clean,
        on=["model_family", "train_size", "noise_model_name"],
        how="left",
    )
    noise_effect["accuracy_drop_from_clean"] = (
        noise_effect["clean_accuracy"] - noise_effect[metric]
    )
    tables["noise_effect_by_model"] = noise_effect

    # 2. Mitigation delta relative to no mitigation.
    # Average beta values together for the main mitigation summary.
    group_cols = [
        "model_family",
        "model_label",
        "train_size",
        "noise_model_name",
        "noise_label",
        "noise_strength",
        "mitigation_method",
        "mitigation_label",
    ]
    avg = df.groupby(group_cols, dropna=False)[metric].mean().reset_index()

    base = (
        avg[avg["mitigation_method"] == "none"]
        .rename(columns={metric: "baseline_accuracy"})
        .drop(columns=["mitigation_method", "mitigation_label"])
    )

    mitigation_delta = avg.merge(
        base,
        on=[
            "model_family",
            "model_label",
            "train_size",
            "noise_model_name",
            "noise_label",
            "noise_strength",
        ],
        how="left",
    )
    mitigation_delta["delta_vs_none"] = (
        mitigation_delta[metric] - mitigation_delta["baseline_accuracy"]
    )
    tables["mitigation_delta_by_noise"] = mitigation_delta.sort_values(
        [
            "model_family",
            "train_size",
            "noise_model_name",
            "noise_strength",
            "mitigation_method",
        ]
    )

    # 3. Average mitigation delta by train size.
    mitigation_by_train_size = (
        mitigation_delta[mitigation_delta["mitigation_method"] != "none"]
        .groupby(
            [
                "model_family",
                "model_label",
                "train_size",
                "noise_model_name",
                "noise_label",
                "mitigation_method",
                "mitigation_label",
            ],
            dropna=False,
        )
        .agg(
            mean_accuracy=(metric, "mean"),
            mean_baseline_accuracy=("baseline_accuracy", "mean"),
            mean_delta_vs_none=("delta_vs_none", "mean"),
        )
        .reset_index()
        .sort_values(
            [
                "model_family",
                "train_size",
                "noise_model_name",
                "mean_delta_vs_none",
            ],
            ascending=[True, True, True, False],
        )
    )
    tables["mitigation_delta_by_train_size"] = mitigation_by_train_size

    # 4. Symmetry beta sweep.
    sym = df[df["mitigation_method"] == "symmetry_regularized"].copy()
    if not sym.empty:
        sym_beta = (
            sym.groupby(
                [
                    "model_family",
                    "model_label",
                    "train_size",
                    "noise_model_name",
                    "noise_label",
                    "noise_strength",
                    "symmetry_regularization_beta",
                    "beta_label",
                ],
                dropna=False,
            )[metric]
            .mean()
            .reset_index()
        )

        sym_base = (
            baseline.groupby(
                [
                    "model_family",
                    "train_size",
                    "noise_model_name",
                    "noise_strength",
                ],
                dropna=False,
            )[metric]
            .mean()
            .reset_index()
            .rename(columns={metric: "baseline_accuracy"})
        )

        sym_beta = sym_beta.merge(
            sym_base,
            on=[
                "model_family",
                "train_size",
                "noise_model_name",
                "noise_strength",
            ],
            how="left",
        )
        sym_beta["delta_vs_none"] = sym_beta[metric] - sym_beta["baseline_accuracy"]
        tables["symmetry_beta_sweep"] = sym_beta.sort_values(
            [
                "model_family",
                "train_size",
                "noise_model_name",
                "noise_strength",
                "symmetry_regularization_beta",
            ]
        )
    else:
        tables["symmetry_beta_sweep"] = pd.DataFrame()

    # 5. Best mitigation by setting.
    best = mitigation_delta.copy()
    best = best.sort_values(
        [
            "model_family",
            "train_size",
            "noise_model_name",
            "noise_strength",
            "delta_vs_none",
        ],
        ascending=[True, True, True, True, False],
    )
    best = best.groupby(
        ["model_family", "train_size", "noise_model_name", "noise_strength"],
        dropna=False,
    ).head(1)
    tables["best_mitigation_by_setting"] = best

    return tables


def save_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for name, table in tables.items():
        path = tables_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        print(f"Wrote {path}")


def plot_accuracy_vs_noise(df: pd.DataFrame, metric: str, figures_dir: Path, dpi: int) -> None:
    # Average over train sizes, seeds already aggregated in summary.
    plot_df = (
        df.groupby(
            [
                "model_label",
                "noise_label",
                "noise_strength",
                "mitigation_method",
                "mitigation_label",
            ],
            dropna=False,
        )[metric]
        .mean()
        .reset_index()
    )

    noise_labels = list(plot_df["noise_label"].drop_duplicates())
    model_labels = list(plot_df["model_label"].drop_duplicates())

    for model_label in model_labels:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 8),
            sharex=True,
            sharey=True,
        )
        axes = axes.ravel()

        for ax, noise_label in zip(axes, noise_labels):
            subset = plot_df[
                (plot_df["model_label"] == model_label)
                & (plot_df["noise_label"] == noise_label)
            ]

            for mitigation_label, group in subset.groupby("mitigation_label"):
                group = group.sort_values("noise_strength")
                ax.plot(
                    group["noise_strength"],
                    group[metric],
                    marker="o",
                    label=mitigation_label,
                )

            ax.set_title(noise_label)
            ax.set_xlabel("Noise strength")
            ax.set_ylabel("Mean test accuracy")
            ax.set_ylim(0.45, 1.02)
            ax.grid(True, alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3)
        fig.suptitle(f"{model_label}: accuracy vs noise strength", y=0.98)
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))

        path = figures_dir / f"accuracy_vs_noise_{model_label.lower().replace('-', '_')}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        print(f"Wrote {path}")


def plot_mitigation_delta(
    mitigation_delta: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> None:
    plot_df = mitigation_delta[mitigation_delta["mitigation_method"] != "none"].copy()

    # Average over train size.
    plot_df = (
        plot_df.groupby(
            [
                "model_label",
                "noise_label",
                "noise_strength",
                "mitigation_label",
            ],
            dropna=False,
        )["delta_vs_none"]
        .mean()
        .reset_index()
    )

    model_labels = list(plot_df["model_label"].drop_duplicates())
    noise_labels = list(plot_df["noise_label"].drop_duplicates())

    for model_label in model_labels:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 8),
            sharex=True,
            sharey=True,
        )
        axes = axes.ravel()

        for ax, noise_label in zip(axes, noise_labels):
            subset = plot_df[
                (plot_df["model_label"] == model_label)
                & (plot_df["noise_label"] == noise_label)
            ]

            for mitigation_label, group in subset.groupby("mitigation_label"):
                group = group.sort_values("noise_strength")
                ax.plot(
                    group["noise_strength"],
                    group["delta_vs_none"],
                    marker="o",
                    label=mitigation_label,
                )

            ax.axhline(0.0, linestyle="--", linewidth=1)
            ax.set_title(noise_label)
            ax.set_xlabel("Noise strength")
            ax.set_ylabel("Accuracy delta vs no mitigation")
            ax.grid(True, alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3)
        fig.suptitle(f"{model_label}: mitigation effect vs no mitigation", y=0.98)
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))

        path = figures_dir / f"mitigation_delta_vs_noise_{model_label.lower().replace('-', '_')}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        print(f"Wrote {path}")


def plot_symmetry_beta_sweep(
    sym_beta: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> None:
    if sym_beta.empty:
        print("No symmetry beta sweep rows found; skipping beta plot.")
        return

    # Average over train size and noise strength to get one compact beta summary.
    plot_df = (
        sym_beta.groupby(
            [
                "model_label",
                "noise_label",
                "symmetry_regularization_beta",
            ],
            dropna=False,
        )
        .agg(
            mean_accuracy=("mean_test_accuracy", "mean"),
            mean_delta_vs_none=("delta_vs_none", "mean"),
        )
        .reset_index()
    )

    model_labels = list(plot_df["model_label"].drop_duplicates())

    for model_label in model_labels:
        fig, ax = plt.subplots(figsize=(8, 5))

        subset = plot_df[plot_df["model_label"] == model_label]
        for noise_label, group in subset.groupby("noise_label"):
            group = group.sort_values("symmetry_regularization_beta")
            ax.plot(
                group["symmetry_regularization_beta"],
                group["mean_delta_vs_none"],
                marker="o",
                label=noise_label,
            )

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel("Symmetry regularization beta")
        ax.set_ylabel("Mean accuracy delta vs no mitigation")
        ax.set_title(f"{model_label}: beta sweep effect")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        path = figures_dir / f"symmetry_beta_sweep_{model_label.lower().replace('-', '_')}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        print(f"Wrote {path}")


def plot_noise_aware_delta(
    mitigation_delta: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> None:
    plot_df = mitigation_delta[
        mitigation_delta["mitigation_method"] == "noise_aware_training"
    ].copy()

    if plot_df.empty:
        print("No noise-aware rows found; skipping noise-aware delta plot.")
        return

    plot_df = (
        plot_df.groupby(
            [
                "model_label",
                "noise_label",
                "noise_strength",
            ],
            dropna=False,
        )["delta_vs_none"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    for (model_label, noise_label), group in plot_df.groupby(["model_label", "noise_label"]):
        group = group.sort_values("noise_strength")
        ax.plot(
            group["noise_strength"],
            group["delta_vs_none"],
            marker="o",
            label=f"{model_label}, {noise_label}",
        )

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Noise strength")
    ax.set_ylabel("Noise-aware training delta vs no mitigation")
    ax.set_title("Effect of noise-aware training")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    path = figures_dir / "noise_aware_delta_vs_noise.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {path}")


def make_figures(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    metric: str,
    dpi: int,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_vs_noise(df, metric, figures_dir, dpi)
    plot_mitigation_delta(tables["mitigation_delta_by_noise"], figures_dir, dpi)
    plot_symmetry_beta_sweep(tables["symmetry_beta_sweep"], figures_dir, dpi)
    plot_noise_aware_delta(tables["mitigation_delta_by_noise"], figures_dir, dpi)


def print_console_summary(tables: dict[str, pd.DataFrame], metric: str) -> None:
    print("\n===== Core average mitigation delta vs none =====")
    mitigation_delta = tables["mitigation_delta_by_noise"]
    summary = (
        mitigation_delta[mitigation_delta["mitigation_method"] != "none"]
        .groupby(
            [
                "model_family",
                "noise_model_name",
                "mitigation_method",
            ],
            dropna=False,
        )["delta_vs_none"]
        .mean()
        .sort_values(ascending=False)
    )
    print(summary.to_string())

    print("\n===== Largest average noise drops from clean, no mitigation =====")
    noise_effect = tables["noise_effect_by_model"]
    high_noise = (
        noise_effect.groupby(
            [
                "model_family",
                "noise_model_name",
                "noise_strength",
            ],
            dropna=False,
        )["accuracy_drop_from_clean"]
        .mean()
        .reset_index()
        .sort_values("accuracy_drop_from_clean", ascending=False)
        .head(20)
    )
    print(high_noise.to_string(index=False))

    print("\n===== Best mitigation by setting preview =====")
    best = tables["best_mitigation_by_setting"]
    print(
        best[
            [
                "model_family",
                "train_size",
                "noise_model_name",
                "noise_strength",
                "mitigation_method",
                metric,
                "baseline_accuracy",
                "delta_vs_none",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


def main() -> None:
    args = parse_args()

    summary_csv = args.summary_csv
    output_dir = args.output_dir or summary_csv.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)
    ensure_required_columns(df, args.metric)
    df = normalize_dataframe(df)

    tables = aggregate_for_core_tables(df, args.metric)
    save_tables(tables, output_dir)
    make_figures(df, tables, output_dir, args.metric, args.dpi)
    print_console_summary(tables, args.metric)

    print("\nDone.")
    print(f"Analysis written to: {output_dir}")


if __name__ == "__main__":
    main()