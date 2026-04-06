#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_CSV = Path("data/comparisons/meta_summary/model_comparison_wide.csv")
OUTPUT_DIR = Path("data/comparisons/meta_summary/figures")


def _setup_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    df = df.sort_values(["num_qubits", "train_size"]).copy()

    required_cols = [
        "num_qubits",
        "train_size",
        "mean_test_accuracy_su2_qcnn",
        "mean_test_accuracy_hea_qcnn",
        "mean_test_loss_su2_qcnn",
        "mean_test_loss_hea_qcnn",
        "mean_runtime_seconds_su2_qcnn",
        "mean_runtime_seconds_hea_qcnn",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def _panel_plot(
    df: pd.DataFrame,
    y_su2: str,
    y_hea: str,
    ylabel: str,
    title: str,
    filename: str,
    *,
    yscale: str = "linear",
    ylim: tuple[float, float] | None = None,
) -> None:
    num_qubits_values = sorted(df["num_qubits"].unique().tolist())
    n_panels = len(num_qubits_values)
    ncols = 3
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, num_qubits in zip(axes, num_qubits_values):
        sub = df[df["num_qubits"] == num_qubits].sort_values("train_size")

        ax.plot(
            sub["train_size"],
            sub[y_su2],
            marker="o",
            linewidth=2,
            label="SU2-QCNN",
        )
        ax.plot(
            sub["train_size"],
            sub[y_hea],
            marker="s",
            linewidth=2,
            label="HEA-QCNN",
        )

        ax.set_title(f"n = {num_qubits}")
        ax.set_xlabel("Train size")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sub["train_size"].tolist())
        ax.set_yscale(yscale)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)

    for ax in axes[n_panels:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.suptitle(title, y=0.985)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    outpath = OUTPUT_DIR / filename
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outpath}")


def _single_metric_gap_plot(
    df: pd.DataFrame,
    y_gap: str,
    ylabel: str,
    title: str,
    filename: str,
    *,
    hline_zero: bool = True,
) -> None:
    num_qubits_values = sorted(df["num_qubits"].unique().tolist())
    n_panels = len(num_qubits_values)
    ncols = 3
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, num_qubits in zip(axes, num_qubits_values):
        sub = df[df["num_qubits"] == num_qubits].sort_values("train_size")

        ax.plot(
            sub["train_size"],
            sub[y_gap],
            marker="o",
            linewidth=2,
        )

        if hline_zero:
            ax.axhline(0.0, linewidth=1)

        ax.set_title(f"n = {num_qubits}")
        ax.set_xlabel("Train size")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sub["train_size"].tolist())
        ax.grid(True, alpha=0.3)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(title, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    outpath = OUTPUT_DIR / filename
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outpath}")


def _heatmap(
    pivot_df: pd.DataFrame,
    title: str,
    filename: str,
    cbar_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot_df.values, aspect="auto")

    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_xticklabels([str(x) for x in pivot_df.columns.tolist()])
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_yticklabels([str(y) for y in pivot_df.index.tolist()])

    ax.set_xlabel("Train size")
    ax.set_ylabel("Num qubits")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    for i in range(pivot_df.shape[0]):
        for j in range(pivot_df.shape[1]):
            val = pivot_df.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    outpath = OUTPUT_DIR / filename
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outpath}")


def main() -> None:
    _setup_output_dir()
    df = _load_data()

    if "delta_test_accuracy_su2_minus_hea" not in df.columns:
        df["delta_test_accuracy_su2_minus_hea"] = (
            df["mean_test_accuracy_su2_qcnn"] - df["mean_test_accuracy_hea_qcnn"]
        )

    if "delta_test_loss_su2_minus_hea" not in df.columns:
        df["delta_test_loss_su2_minus_hea"] = (
            df["mean_test_loss_su2_qcnn"] - df["mean_test_loss_hea_qcnn"]
        )

    if "runtime_ratio_hea_over_su2" not in df.columns:
        df["runtime_ratio_hea_over_su2"] = (
            df["mean_runtime_seconds_hea_qcnn"] / df["mean_runtime_seconds_su2_qcnn"]
        )

    _panel_plot(
        df,
        y_su2="mean_test_accuracy_su2_qcnn",
        y_hea="mean_test_accuracy_hea_qcnn",
        ylabel="Mean test accuracy",
        title="EQNN vs HEA-QCNN: Mean test accuracy vs train size",
        filename="eqnn_vs_hea_test_accuracy_by_n.png",
        ylim=(0.6, 1.02),
    )

    _panel_plot(
        df,
        y_su2="mean_test_loss_su2_qcnn",
        y_hea="mean_test_loss_hea_qcnn",
        ylabel="Mean test loss",
        title="EQNN vs HEA-QCNN: Mean test loss vs train size",
        filename="eqnn_vs_hea_test_loss_by_n.png",
    )

    _panel_plot(
        df,
        y_su2="mean_runtime_seconds_su2_qcnn",
        y_hea="mean_runtime_seconds_hea_qcnn",
        ylabel="Mean runtime (seconds)",
        title="EQNN vs HEA-QCNN: Mean runtime vs train size",
        filename="eqnn_vs_hea_runtime_by_n.png",
        yscale="log",
    )

    _single_metric_gap_plot(
        df,
        y_gap="delta_test_accuracy_su2_minus_hea",
        ylabel="SU2 minus HEA test accuracy",
        title="Generalization gap: SU2-QCNN minus HEA-QCNN test accuracy",
        filename="eqnn_minus_hea_accuracy_gap_by_n.png",
    )

    _single_metric_gap_plot(
        df,
        y_gap="delta_test_loss_su2_minus_hea",
        ylabel="SU2 minus HEA test loss",
        title="Loss gap: SU2-QCNN minus HEA-QCNN test loss",
        filename="eqnn_minus_hea_test_loss_gap_by_n.png",
    )

    acc_gap = df.pivot(
        index="num_qubits",
        columns="train_size",
        values="delta_test_accuracy_su2_minus_hea",
    ).sort_index().sort_index(axis=1)

    _heatmap(
        acc_gap,
        title="Heatmap: SU2 minus HEA test accuracy",
        filename="heatmap_accuracy_gap_su2_minus_hea.png",
        cbar_label="Accuracy gap",
    )

    runtime_ratio = df.pivot(
        index="num_qubits",
        columns="train_size",
        values="runtime_ratio_hea_over_su2",
    ).sort_index().sort_index(axis=1)

    _heatmap(
        runtime_ratio,
        title="Heatmap: runtime ratio (HEA / SU2)",
        filename="heatmap_runtime_ratio_hea_over_su2.png",
        cbar_label="Runtime ratio",
    )


if __name__ == "__main__":
    main()