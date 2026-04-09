#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "su2_qcnn": "SU2-QCNN",
    "hea_qcnn": "HEA-QCNN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot noisy comparison summary results.")
    parser.add_argument(
        "summary_csv",
        type=Path,
        help="Path to noisy comparison summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write figures. Defaults to <summary_csv parent>/figures",
    )
    return parser.parse_args()


def _setup_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_data(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    df = pd.read_csv(input_csv)
    required = [
        "model_family",
        "num_qubits",
        "train_size",
        "noise_strength",
        "mean_test_accuracy",
        "mean_test_loss",
        "mean_runtime_seconds",
        "mean_train_accuracy",
        "mean_train_loss",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df.sort_values(["num_qubits", "train_size", "model_family", "noise_strength"]).copy()


def _panel_plot(
    df: pd.DataFrame,
    output_dir: Path,
    ycol: str,
    ylabel: str,
    title: str,
    filename: str,
    *,
    yscale: str = "linear",
    ylim: tuple[float, float] | None = None,
) -> None:
    panel_keys = sorted(
        df[["num_qubits", "train_size"]].drop_duplicates().itertuples(index=False, name=None)
    )
    n_panels = len(panel_keys)
    ncols = 2
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.0 * ncols, 4.3 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (num_qubits, train_size) in zip(axes, panel_keys):
        sub = df[(df["num_qubits"] == num_qubits) & (df["train_size"] == train_size)].copy()

        for model_family in ["su2_qcnn", "hea_qcnn"]:
            model_sub = sub[sub["model_family"] == model_family].sort_values("noise_strength")
            if model_sub.empty:
                continue

            ax.plot(
                model_sub["noise_strength"],
                model_sub[ycol],
                marker="o",
                linewidth=2,
                label=MODEL_LABELS.get(model_family, model_family),
            )

        ax.set_title(f"n = {num_qubits}, train size = {train_size}")
        ax.set_xlabel("Noise strength")
        ax.set_ylabel(ylabel)
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

    outpath = output_dir / filename
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outpath}")


def _gap_plot(
    wide_df: pd.DataFrame,
    output_dir: Path,
    ycol: str,
    ylabel: str,
    title: str,
    filename: str,
    *,
    hline_zero: bool = True,
) -> None:
    panel_keys = sorted(
        wide_df[["num_qubits", "train_size"]].drop_duplicates().itertuples(index=False, name=None)
    )
    n_panels = len(panel_keys)
    ncols = 2
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.0 * ncols, 4.3 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (num_qubits, train_size) in zip(axes, panel_keys):
        sub = wide_df[
            (wide_df["num_qubits"] == num_qubits) & (wide_df["train_size"] == train_size)
        ].sort_values("noise_strength")

        ax.plot(
            sub["noise_strength"],
            sub[ycol],
            marker="o",
            linewidth=2,
        )

        if hline_zero:
            ax.axhline(0.0, linewidth=1)

        ax.set_title(f"n = {num_qubits}, train size = {train_size}")
        ax.set_xlabel("Noise strength")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(title, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    outpath = output_dir / filename
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outpath}")


def main() -> None:
    args = parse_args()
    input_csv: Path = args.summary_csv.resolve()
    output_dir: Path = (
        args.output_dir.resolve() if args.output_dir is not None else input_csv.parent / "figures"
    )

    _setup_output_dir(output_dir)
    df = _load_data(input_csv)

    _panel_plot(
        df,
        output_dir,
        ycol="mean_test_accuracy",
        ylabel="Mean test accuracy",
        title="Noisy comparison: test accuracy vs noise strength",
        filename="noisy_test_accuracy_vs_noise.png",
        ylim=(0.5, 1.01),
    )

    _panel_plot(
        df,
        output_dir,
        ycol="mean_test_loss",
        ylabel="Mean test loss",
        title="Noisy comparison: test loss vs noise strength",
        filename="noisy_test_loss_vs_noise.png",
    )

    _panel_plot(
        df,
        output_dir,
        ycol="mean_runtime_seconds",
        ylabel="Mean runtime (seconds)",
        title="Noisy comparison: runtime vs noise strength",
        filename="noisy_runtime_vs_noise.png",
        yscale="log",
    )

    _panel_plot(
        df,
        output_dir,
        ycol="mean_train_accuracy",
        ylabel="Mean train accuracy",
        title="Noisy comparison: train accuracy vs noise strength",
        filename="noisy_train_accuracy_vs_noise.png",
        ylim=(0.5, 1.01),
    )

    _panel_plot(
        df,
        output_dir,
        ycol="mean_train_loss",
        ylabel="Mean train loss",
        title="Noisy comparison: train loss vs noise strength",
        filename="noisy_train_loss_vs_noise.png",
    )

    wide = df.pivot_table(
        index=["num_qubits", "train_size", "noise_strength"],
        columns="model_family",
        values=["mean_test_accuracy", "mean_test_loss", "mean_runtime_seconds"],
    )
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()

    if "mean_test_accuracy_su2_qcnn" in wide.columns and "mean_test_accuracy_hea_qcnn" in wide.columns:
        wide["delta_test_accuracy_su2_minus_hea"] = (
            wide["mean_test_accuracy_su2_qcnn"] - wide["mean_test_accuracy_hea_qcnn"]
        )
        _gap_plot(
            wide,
            output_dir,
            ycol="delta_test_accuracy_su2_minus_hea",
            ylabel="SU2 minus HEA test accuracy",
            title="Noise robustness gap: SU2-QCNN minus HEA-QCNN test accuracy",
            filename="noisy_accuracy_gap_su2_minus_hea.png",
        )

    if "mean_test_loss_su2_qcnn" in wide.columns and "mean_test_loss_hea_qcnn" in wide.columns:
        wide["delta_test_loss_su2_minus_hea"] = (
            wide["mean_test_loss_su2_qcnn"] - wide["mean_test_loss_hea_qcnn"]
        )
        _gap_plot(
            wide,
            output_dir,
            ycol="delta_test_loss_su2_minus_hea",
            ylabel="SU2 minus HEA test loss",
            title="Noise loss gap: SU2-QCNN minus HEA-QCNN test loss",
            filename="noisy_test_loss_gap_su2_minus_hea.png",
        )

    if "mean_runtime_seconds_su2_qcnn" in wide.columns and "mean_runtime_seconds_hea_qcnn" in wide.columns:
        wide["runtime_ratio_hea_over_su2"] = (
            wide["mean_runtime_seconds_hea_qcnn"] / wide["mean_runtime_seconds_su2_qcnn"]
        )
        _gap_plot(
            wide,
            output_dir,
            ycol="runtime_ratio_hea_over_su2",
            ylabel="Runtime ratio (HEA / SU2)",
            title="Mixed-state runtime ratio: HEA-QCNN over SU2-QCNN",
            filename="noisy_runtime_ratio_hea_over_su2.png",
            hline_zero=False,
        )


if __name__ == "__main__":
    main()