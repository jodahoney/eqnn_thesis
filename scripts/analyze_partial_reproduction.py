#!/usr/bin/env python3
"""
Analyze partial paper-reproduction outputs from the split Sherlock array jobs.

Expected directory structure:
  data/reproduction/<run_id>/n<N>/train_size_<T>/summary.csv

Each summary.csv is expected to contain exactly one row, produced by
run-paper-reproduction with a single train size and multiple seeds.

Optional:
  You can pass a sacct-export CSV/TSV to add runtime analyses.

Examples:
  python scripts/analyze_partial_reproduction.py \
      --input-dir /scratch/users/jdehoney/eqnn_thesis/data/reproduction/paper_reproduction_19558980 \
      --output-dir /scratch/users/jdehoney/eqnn_thesis/data/analysis/paper_reproduction_19558980

  sacct -j 19558980 --format=JobID,State,Elapsed,Timelimit -X -n -P > sacct_19558980.txt

  python scripts/analyze_partial_reproduction.py \
      --input-dir /scratch/users/jdehoney/eqnn_thesis/data/reproduction/paper_reproduction_19558980 \
      --output-dir /scratch/users/jdehoney/eqnn_thesis/data/analysis/paper_reproduction_19558980 \
      --sacct-file sacct_19558980.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_FIELDS_NUMERIC = [
    "train_size",
    "num_runs",
    "mean_train_accuracy",
    "variance_train_accuracy",
    "mean_test_accuracy",
    "variance_test_accuracy",
    "mean_train_loss",
    "variance_train_loss",
    "mean_test_loss",
    "variance_test_loss",
    "mean_threshold",
    "variance_threshold",
]


@dataclass(frozen=True)
class ParsedPath:
    num_qubits: int
    train_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sacct-file",
        type=Path,
        default=None,
        help="Optional pipe-delimited sacct output from: sacct -j <jobid> --format=JobID,State,Elapsed,Timelimit -X -n -P",
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.99,
        help="Threshold for sample-efficiency summary.",
    )
    return parser.parse_args()


def parse_summary_path(path: Path) -> ParsedPath:
    """
    Expect:
      .../n10/train_size_6/summary.csv
    """
    train_dir = path.parent
    n_dir = train_dir.parent

    train_match = re.fullmatch(r"train_size_(\d+)", train_dir.name)
    n_match = re.fullmatch(r"n(\d+)", n_dir.name)

    if train_match is None or n_match is None:
        raise ValueError(f"Could not parse N/train_size from path: {path}")

    return ParsedPath(
        num_qubits=int(n_match.group(1)),
        train_size=int(train_match.group(1)),
    )


def read_one_row_summary(path: Path) -> dict[str, Any]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}")

    row = rows[0]
    for key in SUMMARY_FIELDS_NUMERIC:
        row[key] = float(row[key]) if key != "train_size" and key != "num_runs" else int(float(row[key]))
    row["summary_csv"] = str(path.resolve())
    return row


def collect_summary_rows(input_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for summary_path in sorted(input_dir.glob("n*/train_size_*/summary.csv")):
        parsed = parse_summary_path(summary_path)
        row = read_one_row_summary(summary_path)

        if row["train_size"] != parsed.train_size:
            raise ValueError(
                f"Mismatch between path train_size={parsed.train_size} and CSV train_size={row['train_size']} in {summary_path}"
            )

        row["num_qubits"] = parsed.num_qubits
        row["train_size_path"] = parsed.train_size
        row["task_dir"] = str(summary_path.parent.resolve())

        runs_json = summary_path.parent / "runs.json"
        summary_json = summary_path.parent / "summary.json"
        config_json = summary_path.parent / "paper_reproduction_config.json"

        row["runs_json"] = str(runs_json.resolve()) if runs_json.exists() else None
        row["summary_json"] = str(summary_json.resolve()) if summary_json.exists() else None
        row["config_json"] = str(config_json.resolve()) if config_json.exists() else None

        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No summary.csv files found under {input_dir}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["num_qubits", "train_size"]).reset_index(drop=True)
    return df


def parse_elapsed_to_hours(s: str) -> float:
    """
    Slurm Elapsed can look like:
      MM:SS
      HH:MM:SS
      D-HH:MM:SS
    """
    s = s.strip()
    if not s:
        return math.nan

    days = 0
    if "-" in s:
        day_part, time_part = s.split("-", maxsplit=1)
        days = int(day_part)
    else:
        time_part = s

    parts = [int(x) for x in time_part.split(":")]
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Unrecognized elapsed format: {s}")

    total_hours = 24 * days + hours + minutes / 60.0 + seconds / 3600.0
    return total_hours


def infer_task_index(num_qubits: int, train_size: int, n_values: list[int], t_values: list[int]) -> int:
    n_index = n_values.index(num_qubits)
    t_index = t_values.index(train_size)
    return n_index * len(t_values) + t_index


def read_sacct(sacct_file: Path, n_values: list[int], t_values: list[int]) -> pd.DataFrame:
    rows = []
    with sacct_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            job_id, state, elapsed, timelimit = parts[:4]

            m = re.fullmatch(r".*_(\d+)", job_id)
            if m is None:
                continue

            task_id = int(m.group(1))
            n_index = task_id // len(t_values)
            t_index = task_id % len(t_values)
            if n_index >= len(n_values):
                continue

            rows.append(
                {
                    "task_id": task_id,
                    "num_qubits": n_values[n_index],
                    "train_size": t_values[t_index],
                    "state": state,
                    "elapsed": elapsed,
                    "timelimit": timelimit,
                    "elapsed_hours": parse_elapsed_to_hours(elapsed),
                    "timelimit_hours": parse_elapsed_to_hours(timelimit),
                }
            )

    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_markdown_summary(df: pd.DataFrame, output_path: Path, accuracy_threshold: float) -> None:
    lines: list[str] = []

    completed = len(df)
    n_values = sorted(df["num_qubits"].unique().tolist())
    t_values = sorted(df["train_size"].unique().tolist())

    lines.append("# Partial reproduction analysis")
    lines.append("")
    lines.append(f"- Completed `(N, train_size)` tasks found: **{completed}**")
    lines.append(f"- System sizes present: **{n_values}**")
    lines.append(f"- Train sizes present: **{t_values}**")
    lines.append("")

    grouped_n = df.groupby("num_qubits")
    lines.append("## Per-N overview")
    lines.append("")
    for n, sub in grouped_n:
        best_acc = sub["mean_test_accuracy"].max()
        min_train_loss = sub["mean_train_loss"].min()
        min_test_loss = sub["mean_test_loss"].min()
        lines.append(
            f"- **N={n}**: completed train sizes = {sorted(sub['train_size'].tolist())}, "
            f"best mean test accuracy = {best_acc:.4f}, "
            f"min mean train loss = {min_train_loss:.6f}, "
            f"min mean test loss = {min_test_loss:.6f}"
        )
    lines.append("")

    sample_eff = (
        df[df["mean_test_accuracy"] >= accuracy_threshold]
        .groupby("num_qubits", as_index=False)["train_size"]
        .min()
        .rename(columns={"train_size": f"min_train_size_for_acc_ge_{accuracy_threshold:.2f}"})
    )

    lines.append(f"## Sample efficiency at accuracy >= {accuracy_threshold:.2f}")
    lines.append("")
    if sample_eff.empty:
        lines.append("No completed point reached the requested threshold.")
    else:
        for _, row in sample_eff.iterrows():
            lines.append(
                f"- **N={int(row['num_qubits'])}**: minimum observed train size = "
                f"**{int(row[f'min_train_size_for_acc_ge_{accuracy_threshold:.2f}'])}**"
            )
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n")


def line_plot_by_n(
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    for n in sorted(df["num_qubits"].unique()):
        sub = df[df["num_qubits"] == n].sort_values("train_size")
        plt.plot(sub["train_size"], sub[y_col], marker="o", label=f"N={n}")
    plt.xlabel("Train size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    output_path: Path,
    cmap: str = "viridis",
) -> None:
    pivot = df.pivot(index="num_qubits", columns="train_size", values=value_col).sort_index().sort_index(axis=1)

    plt.figure(figsize=(8, 5))
    plt.imshow(pivot.values, aspect="auto", interpolation="nearest", cmap=cmap)
    plt.colorbar(label=value_col)
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Train size")
    plt.ylabel("Num qubits")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def runtime_plot(runtime_df: pd.DataFrame, output_dir: Path) -> None:
    if runtime_df.empty:
        return

    completed = runtime_df[runtime_df["state"] == "COMPLETED"].copy()
    if completed.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(completed["task_id"], completed["elapsed_hours"])
    plt.xlabel("Array task id")
    plt.ylabel("Elapsed time (hours)")
    plt.title("Completed task runtime vs array task id")
    plt.tight_layout()
    plt.savefig(output_dir / "runtime_vs_task_id.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for n in sorted(completed["num_qubits"].unique()):
        sub = completed[completed["num_qubits"] == n].sort_values("train_size")
        plt.plot(sub["train_size"], sub["elapsed_hours"], marker="o", label=f"N={n}")
    plt.xlabel("Train size")
    plt.ylabel("Elapsed time (hours)")
    plt.title("Runtime vs train size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "runtime_vs_train_size.png", dpi=200)
    plt.close()

    heatmap(
        completed,
        value_col="elapsed_hours",
        title="Runtime heatmap (completed tasks only)",
        output_path=output_dir / "runtime_heatmap.png",
        cmap="magma",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_summary_rows(args.input_dir)
    save_table(df, args.output_dir / "combined_summary.csv")

    write_markdown_summary(
        df,
        args.output_dir / "analysis_summary.md",
        accuracy_threshold=args.accuracy_threshold,
    )

    line_plot_by_n(
        df,
        y_col="mean_test_accuracy",
        ylabel="Mean test accuracy",
        title="Mean test accuracy vs train size",
        output_path=args.output_dir / "mean_test_accuracy_vs_train_size.png",
    )
    line_plot_by_n(
        df,
        y_col="mean_threshold",
        ylabel="Mean classification threshold",
        title="Mean threshold vs train size",
        output_path=args.output_dir / "mean_threshold_vs_train_size.png",
    )
    line_plot_by_n(
        df,
        y_col="mean_test_loss",
        ylabel="Mean test loss",
        title="Mean test loss vs train size",
        output_path=args.output_dir / "mean_test_loss_vs_train_size.png",
    )
    line_plot_by_n(
        df,
        y_col="variance_test_accuracy",
        ylabel="Variance of test accuracy",
        title="Variance of test accuracy vs train size",
        output_path=args.output_dir / "variance_test_accuracy_vs_train_size.png",
    )

    heatmap(
        df,
        value_col="mean_test_accuracy",
        title="Mean test accuracy heatmap",
        output_path=args.output_dir / "mean_test_accuracy_heatmap.png",
    )
    heatmap(
        df,
        value_col="mean_threshold",
        title="Mean threshold heatmap",
        output_path=args.output_dir / "mean_threshold_heatmap.png",
    )
    heatmap(
        df,
        value_col="mean_test_loss",
        title="Mean test loss heatmap",
        output_path=args.output_dir / "mean_test_loss_heatmap.png",
    )

    sample_eff = (
        df[df["mean_test_accuracy"] >= args.accuracy_threshold]
        .groupby("num_qubits", as_index=False)["train_size"]
        .min()
        .rename(columns={"train_size": "min_train_size"})
    )
    save_table(sample_eff, args.output_dir / "sample_efficiency.csv")

    if args.sacct_file is not None:
        n_values = [6, 7, 8, 9, 10, 11, 12, 13]
        t_values = [2, 4, 6, 8, 10, 12]
        runtime_df = read_sacct(args.sacct_file, n_values=n_values, t_values=t_values)
        save_table(runtime_df, args.output_dir / "runtime_table.csv")
        runtime_plot(runtime_df, args.output_dir)

        merged = df.merge(runtime_df, on=["num_qubits", "train_size"], how="left")
        save_table(merged, args.output_dir / "combined_summary_with_runtime.csv")

    print(f"Wrote analysis outputs to {args.output_dir}")


if __name__ == "__main__":
    main()