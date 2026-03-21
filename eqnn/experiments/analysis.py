"""Utilities for aggregating completed experiment directories."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_DEFAULT_GROUP_BY = ("num_qubits", "model_family", "pooling_mode", "readout_mode")


def summarize_experiment_directory(
    experiment_root: str | Path,
    *,
    group_by: Iterable[str] = _DEFAULT_GROUP_BY,
    filters: dict[str, Any] | None = None,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Aggregate completed experiment metrics into comparison-ready rows."""

    root_path = Path(experiment_root)
    metrics_paths = sorted(root_path.rglob("metrics.json"))
    if not metrics_paths:
        raise ValueError(f"No metrics.json files were found under {root_path}")

    group_fields = tuple(str(field) for field in group_by)
    filter_items = dict(filters or {})

    grouped_metrics: dict[tuple[Any, ...], list[tuple[Path, dict[str, Any]]]] = {}
    for metrics_path in metrics_paths:
        metrics = json.loads(metrics_path.read_text())
        experiment_config = metrics["experiment_config"]
        if any(experiment_config.get(name) != value for name, value in filter_items.items()):
            continue

        key = tuple(experiment_config.get(field) for field in group_fields)
        grouped_metrics.setdefault(key, []).append((metrics_path, metrics))

    if not grouped_metrics:
        raise ValueError("No completed experiments matched the requested filters")

    summary_rows: list[dict[str, Any]] = []
    for key, metrics_records in grouped_metrics.items():
        row = {field: value for field, value in zip(group_fields, key)}
        row["num_runs"] = len(metrics_records)

        for metric_name in ("train_accuracy", "train_loss", "test_accuracy", "test_loss"):
            metric_values = np.asarray(
                [_lookup_metric(metrics, metric_name) for _, metrics in metrics_records],
                dtype=np.float64,
            )
            row[f"mean_{metric_name}"] = float(np.mean(metric_values))
            row[f"std_{metric_name}"] = float(np.std(metric_values))

        row["experiment_names"] = [metrics["experiment_name"] for _, metrics in metrics_records]
        row["output_dirs"] = [
            str(Path(metrics["output_dir"]).resolve()) if "output_dir" in metrics else str(metrics_path.parent.resolve())
            for metrics_path, metrics in metrics_records
        ]
        summary_rows.append(row)

    summary_rows.sort(
        key=lambda row: (
            -float(row["mean_test_accuracy"]),
            float(row["mean_test_loss"]),
            *(str(row[field]) for field in group_fields),
        )
    )

    if output_json is not None:
        Path(output_json).write_text(json.dumps(summary_rows, indent=2, sort_keys=True) + "\n")
    if output_csv is not None:
        _write_summary_csv(Path(output_csv), summary_rows, group_fields)

    return summary_rows


def _lookup_metric(metrics: dict[str, Any], metric_name: str) -> float:
    split_name, metric_field = metric_name.split("_", maxsplit=1)
    return float(metrics[f"{split_name}_metrics"][metric_field])


def _write_summary_csv(
    path: Path,
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
) -> None:
    fieldnames = list(group_fields) + [
        "num_runs",
        "mean_train_accuracy",
        "std_train_accuracy",
        "mean_train_loss",
        "std_train_loss",
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_test_loss",
        "std_test_loss",
        "experiment_names",
        "output_dirs",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["experiment_names"] = json.dumps(row["experiment_names"])
            csv_row["output_dirs"] = json.dumps(row["output_dirs"])
            writer.writerow(csv_row)
