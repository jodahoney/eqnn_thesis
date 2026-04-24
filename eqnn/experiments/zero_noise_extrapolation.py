"""Post-hoc zero-noise extrapolation helpers for noisy comparison outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from eqnn.experiments.noisy_comparison import load_completed_noisy_comparison_runs


def fit_zero_noise_extrapolation(
    rows: list[dict[str, Any]],
    *,
    metric_name: str = "test_accuracy",
    fit_type: str = "linear",
) -> list[dict[str, Any]]:
    if fit_type != "linear":
        raise ValueError("Only fit_type='linear' is currently supported")

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        metric_value = row.get(metric_name)
        noise_strength = row.get("noise_strength")
        if metric_value is None or noise_strength is None:
            continue
        key = (
            row.get("backend_name"),
            row.get("model_family"),
            row.get("num_qubits"),
            row.get("train_size"),
            row.get("epochs"),
            row.get("seed"),
            row.get("noise_model_name"),
            row.get("noise_application_scope"),
            row.get("noisy_qubit_index"),
            row.get("coherent_overrotation_mode"),
        )
        grouped.setdefault(key, []).append(row)

    zne_rows: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        points: list[tuple[float, float]] = []
        for row in group_rows:
            try:
                x_value = float(row["noise_strength"])
                y_value = float(row[metric_name])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(x_value) or not np.isfinite(y_value):
                continue
            points.append((x_value, y_value))

        unique_x_values = sorted({x_value for x_value, _ in points})
        if len(points) < 2 or len(unique_x_values) < 2:
            continue

        x = np.asarray([point[0] for point in points], dtype=np.float64)
        y = np.asarray([point[1] for point in points], dtype=np.float64)
        slope, intercept = np.polyfit(x, y, deg=1)
        zne_rows.append(
            {
                "backend_name": key[0],
                "model_family": key[1],
                "num_qubits": key[2],
                "train_size": key[3],
                "epochs": key[4],
                "seed": key[5],
                "noise_model_name": key[6],
                "noise_application_scope": key[7],
                "noisy_qubit_index": key[8],
                "coherent_overrotation_mode": key[9],
                "zne_metric_name": metric_name,
                "zne_fit_type": fit_type,
                "zne_estimate": float(intercept),
                "zne_slope": float(slope),
                "zne_intercept": float(intercept),
                "zne_num_points": int(len(points)),
            }
        )

    zne_rows.sort(
        key=lambda row: (
            str(row.get("backend_name")),
            str(row.get("model_family")),
            -1 if row.get("num_qubits") is None else int(row["num_qubits"]),
            -1 if row.get("train_size") is None else int(row["train_size"]),
            -1 if row.get("epochs") is None else int(row["epochs"]),
            str(row.get("noise_model_name")),
            "" if row.get("noise_application_scope") is None else str(row["noise_application_scope"]),
            -1 if row.get("noisy_qubit_index") is None else int(row["noisy_qubit_index"]),
            "" if row.get("coherent_overrotation_mode") is None else str(row["coherent_overrotation_mode"]),
            -1 if row.get("seed") is None else int(row["seed"]),
        )
    )
    return zne_rows


def summarize_zero_noise_extrapolation_directory(
    input_dir: str | Path,
    *,
    metric_name: str = "test_accuracy",
    fit_type: str = "linear",
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> dict[str, Any]:
    input_path = Path(input_dir)
    run_rows = load_completed_noisy_comparison_runs(input_path)
    zne_rows = fit_zero_noise_extrapolation(run_rows, metric_name=metric_name, fit_type=fit_type)

    resolved_output_json = input_path / "zne_summary.json" if output_json is None else Path(output_json)
    resolved_output_csv = input_path / "zne_summary.csv" if output_csv is None else Path(output_csv)
    resolved_output_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_csv.parent.mkdir(parents=True, exist_ok=True)

    resolved_output_json.write_text(json.dumps(zne_rows, indent=2, sort_keys=True) + "\n")
    _write_zne_csv(resolved_output_csv, zne_rows)
    return {
        "zne_rows": zne_rows,
        "output_json": str(resolved_output_json.resolve()),
        "output_csv": str(resolved_output_csv.resolve()),
    }


def _write_zne_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        fieldnames = (
            "backend_name",
            "model_family",
            "num_qubits",
            "train_size",
            "epochs",
            "seed",
            "noise_model_name",
            "noise_application_scope",
            "noisy_qubit_index",
            "coherent_overrotation_mode",
            "zne_metric_name",
            "zne_fit_type",
            "zne_estimate",
            "zne_slope",
            "zne_intercept",
            "zne_num_points",
        )
    else:
        fieldnames = tuple(rows[0].keys())

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "fit_zero_noise_extrapolation",
    "summarize_zero_noise_extrapolation_directory",
]
