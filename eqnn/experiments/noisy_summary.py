"""Utilities for recursively summarizing noisy comparison runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eqnn.experiments.noisy_comparison import (
    aggregate_noisy_comparison_runs,
    load_completed_noisy_comparison_runs,
)


def summarize_noisy_comparison_directory(
    input_dir: str | Path,
    *,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
    runs_output_json: str | Path | None = None,
) -> dict[str, Any]:
    input_path = Path(input_dir)
    run_rows = load_completed_noisy_comparison_runs(input_path)
    summary_rows = aggregate_noisy_comparison_runs(run_rows)

    resolved_output_json = input_path / "summary.json" if output_json is None else Path(output_json)
    resolved_output_csv = input_path / "summary.csv" if output_csv is None else Path(output_csv)
    resolved_runs_json = input_path / "runs.json" if runs_output_json is None else Path(runs_output_json)

    resolved_output_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_csv.parent.mkdir(parents=True, exist_ok=True)
    resolved_runs_json.parent.mkdir(parents=True, exist_ok=True)

    resolved_runs_json.write_text(json.dumps(run_rows, indent=2, sort_keys=True) + "\n")
    resolved_output_json.write_text(json.dumps(summary_rows, indent=2, sort_keys=True) + "\n")
    _write_tidy_summary_csv(resolved_output_csv, summary_rows)

    return {
        "runs": run_rows,
        "summary": summary_rows,
        "runs_output_json": str(resolved_runs_json.resolve()),
        "summary_output_json": str(resolved_output_json.resolve()),
        "summary_output_csv": str(resolved_output_csv.resolve()),
    }


def _write_tidy_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        fieldnames = (
            "backend_name",
            "model_family",
            "num_qubits",
            "train_size",
            "epochs",
            "noise_model_name",
            "noise_strength",
            "num_runs",
        )
    else:
        preferred_prefix = (
            "backend_name",
            "model_family",
            "num_qubits",
            "train_size",
            "epochs",
            "noise_model_name",
            "noise_strength",
            "num_runs",
        )
        remaining = sorted({key for row in rows for key in row.keys() if key not in preferred_prefix})
        fieldnames = tuple(preferred_prefix) + tuple(remaining)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["summarize_noisy_comparison_directory"]
