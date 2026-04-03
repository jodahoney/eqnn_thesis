"""Small backend comparison utilities for runtime-oriented EQNN experiments."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from eqnn.datasets.heisenberg import HeisenbergDatasetConfig, generate_dataset
from eqnn.datasets.io import save_dataset_bundle
from eqnn.experiments.runner import ExperimentConfig, run_training_experiment
from eqnn.training import TrainingConfig
from eqnn.utils.timing import RuntimeProfile


@dataclass(frozen=True)
class BackendBenchmarkConfig:
    backend_names: tuple[str, ...]
    dataset_config: HeisenbergDatasetConfig
    experiment_config: ExperimentConfig
    training_config: TrainingConfig

    def __post_init__(self) -> None:
        if not self.backend_names:
            raise ValueError("backend_names must not be empty")
        invalid = tuple(
            backend_name for backend_name in self.backend_names if backend_name not in {"numpy_pure", "torch_pure"}
        )
        if invalid:
            raise ValueError(f"Unsupported backend_names: {invalid}")


def run_backend_benchmark(
    config: BackendBenchmarkConfig,
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "backend_benchmark_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n"
    )

    dataset = generate_dataset(config.dataset_config)
    save_dataset_bundle(dataset, output_path / "dataset")

    rows: list[dict[str, Any]] = []
    for backend_name in config.backend_names:
        profile = RuntimeProfile()
        backend_experiment_config = replace(config.experiment_config, backend_name=backend_name)
        experiment_output_dir = output_path / "experiments" / backend_name
        start_time = time.perf_counter()
        result = run_training_experiment(
            dataset,
            backend_experiment_config,
            config.training_config,
            output_dir=experiment_output_dir,
            experiment_name=f"backend_benchmark_{backend_name}",
            profile=profile,
        )
        runtime_seconds = float(time.perf_counter() - start_time)
        row = {
            "backend_name": backend_name,
            "model_family": backend_experiment_config.model_family,
            "num_qubits": backend_experiment_config.num_qubits,
            "train_accuracy": float(result["train_metrics"]["accuracy"]),
            "train_loss": float(result["train_metrics"]["loss"]),
            "test_accuracy": float(result["test_metrics"]["accuracy"]),
            "test_loss": float(result["test_metrics"]["loss"]),
            "runtime_seconds": runtime_seconds,
            "output_dir": str(experiment_output_dir.resolve()),
        }
        rows.append(row)

    rows.sort(key=lambda row: str(row["backend_name"]))
    (output_path / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    _write_summary_csv(output_path / "summary.csv", rows)
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "backend_name",
        "model_family",
        "num_qubits",
        "train_accuracy",
        "train_loss",
        "test_accuracy",
        "test_loss",
        "runtime_seconds",
        "output_dir",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["BackendBenchmarkConfig", "run_backend_benchmark"]
