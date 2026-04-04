"""Calibration sweep utilities for selecting practical training budgets."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from eqnn.datasets.heisenberg import DatasetBundle
from eqnn.experiments.reproduction import PaperDatasetConfig, generate_paper_dataset
from eqnn.experiments.runner import ExperimentConfig, run_training_experiment
from eqnn.training import TrainingConfig
from eqnn.utils.timing import RuntimeProfile, timed


@dataclass(frozen=True)
class CalibrationSweepConfig:
    model_families: tuple[str, ...] = ("su2_qcnn", "hea_qcnn")
    backend_name: str = "numpy_pure"
    num_qubits_values: tuple[int, ...] = (6, 10)
    train_sizes: tuple[int, ...] = (12,)
    epochs_values: tuple[int, ...] = (50, 150, 300, 500, 750)
    random_seeds: tuple[int, ...] = (0, 1, 2)
    learning_rate: float = 5e-2
    gradient_backend: str = "exact"
    initialization_strategy: str = "noisy_current"
    initialization_noise_scale: float = 5e-2
    critical_ratio: float = 1.0
    left_ratio_min: float = 0.0
    right_ratio_max: float = 2.0
    dense_test_points: int = 101
    eigensolver: str = "auto"
    loss: str = "mse"
    batch_size: int = 2
    optimizer: str = "adam"
    threshold_update: str = "paper_nearest_critical"
    threshold_critical_ratio: float = 1.0
    boundary: str = "open"
    pooling_mode: str = "partial_trace"
    pooling_keep: str = "left"
    readout_mode: str = "swap"
    shared_convolution_parameter: bool = True

    def __post_init__(self) -> None:
        if not self.model_families:
            raise ValueError("model_families must not be empty")
        if not self.num_qubits_values:
            raise ValueError("num_qubits_values must not be empty")
        if not self.train_sizes:
            raise ValueError("train_sizes must not be empty")
        if not self.epochs_values:
            raise ValueError("epochs_values must not be empty")
        if not self.random_seeds:
            raise ValueError("random_seeds must not be empty")
        allowed_families = {"su2_qcnn", "hea_qcnn", "baseline_qcnn"}
        invalid = tuple(family for family in self.model_families if family not in allowed_families)
        if invalid:
            raise ValueError(f"Unsupported model_families: {invalid}")
        if self.backend_name not in {"numpy_pure", "torch_pure"}:
            raise ValueError("backend_name must be 'numpy_pure' or 'torch_pure'")
        if self.loss != "mse":
            raise ValueError("Calibration sweeps are locked to loss='mse'")
        if self.batch_size != 2:
            raise ValueError("Calibration sweeps are locked to batch_size=2")
        if self.optimizer != "adam":
            raise ValueError("Calibration sweeps are locked to optimizer='adam'")
        if self.threshold_update != "paper_nearest_critical":
            raise ValueError("Calibration sweeps are locked to threshold_update='paper_nearest_critical'")
        if self.boundary != "open":
            raise ValueError("Calibration sweeps are currently locked to boundary='open'")
        if self.pooling_mode != "partial_trace":
            raise ValueError("Calibration sweeps are currently locked to pooling_mode='partial_trace'")
        if self.readout_mode != "swap":
            raise ValueError("Calibration sweeps are currently locked to readout_mode='swap'")


@dataclass(frozen=True)
class CalibrationJob:
    index: int
    model_family: str
    num_qubits: int
    train_size: int
    epochs: int
    seed: int


def enumerate_calibration_jobs(config: CalibrationSweepConfig) -> list[CalibrationJob]:
    jobs: list[CalibrationJob] = []
    for index, (model_family, num_qubits, train_size, epochs, seed) in enumerate(
        product(
            config.model_families,
            config.num_qubits_values,
            config.train_sizes,
            config.epochs_values,
            config.random_seeds,
        )
    ):
        jobs.append(
            CalibrationJob(
                index=index,
                model_family=str(model_family),
                num_qubits=int(num_qubits),
                train_size=int(train_size),
                epochs=int(epochs),
                seed=int(seed),
            )
        )
    return jobs


def calibration_job_from_index(config: CalibrationSweepConfig, index: int) -> CalibrationJob:
    jobs = enumerate_calibration_jobs(config)
    if index < 0 or index >= len(jobs):
        raise IndexError(f"Calibration job index {index} is out of range for {len(jobs)} jobs")
    return jobs[index]


def run_calibration_sweep(
    config: CalibrationSweepConfig,
    output_dir: str | Path,
    *,
    job_index: int | None = None,
    aggregate_only: bool = False,
    force_rerun: bool = False,
    profile: RuntimeProfile | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    with timed(profile, "calibration.output.prepare_root"):
        output_path.mkdir(parents=True, exist_ok=True)

    config_path = output_path / "calibration_sweep_config.json"
    with timed(profile, "calibration.write_config"):
        config_path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")

    dataset_cache: dict[tuple[int, int], DatasetBundle] = {}

    if not aggregate_only:
        jobs = [calibration_job_from_index(config, job_index)] if job_index is not None else enumerate_calibration_jobs(config)
        for job in jobs:
            _run_calibration_job(
                config,
                job,
                output_path,
                dataset_cache=dataset_cache,
                force_rerun=force_rerun,
                profile=profile,
            )
        if job_index is not None:
            run_row = _load_run_row(_job_output_dir(output_path, config, jobs[0]) / "calibration_run.json")
            return {"job": asdict(jobs[0]), "run": run_row}

    run_rows = load_completed_calibration_runs(output_path)
    summary_rows = aggregate_calibration_runs(run_rows)

    with timed(profile, "calibration.write_runs_json"):
        (output_path / "runs.json").write_text(json.dumps(_serialize_for_json(run_rows), indent=2, sort_keys=True) + "\n")
    with timed(profile, "calibration.write_summary_json"):
        (output_path / "summary.json").write_text(
            json.dumps(_serialize_for_json(summary_rows), indent=2, sort_keys=True) + "\n"
        )
    with timed(profile, "calibration.write_summary_csv"):
        _write_summary_csv(output_path / "summary.csv", summary_rows)

    return {"summary": summary_rows, "runs": run_rows}


def load_completed_calibration_runs(output_dir: str | Path) -> list[dict[str, Any]]:
    output_path = Path(output_dir)
    run_paths = sorted(output_path.rglob("calibration_run.json"))
    if not run_paths:
        raise ValueError(f"No calibration_run.json files were found under {output_path}")
    rows = [_load_run_row(path) for path in run_paths]
    rows.sort(
        key=lambda row: (
            int(row["job_index"]),
            str(row["backend_name"]),
            str(row["model_family"]),
            int(row["num_qubits"]),
            int(row["train_size"]),
            int(row["epochs"]),
            int(row["seed"]),
        )
    )
    return rows


def aggregate_calibration_runs(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in run_rows:
        key = (
            row["backend_name"],
            row["model_family"],
            int(row["num_qubits"]),
            int(row["train_size"]),
            int(row["epochs"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        summary_row = {
            "backend_name": key[0],
            "model_family": key[1],
            "num_qubits": key[2],
            "train_size": key[3],
            "epochs": key[4],
            "num_runs": len(rows),
        }
        for metric_name in (
            "train_accuracy",
            "test_accuracy",
            "train_loss",
            "test_loss",
            "classification_threshold",
        ):
            values = np.asarray([float(row[metric_name]) for row in rows], dtype=np.float64)
            summary_row[f"mean_{metric_name}"] = float(np.mean(values))
            summary_row[f"variance_{metric_name}"] = float(np.var(values))

        runtime_values = [
            float(row["runtime_seconds"])
            for row in rows
            if row.get("runtime_seconds") is not None
        ]
        if runtime_values:
            runtime_array = np.asarray(runtime_values, dtype=np.float64)
            summary_row["mean_runtime_seconds"] = float(np.mean(runtime_array))
            summary_row["variance_runtime_seconds"] = float(np.var(runtime_array))
        else:
            summary_row["mean_runtime_seconds"] = None
            summary_row["variance_runtime_seconds"] = None
        summary_rows.append(summary_row)

    summary_rows.sort(
        key=lambda row: (
            str(row["backend_name"]),
            str(row["model_family"]),
            int(row["num_qubits"]),
            int(row["train_size"]),
            int(row["epochs"]),
        )
    )
    return summary_rows


def _run_calibration_job(
    config: CalibrationSweepConfig,
    job: CalibrationJob,
    output_path: Path,
    *,
    dataset_cache: dict[tuple[int, int], DatasetBundle],
    force_rerun: bool,
    profile: RuntimeProfile | None = None,
) -> dict[str, Any]:
    run_output_dir = _job_output_dir(output_path, config, job)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    run_row_path = run_output_dir / "calibration_run.json"

    if run_row_path.exists() and not force_rerun:
        return _load_run_row(run_row_path)

    run_profile = RuntimeProfile()
    with timed(run_profile, "calibration.single_run"):
        dataset_key = (job.num_qubits, job.train_size)
        if dataset_key not in dataset_cache:
            dataset_cache[dataset_key] = generate_paper_dataset(
                PaperDatasetConfig(
                    num_qubits=job.num_qubits,
                    train_size=job.train_size,
                    critical_ratio=config.critical_ratio,
                    left_ratio_min=config.left_ratio_min,
                    right_ratio_max=config.right_ratio_max,
                    dense_test_points=config.dense_test_points,
                    boundary=config.boundary,
                    eigensolver=config.eigensolver,
                ),
                profile=profile,
            )
        dataset = dataset_cache[dataset_key]

        result = run_training_experiment(
            dataset,
            ExperimentConfig(
                model_family=job.model_family,
                backend_name=config.backend_name,
                num_qubits=job.num_qubits,
                boundary=config.boundary,
                shared_convolution_parameter=config.shared_convolution_parameter,
                pooling_mode=config.pooling_mode,
                pooling_keep=config.pooling_keep,
                readout_mode=config.readout_mode,
            ),
            TrainingConfig(
                epochs=job.epochs,
                learning_rate=config.learning_rate,
                loss=config.loss,
                batch_size=config.batch_size,
                gradient_backend=config.gradient_backend,
                optimizer=config.optimizer,
                initialization_strategy=config.initialization_strategy,
                initialization_noise_scale=config.initialization_noise_scale,
                num_restarts=1,
                random_seed=job.seed,
                classification_threshold=0.5,
                threshold_update=config.threshold_update,
                threshold_critical_ratio=config.threshold_critical_ratio,
            ),
            output_dir=run_output_dir,
            experiment_name=_job_experiment_name(config, job),
            profile=run_profile,
        )

    runtime_summary = run_profile.summary()
    runtime_seconds = float(runtime_summary["calibration.single_run"]["total_seconds"])
    run_row = {
        "job_index": int(job.index),
        "experiment_name": str(result["experiment_name"]),
        "backend_name": str(config.backend_name),
        "model_family": str(job.model_family),
        "num_qubits": int(job.num_qubits),
        "train_size": int(job.train_size),
        "epochs": int(job.epochs),
        "seed": int(job.seed),
        "train_accuracy": float(result["train_metrics"]["accuracy"]),
        "test_accuracy": float(result["test_metrics"]["accuracy"]),
        "train_loss": float(result["train_metrics"]["loss"]),
        "test_loss": float(result["test_metrics"]["loss"]),
        "classification_threshold": float(result["classification_threshold"]),
        "runtime_seconds": runtime_seconds,
        "output_dir": str(run_output_dir.resolve()),
    }
    run_row_path.write_text(json.dumps(_serialize_for_json(run_row), indent=2, sort_keys=True) + "\n")
    return run_row


def _job_output_dir(output_path: Path, config: CalibrationSweepConfig, job: CalibrationJob) -> Path:
    return (
        output_path
        / config.backend_name
        / job.model_family
        / f"n{job.num_qubits}"
        / f"train_size_{job.train_size}"
        / f"epochs_{job.epochs}"
        / f"seed_{job.seed}"
    )


def _job_experiment_name(config: CalibrationSweepConfig, job: CalibrationJob) -> str:
    return (
        f"calibration_{job.model_family}_{config.backend_name}_n{job.num_qubits}_"
        f"train{job.train_size}_epochs{job.epochs}_seed{job.seed}"
    )


def _load_run_row(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _serialize_for_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _serialize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_json(item) for item in value]
    return value


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "backend_name",
        "model_family",
        "num_qubits",
        "train_size",
        "epochs",
        "num_runs",
        "mean_train_accuracy",
        "variance_train_accuracy",
        "mean_test_accuracy",
        "variance_test_accuracy",
        "mean_train_loss",
        "variance_train_loss",
        "mean_test_loss",
        "variance_test_loss",
        "mean_classification_threshold",
        "variance_classification_threshold",
        "mean_runtime_seconds",
        "variance_runtime_seconds",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
