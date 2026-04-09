"""Small noisy mixed-state comparison workflow for Qiskit-backed experiments."""

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
from eqnn.experiments.runner import ExperimentConfig, build_backend_with_options, run_training_experiment
from eqnn.noise import SUPPORTED_NOISE_MODELS, NoiseConfig, noise_config_from_strength
from eqnn.training import TrainingConfig
from eqnn.utils.timing import RuntimeProfile, timed


@dataclass(frozen=True)
class NoisyComparisonConfig:
    model_families: tuple[str, ...] = ("su2_qcnn", "hea_qcnn")
    num_qubits_values: tuple[int, ...] = (4, 6)
    train_sizes: tuple[int, ...] = (4, 8, 12)
    epochs_values: tuple[int, ...] = (10,)
    random_seeds: tuple[int, ...] = (0, 1)
    backend_name: str = "qiskit_mixed"
    noise_model_name: str = "depolarizing"
    noise_strength_values: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1)
    odd_qubits_only: bool = False
    learning_rate: float = 5e-2
    gradient_backend: str = "finite_difference"
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
        if not self.noise_strength_values:
            raise ValueError("noise_strength_values must not be empty")
        allowed_families = {"su2_qcnn", "hea_qcnn", "baseline_qcnn"}
        invalid = tuple(family for family in self.model_families if family not in allowed_families)
        if invalid:
            raise ValueError(f"Unsupported model_families: {invalid}")
        if self.backend_name != "qiskit_mixed":
            raise ValueError("Noisy mixed-state comparisons currently require backend_name='qiskit_mixed'")
        canonical_noise_model = NoiseConfig(noise_model_name=self.noise_model_name).noise_model_name
        object.__setattr__(self, "noise_model_name", canonical_noise_model)
        if self.loss != "mse":
            raise ValueError("Noisy comparisons are locked to loss='mse'")
        if self.batch_size != 2:
            raise ValueError("Noisy comparisons are locked to batch_size=2")
        if self.optimizer != "adam":
            raise ValueError("Noisy comparisons are locked to optimizer='adam'")
        if self.boundary != "open":
            raise ValueError("Noisy comparisons are currently locked to boundary='open'")
        if self.pooling_mode != "partial_trace":
            raise ValueError("Noisy comparisons are currently locked to pooling_mode='partial_trace'")
        if self.readout_mode != "swap":
            raise ValueError("Noisy comparisons are currently locked to readout_mode='swap'")
        resolved_num_qubits = self.resolved_num_qubits_values
        if not resolved_num_qubits:
            raise ValueError("No odd qubit counts remain after applying odd_qubits_only=True")
        for value in resolved_num_qubits:
            if int(value) < 2:
                raise ValueError("num_qubits_values must be integers at least 2")
        for train_size in self.train_sizes:
            if int(train_size) < 2 or int(train_size) % 2 != 0:
                raise ValueError("train_sizes must be even integers at least 2")
        for value in self.noise_strength_values:
            if canonical_noise_model == "none" and float(value) != 0.0:
                raise ValueError("noise strengths must be exactly 0.0 when noise_model_name='none'")
            if canonical_noise_model == "coherent_overrotation":
                if not np.isfinite(float(value)):
                    raise ValueError("noise strengths must be finite for coherent_overrotation")
            elif not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"noise strengths must lie in [0, 1] for noise_model_name='{canonical_noise_model}'"
                )

    @property
    def resolved_num_qubits_values(self) -> tuple[int, ...]:
        values = tuple(int(value) for value in self.num_qubits_values)
        if not self.odd_qubits_only:
            return values
        return tuple(value for value in values if value % 2 == 1)


@dataclass(frozen=True)
class NoisyComparisonJob:
    index: int
    model_family: str
    num_qubits: int
    train_size: int
    epochs: int
    noise_strength: float
    seed: int


def enumerate_noisy_comparison_jobs(config: NoisyComparisonConfig) -> list[NoisyComparisonJob]:
    jobs: list[NoisyComparisonJob] = []
    for index, (model_family, num_qubits, train_size, epochs, noise_strength, seed) in enumerate(
        product(
            config.model_families,
            config.resolved_num_qubits_values,
            config.train_sizes,
            config.epochs_values,
            config.noise_strength_values,
            config.random_seeds,
        )
    ):
        jobs.append(
            NoisyComparisonJob(
                index=index,
                model_family=str(model_family),
                num_qubits=int(num_qubits),
                train_size=int(train_size),
                epochs=int(epochs),
                noise_strength=float(noise_strength),
                seed=int(seed),
            )
        )
    return jobs


def noisy_comparison_job_from_index(config: NoisyComparisonConfig, index: int) -> NoisyComparisonJob:
    jobs = enumerate_noisy_comparison_jobs(config)
    if index < 0 or index >= len(jobs):
        raise IndexError(f"Noisy comparison job index {index} is out of range for {len(jobs)} jobs")
    return jobs[index]


def run_noisy_comparison(
    config: NoisyComparisonConfig,
    output_dir: str | Path,
    *,
    job_index: int | None = None,
    aggregate_only: bool = False,
    force_rerun: bool = False,
    profile: RuntimeProfile | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    with timed(profile, "noisy.output.prepare_root"):
        output_path.mkdir(parents=True, exist_ok=True)

    with timed(profile, "noisy.write_config"):
        (output_path / "noisy_comparison_config.json").write_text(
            json.dumps(
                _serialize_for_json(
                    {
                        **asdict(config),
                        "resolved_num_qubits_values": list(config.resolved_num_qubits_values),
                        "supported_noise_models": list(SUPPORTED_NOISE_MODELS),
                    }
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    dataset_cache: dict[tuple[int, int], DatasetBundle] = {}

    if not aggregate_only:
        jobs = (
            [noisy_comparison_job_from_index(config, job_index)]
            if job_index is not None
            else enumerate_noisy_comparison_jobs(config)
        )
        for job in jobs:
            _run_noisy_comparison_job(
                config,
                job,
                output_path,
                dataset_cache=dataset_cache,
                force_rerun=force_rerun,
                profile=profile,
            )
        if job_index is not None:
            run_row = _load_run_row(_job_output_dir(output_path, config, jobs[0]) / "noisy_run.json")
            return {"job": asdict(jobs[0]), "run": run_row}

    run_rows = load_completed_noisy_comparison_runs(output_path)
    summary_rows = aggregate_noisy_comparison_runs(run_rows)

    with timed(profile, "noisy.write_runs_json"):
        (output_path / "runs.json").write_text(json.dumps(_serialize_for_json(run_rows), indent=2, sort_keys=True) + "\n")
    with timed(profile, "noisy.write_summary_json"):
        (output_path / "summary.json").write_text(
            json.dumps(_serialize_for_json(summary_rows), indent=2, sort_keys=True) + "\n"
        )
    with timed(profile, "noisy.write_summary_csv"):
        _write_summary_csv(output_path / "summary.csv", summary_rows)

    return {"summary": summary_rows, "runs": run_rows}


def load_completed_noisy_comparison_runs(output_dir: str | Path) -> list[dict[str, Any]]:
    output_path = Path(output_dir)
    run_paths = sorted(output_path.rglob("noisy_run.json"))
    if not run_paths:
        raise ValueError(f"No noisy_run.json files were found under {output_path}")
    rows = [_load_run_row(path) for path in run_paths]
    rows.sort(
        key=lambda row: (
            int(row["job_index"]),
            str(row["backend_name"]),
            str(row["model_family"]),
            int(row["num_qubits"]),
            int(row["train_size"]),
            int(row["epochs"]),
            str(row["noise_model_name"]),
            float(row["noise_strength"]),
            int(row["seed"]),
        )
    )
    return rows


def aggregate_noisy_comparison_runs(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in run_rows:
        key = (
            row["backend_name"],
            row["model_family"],
            int(row["num_qubits"]),
            int(row["train_size"]),
            int(row["epochs"]),
            row["noise_model_name"],
            float(row["noise_strength"]),
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
            "noise_model_name": key[5],
            "noise_strength": key[6],
            "num_runs": len(rows),
        }
        for metric_name in (
            "train_accuracy",
            "test_accuracy",
            "train_loss",
            "test_loss",
            "classification_threshold",
            "build_time_seconds",
            "forward_time_seconds",
            "gradient_time_seconds",
            "total_training_time_seconds",
        ):
            metric_values = [
                float(row[metric_name])
                for row in rows
                if row.get(metric_name) is not None
            ]
            if metric_values:
                values = np.asarray(metric_values, dtype=np.float64)
                summary_row[f"mean_{metric_name}"] = float(np.mean(values))
                summary_row[f"variance_{metric_name}"] = float(np.var(values))
            else:
                summary_row[f"mean_{metric_name}"] = None
                summary_row[f"variance_{metric_name}"] = None

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
            str(row["noise_model_name"]),
            float(row["noise_strength"]),
        )
    )
    return summary_rows


def _run_noisy_comparison_job(
    config: NoisyComparisonConfig,
    job: NoisyComparisonJob,
    output_path: Path,
    *,
    dataset_cache: dict[tuple[int, int], DatasetBundle],
    force_rerun: bool,
    profile: RuntimeProfile | None = None,
) -> dict[str, Any]:
    run_output_dir = _job_output_dir(output_path, config, job)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    run_row_path = run_output_dir / "noisy_run.json"

    if run_row_path.exists() and not force_rerun:
        return _load_run_row(run_row_path)

    run_profile = RuntimeProfile()
    with timed(run_profile, "noisy.single_run"):
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

        resolved_noise_config = noise_config_from_strength(config.noise_model_name, job.noise_strength)
        backend = build_backend_with_options(config.backend_name, noise_config=resolved_noise_config)

        experiment_config = ExperimentConfig(
            model_family=job.model_family,
            backend_name=config.backend_name,
            num_qubits=job.num_qubits,
            boundary=config.boundary,
            shared_convolution_parameter=config.shared_convolution_parameter,
            pooling_mode=config.pooling_mode,
            pooling_keep=config.pooling_keep,
            readout_mode=config.readout_mode,
        )
        training_config = TrainingConfig(
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
        )

        (run_output_dir / "noisy_job_config.json").write_text(
            json.dumps(
                _serialize_for_json(
                    {
                        "job": asdict(job),
                        "config": asdict(config),
                        "resolved_num_qubits_values": list(config.resolved_num_qubits_values),
                        "noise_config": resolved_noise_config.to_dict(),
                        "noise_metadata": resolved_noise_config.to_metadata(),
                    }
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        result = run_training_experiment(
            dataset,
            experiment_config,
            training_config,
            output_dir=run_output_dir,
            experiment_name=_job_experiment_name(config, job),
            backend=backend,
            profile=run_profile,
        )

    runtime_summary = run_profile.summary()
    runtime_seconds = float(runtime_summary["noisy.single_run"]["total_seconds"])
    runtime_breakdown = dict(result.get("runtime_breakdown", {}))
    run_row = {
        "job_index": int(job.index),
        "experiment_name": str(result["experiment_name"]),
        "backend_name": str(config.backend_name),
        "model_family": str(job.model_family),
        "num_qubits": int(job.num_qubits),
        "train_size": int(job.train_size),
        "epochs": int(job.epochs),
        "noise_model_name": str(resolved_noise_config.noise_model_name),
        "noise_strength": float(job.noise_strength),
        "seed": int(job.seed),
        "train_accuracy": float(result["train_metrics"]["accuracy"]),
        "test_accuracy": float(result["test_metrics"]["accuracy"]),
        "train_loss": float(result["train_metrics"]["loss"]),
        "test_loss": float(result["test_metrics"]["loss"]),
        "classification_threshold": float(result["classification_threshold"]),
        "build_time_seconds": float(runtime_breakdown.get("build_time_seconds", 0.0)),
        "forward_time_seconds": float(runtime_breakdown.get("forward_time_seconds", 0.0)),
        "gradient_time_seconds": float(runtime_breakdown.get("gradient_time_seconds", 0.0)),
        "total_training_time_seconds": float(runtime_breakdown.get("total_training_time_seconds", 0.0)),
        "runtime_seconds": runtime_seconds,
        "output_dir": str(run_output_dir.resolve()),
    }
    run_metadata = {
        "job": asdict(job),
        "config": asdict(config),
        "resolved_num_qubits_values": list(config.resolved_num_qubits_values),
        "noise_config": resolved_noise_config.to_dict(),
        "noise_metadata": resolved_noise_config.to_metadata(),
        "dataset_metadata": result.get("dataset_metadata"),
        "experiment_config": result.get("experiment_config"),
        "training_config": result.get("training_config"),
        "runtime_profile": result.get("runtime_profile"),
        "runtime_breakdown": runtime_breakdown,
        "run_summary": run_row,
    }
    run_row_path.write_text(json.dumps(_serialize_for_json(run_row), indent=2, sort_keys=True) + "\n")
    (run_output_dir / "runtime_profile.json").write_text(
        json.dumps(_serialize_for_json(result.get("runtime_profile", {})), indent=2, sort_keys=True) + "\n"
    )
    (run_output_dir / "runtime_breakdown.json").write_text(
        json.dumps(_serialize_for_json(runtime_breakdown), indent=2, sort_keys=True) + "\n"
    )
    (run_output_dir / "noisy_run_metadata.json").write_text(
        json.dumps(_serialize_for_json(run_metadata), indent=2, sort_keys=True) + "\n"
    )
    return run_row


def _job_output_dir(output_path: Path, config: NoisyComparisonConfig, job: NoisyComparisonJob) -> Path:
    return (
        output_path
        / config.backend_name
        / job.model_family
        / f"n{job.num_qubits}"
        / f"train_size_{job.train_size}"
        / f"epochs_{job.epochs}"
        / f"noise_{config.noise_model_name}_{_noise_strength_slug(job.noise_strength)}"
        / f"seed_{job.seed}"
    )


def _job_experiment_name(config: NoisyComparisonConfig, job: NoisyComparisonJob) -> str:
    return (
        f"noisy_comparison_{job.model_family}_{config.backend_name}_n{job.num_qubits}_"
        f"train{job.train_size}_epochs{job.epochs}_"
        f"{config.noise_model_name}_{_noise_strength_slug(job.noise_strength)}_seed{job.seed}"
    )


def _noise_strength_slug(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if text == "":
        text = "0"
    return text.replace("-", "m").replace(".", "p")


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
        "noise_model_name",
        "noise_strength",
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
        "mean_build_time_seconds",
        "variance_build_time_seconds",
        "mean_forward_time_seconds",
        "variance_forward_time_seconds",
        "mean_gradient_time_seconds",
        "variance_gradient_time_seconds",
        "mean_total_training_time_seconds",
        "variance_total_training_time_seconds",
        "mean_runtime_seconds",
        "variance_runtime_seconds",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "NoisyComparisonConfig",
    "NoisyComparisonJob",
    "aggregate_noisy_comparison_runs",
    "enumerate_noisy_comparison_jobs",
    "load_completed_noisy_comparison_runs",
    "noise_config_from_strength",
    "noisy_comparison_job_from_index",
    "run_noisy_comparison",
]
