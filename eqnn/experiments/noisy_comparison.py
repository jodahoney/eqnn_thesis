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
from eqnn.experiments.runner import (
    ExperimentConfig,
    build_backend_with_options,
    build_model,
    run_training_experiment,
)
from eqnn.noise import SUPPORTED_NOISE_MODELS, NoiseConfig, noise_config_from_strength
from eqnn.training import TrainingConfig
from eqnn.utils.timing import RuntimeProfile, timed
from eqnn.verification import estimate_equivariance_error


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
    coherent_overrotation_mode: str = "fixed"
    coherent_overrotation_probability: float = 1.0
    coherent_overrotation_angle_std: float = 0.0
    coherent_overrotation_seed: int | None = None
    noise_application_scope: str = "active"
    noisy_qubit_indices: tuple[int | None, ...] = (None,)
    single_qubit_error_profile: tuple[float, ...] | None = None
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
    compute_symmetry_diagnostics: bool = False
    num_symmetry_samples: int = 8
    num_state_samples_for_diagnostic: int = 8

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
        if not self.noisy_qubit_indices:
            raise ValueError("noisy_qubit_indices must not be empty")
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
        if self.noise_application_scope not in {"active", "all", "selected_qubits"}:
            raise ValueError("noise_application_scope must be 'active', 'all', or 'selected_qubits'")
        if self.coherent_overrotation_mode not in {"fixed", "stochastic", "random_angle", "pair_dependent"}:
            raise ValueError(
                "coherent_overrotation_mode must be 'fixed', 'stochastic', 'random_angle', or 'pair_dependent'"
            )
        if not 0.0 <= float(self.coherent_overrotation_probability) <= 1.0:
            raise ValueError("coherent_overrotation_probability must lie in [0, 1]")
        if float(self.coherent_overrotation_angle_std) < 0.0:
            raise ValueError("coherent_overrotation_angle_std must be non-negative")
        if self.single_qubit_error_profile is not None:
            normalized_profile = tuple(float(value) for value in self.single_qubit_error_profile)
            if any(not 0.0 <= value <= 1.0 for value in normalized_profile):
                raise ValueError("single_qubit_error_profile values must lie in [0, 1]")
            object.__setattr__(self, "single_qubit_error_profile", normalized_profile)
        normalized_noisy_indices = tuple(
            None if value is None else int(value) for value in self.noisy_qubit_indices
        )
        if any(value is not None and value < 0 for value in normalized_noisy_indices):
            raise ValueError("noisy_qubit_indices must contain non-negative integers or None")
        object.__setattr__(self, "noisy_qubit_indices", normalized_noisy_indices)
        if self.noise_application_scope == "selected_qubits" and all(value is None for value in normalized_noisy_indices):
            raise ValueError(
                "noise_application_scope='selected_qubits' requires at least one explicit noisy_qubit_index"
            )
        if self.num_symmetry_samples < 1:
            raise ValueError("num_symmetry_samples must be at least 1")
        if self.num_state_samples_for_diagnostic < 1:
            raise ValueError("num_state_samples_for_diagnostic must be at least 1")
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
    noisy_qubit_index: int | None
    seed: int


def enumerate_noisy_comparison_jobs(config: NoisyComparisonConfig) -> list[NoisyComparisonJob]:
    jobs: list[NoisyComparisonJob] = []
    for index, (model_family, num_qubits, train_size, epochs, noise_strength, noisy_qubit_index, seed) in enumerate(
        product(
            config.model_families,
            config.resolved_num_qubits_values,
            config.train_sizes,
            config.epochs_values,
            config.noise_strength_values,
            config.noisy_qubit_indices,
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
                noisy_qubit_index=None if noisy_qubit_index is None else int(noisy_qubit_index),
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
            int(row.get("job_index", 0)),
            str(row.get("backend_name")),
            str(row.get("model_family")),
            int(row.get("num_qubits", 0)),
            int(row.get("train_size", 0)),
            int(row.get("epochs", 0)),
            str(row.get("noise_model_name")),
            float(row.get("noise_strength", 0.0)),
            -1 if row.get("noisy_qubit_index") is None else int(row["noisy_qubit_index"]),
            int(row.get("seed", 0)),
        )
    )
    return rows


def aggregate_noisy_comparison_runs(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in run_rows:
        key = (
            row.get("backend_name"),
            row.get("model_family"),
            int(row.get("num_qubits", 0)),
            int(row.get("train_size", 0)),
            int(row.get("epochs", 0)),
            row.get("noise_model_name"),
            float(row.get("noise_strength", 0.0)),
            row.get("noise_primary_strength"),
            row.get("noise_application_scope"),
            row.get("noisy_qubit_index"),
            _hashable_key_value(row.get("noisy_qubits")),
            _hashable_key_value(row.get("single_qubit_error_profile")),
            row.get("single_qubit_depolarizing_error"),
            row.get("two_qubit_depolarizing_error"),
            row.get("amplitude_damping_gamma"),
            row.get("phase_damping_gamma"),
            row.get("coherent_overrotation_angle"),
            row.get("coherent_overrotation_axis"),
            row.get("coherent_overrotation_mode"),
            row.get("coherent_overrotation_probability"),
            row.get("coherent_overrotation_angle_std"),
            row.get("coherent_overrotation_seed"),
            _hashable_key_value(row.get("pair_dependent_overrotation_angles")),
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
            "noise_primary_strength": key[7],
            "noise_application_scope": key[8],
            "noisy_qubit_index": key[9],
            "noisy_qubits": _serialize_for_json(key[10]),
            "single_qubit_error_profile": _serialize_for_json(key[11]),
            "single_qubit_depolarizing_error": key[12],
            "two_qubit_depolarizing_error": key[13],
            "amplitude_damping_gamma": key[14],
            "phase_damping_gamma": key[15],
            "coherent_overrotation_angle": key[16],
            "coherent_overrotation_axis": key[17],
            "coherent_overrotation_mode": key[18],
            "coherent_overrotation_probability": key[19],
            "coherent_overrotation_angle_std": key[20],
            "coherent_overrotation_seed": key[21],
            "pair_dependent_overrotation_angles": _serialize_for_json(key[22]),
            "num_runs": len(rows),
        }
        for metric_name in (
            "train_accuracy",
            "test_accuracy",
            "train_loss",
            "test_loss",
            "classification_threshold",
            "test_equivariance_error_mean",
            "test_equivariance_error_max",
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
            -1.0 if row.get("noise_primary_strength") is None else float(row["noise_primary_strength"]),
            str(row.get("noise_application_scope")),
            -1 if row.get("noisy_qubit_index") is None else int(row["noisy_qubit_index"]),
            str(row.get("noisy_qubits")),
            str(row.get("single_qubit_error_profile")),
            -1.0
            if row.get("single_qubit_depolarizing_error") is None
            else float(row["single_qubit_depolarizing_error"]),
            -1.0
            if row.get("two_qubit_depolarizing_error") is None
            else float(row["two_qubit_depolarizing_error"]),
            -1.0 if row.get("amplitude_damping_gamma") is None else float(row["amplitude_damping_gamma"]),
            -1.0 if row.get("phase_damping_gamma") is None else float(row["phase_damping_gamma"]),
            -1.0
            if row.get("coherent_overrotation_angle") is None
            else float(row["coherent_overrotation_angle"]),
            str(row.get("coherent_overrotation_axis")),
            str(row.get("coherent_overrotation_mode")),
            -1.0
            if row.get("coherent_overrotation_probability") is None
            else float(row["coherent_overrotation_probability"]),
            -1.0
            if row.get("coherent_overrotation_angle_std") is None
            else float(row["coherent_overrotation_angle_std"]),
            -1 if row.get("coherent_overrotation_seed") is None else int(row["coherent_overrotation_seed"]),
            str(row.get("pair_dependent_overrotation_angles")),
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

        resolved_noise_config = noise_config_from_strength(
            config.noise_model_name,
            job.noise_strength,
            coherent_overrotation_mode=config.coherent_overrotation_mode,
            coherent_overrotation_probability=config.coherent_overrotation_probability,
            coherent_overrotation_angle_std=config.coherent_overrotation_angle_std,
            coherent_overrotation_seed=config.coherent_overrotation_seed,
            noise_application_scope=(
                "selected_qubits" if job.noisy_qubit_index is not None else config.noise_application_scope
            ),
            noisy_qubits=None if job.noisy_qubit_index is None else (int(job.noisy_qubit_index),),
            single_qubit_error_profile=config.single_qubit_error_profile,
        )
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
                        "noisy_qubit_index": job.noisy_qubit_index,
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

        symmetry_diagnostic = _symmetry_diagnostic_for_run(
            config,
            dataset.test.states,
            model_family=job.model_family,
            num_qubits=job.num_qubits,
            backend=backend,
            result=result,
        )

    runtime_summary = run_profile.summary()
    runtime_seconds = float(runtime_summary["noisy.single_run"]["total_seconds"])
    runtime_breakdown = dict(result.get("runtime_breakdown", {}))
    noise_parameters = resolved_noise_config.parameter_metadata()
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
        "noise_primary_strength": float(resolved_noise_config.primary_strength),
        "noise_application_scope": str(resolved_noise_config.noise_application_scope),
        "noisy_qubit_index": job.noisy_qubit_index,
        "noisy_qubits": None if resolved_noise_config.noisy_qubits is None else list(resolved_noise_config.noisy_qubits),
        "single_qubit_error_profile": (
            None
            if resolved_noise_config.single_qubit_error_profile is None
            else list(resolved_noise_config.single_qubit_error_profile)
        ),
        "single_qubit_depolarizing_error": float(noise_parameters["single_qubit_depolarizing_error"]),
        "two_qubit_depolarizing_error": float(noise_parameters["two_qubit_depolarizing_error"]),
        "amplitude_damping_gamma": float(noise_parameters["amplitude_damping_gamma"]),
        "phase_damping_gamma": float(noise_parameters["phase_damping_gamma"]),
        "coherent_overrotation_angle": float(noise_parameters["coherent_overrotation_angle"]),
        "coherent_overrotation_axis": str(noise_parameters["coherent_overrotation_axis"]),
        "coherent_overrotation_mode": str(noise_parameters["coherent_overrotation_mode"]),
        "coherent_overrotation_probability": float(noise_parameters["coherent_overrotation_probability"]),
        "coherent_overrotation_angle_std": float(noise_parameters["coherent_overrotation_angle_std"]),
        "coherent_overrotation_seed": noise_parameters["coherent_overrotation_seed"],
        "pair_dependent_overrotation_angles": noise_parameters["pair_dependent_overrotation_angles"],
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
        "symmetry_diagnostic_available": bool(symmetry_diagnostic["available"]),
        "symmetry_diagnostic_note": str(symmetry_diagnostic["note"]),
        "test_equivariance_error_mean": symmetry_diagnostic["mean_error"],
        "test_equivariance_error_max": symmetry_diagnostic["max_error"],
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
        "symmetry_diagnostic": symmetry_diagnostic,
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
    path = (
        output_path
        / config.backend_name
        / job.model_family
        / f"n{job.num_qubits}"
        / f"train_size_{job.train_size}"
        / f"epochs_{job.epochs}"
        / f"noise_{config.noise_model_name}_{_noise_strength_slug(job.noise_strength)}"
    )
    if job.noisy_qubit_index is not None or config.noise_application_scope != "active":
        path = path / f"scope_{'selected_qubits' if job.noisy_qubit_index is not None else config.noise_application_scope}"
        path = path / f"noisy_qubit_{'none' if job.noisy_qubit_index is None else job.noisy_qubit_index}"
    return path / f"seed_{job.seed}"


def _job_experiment_name(config: NoisyComparisonConfig, job: NoisyComparisonJob) -> str:
    name = (
        f"noisy_comparison_{job.model_family}_{config.backend_name}_n{job.num_qubits}_"
        f"train{job.train_size}_epochs{job.epochs}_"
        f"{config.noise_model_name}_{_noise_strength_slug(job.noise_strength)}"
    )
    if job.noisy_qubit_index is not None or config.noise_application_scope != "active":
        name += (
            f"_scope_{'selected_qubits' if job.noisy_qubit_index is not None else config.noise_application_scope}"
            f"_qubit_{'none' if job.noisy_qubit_index is None else job.noisy_qubit_index}"
        )
    return f"{name}_seed{job.seed}"


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


def _hashable_key_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return tuple(_hashable_key_value(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return tuple(sorted((str(key), _hashable_key_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_key_value(item) for item in value)
    return value


def _consistent_metadata_value(rows: list[dict[str, Any]], key: str) -> Any:
    values = [row.get(key) for row in rows if key in row]
    if not values:
        return None
    first = _serialize_for_json(values[0])
    if all(_serialize_for_json(value) == first for value in values[1:]):
        return first
    return None


def _symmetry_diagnostic_for_run(
    config: NoisyComparisonConfig,
    test_states: np.ndarray,
    *,
    model_family: str,
    num_qubits: int,
    backend: object,
    result: dict[str, Any],
) -> dict[str, Any]:
    if not config.compute_symmetry_diagnostics:
        return {
            "available": False,
            "note": "disabled",
            "mean_error": None,
            "max_error": None,
        }

    try:
        best_parameters = np.asarray(result["history"]["best_parameters"], dtype=np.float64)
        diagnostic_model = build_model(
            ExperimentConfig(
                model_family=model_family,
                backend_name=config.backend_name,
                num_qubits=num_qubits,
                boundary=config.boundary,
                shared_convolution_parameter=config.shared_convolution_parameter,
                pooling_mode=config.pooling_mode,
                pooling_keep=config.pooling_keep,
                readout_mode=config.readout_mode,
            ),
            parameters=best_parameters,
            backend=backend,
        )
        if hasattr(diagnostic_model, "set_classification_threshold"):
            diagnostic_model.set_classification_threshold(float(result["classification_threshold"]))

        subset_size = min(int(config.num_state_samples_for_diagnostic), int(test_states.shape[0]))
        diagnostic_states = np.asarray(test_states[:subset_size], dtype=np.complex128)
        diagnostic = estimate_equivariance_error(
            diagnostic_model,
            diagnostic_states,
            num_symmetry_samples=config.num_symmetry_samples,
            seed=0,
            backend=backend,
        )
        return {
            "available": bool(diagnostic.get("available", False)),
            "note": str(diagnostic.get("note", "")),
            "mean_error": diagnostic.get("mean_error"),
            "max_error": diagnostic.get("max_error"),
        }
    except Exception as exc:  # pragma: no cover - defensive path for optional diagnostics
        return {
            "available": False,
            "note": f"not_available: {type(exc).__name__}: {exc}",
            "mean_error": None,
            "max_error": None,
        }


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "backend_name",
        "model_family",
        "num_qubits",
        "train_size",
        "epochs",
        "noise_model_name",
        "noise_strength",
        "noise_primary_strength",
        "noise_application_scope",
        "noisy_qubit_index",
        "noisy_qubits",
        "single_qubit_error_profile",
        "single_qubit_depolarizing_error",
        "two_qubit_depolarizing_error",
        "amplitude_damping_gamma",
        "phase_damping_gamma",
        "coherent_overrotation_angle",
        "coherent_overrotation_axis",
        "coherent_overrotation_mode",
        "coherent_overrotation_probability",
        "coherent_overrotation_angle_std",
        "coherent_overrotation_seed",
        "pair_dependent_overrotation_angles",
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
        "mean_test_equivariance_error_mean",
        "variance_test_equivariance_error_mean",
        "mean_test_equivariance_error_max",
        "variance_test_equivariance_error_max",
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
