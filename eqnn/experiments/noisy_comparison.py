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
from eqnn.verification import estimate_equivariance_error, evaluate_with_symmetry_twirling


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
    compute_symmetry_twirled_evaluation: bool = False
    num_symmetry_twirl_samples: int = 8
    symmetry_twirl_seed: int | None = None
    num_state_samples_for_twirled_evaluation: int | None = None
    noise_aware_training: bool = False
    training_noise_strengths: tuple[float, ...] = ()
    training_noise_sampling: str = "per_epoch"
    training_noise_seed: int | None = None
    training_noise_defaulted_from_evaluation_grid: bool = False
    symmetry_regularization: bool = False
    symmetry_regularization_weight: float = 0.0
    symmetry_regularization_beta_values: tuple[float, ...] = ()
    num_symmetry_regularization_samples: int = 2
    symmetry_regularization_frequency: int = 1
    symmetry_regularization_state_samples: int | None = None
    symmetry_regularization_seed: int | None = None

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
        if self.num_symmetry_twirl_samples < 1:
            raise ValueError("num_symmetry_twirl_samples must be at least 1")
        if (
            self.num_state_samples_for_twirled_evaluation is not None
            and self.num_state_samples_for_twirled_evaluation < 1
        ):
            raise ValueError("num_state_samples_for_twirled_evaluation must be at least 1 when provided")
        if self.training_noise_sampling not in {"per_epoch", "per_epoch_random_choice"}:
            raise ValueError("training_noise_sampling must be 'per_epoch' or 'per_epoch_random_choice'")
        normalized_noise_strengths = tuple(float(value) for value in self.noise_strength_values)
        if any(not np.isfinite(value) for value in normalized_noise_strengths):
            raise ValueError("noise_strength_values must be finite")
        object.__setattr__(self, "noise_strength_values", normalized_noise_strengths)
        normalized_training_noise_strengths = tuple(float(value) for value in self.training_noise_strengths)
        training_noise_defaulted = False
        if self.noise_aware_training and not normalized_training_noise_strengths:
            normalized_training_noise_strengths = normalized_noise_strengths
            training_noise_defaulted = True
        if any(not np.isfinite(value) or value < 0.0 for value in normalized_training_noise_strengths):
            raise ValueError("training_noise_strengths must contain nonnegative finite floats")
        object.__setattr__(self, "training_noise_strengths", normalized_training_noise_strengths)
        object.__setattr__(
            self,
            "training_noise_defaulted_from_evaluation_grid",
            bool(training_noise_defaulted or self.training_noise_defaulted_from_evaluation_grid),
        )
        normalized_symmetry_regularization_weight = float(self.symmetry_regularization_weight)
        if (
            not np.isfinite(normalized_symmetry_regularization_weight)
            or normalized_symmetry_regularization_weight < 0.0
        ):
            raise ValueError("symmetry_regularization_weight must be a nonnegative finite float")
        object.__setattr__(self, "symmetry_regularization_weight", normalized_symmetry_regularization_weight)
        normalized_beta_values = tuple(float(value) for value in self.symmetry_regularization_beta_values)
        if any(not np.isfinite(value) or value < 0.0 for value in normalized_beta_values):
            raise ValueError("symmetry_regularization_beta_values must contain nonnegative finite floats")
        if normalized_beta_values:
            object.__setattr__(self, "symmetry_regularization", True)
        object.__setattr__(self, "symmetry_regularization_beta_values", normalized_beta_values)
        if self.num_symmetry_regularization_samples < 1:
            raise ValueError("num_symmetry_regularization_samples must be at least 1")
        if self.symmetry_regularization_frequency < 1:
            raise ValueError("symmetry_regularization_frequency must be at least 1")
        if self.symmetry_regularization_state_samples is not None and self.symmetry_regularization_state_samples < 1:
            raise ValueError("symmetry_regularization_state_samples must be at least 1 when provided")
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
        for value in self.training_noise_strengths:
            if canonical_noise_model == "none" and float(value) != 0.0:
                raise ValueError("training_noise_strengths must be exactly 0.0 when noise_model_name='none'")
            if canonical_noise_model == "coherent_overrotation":
                if not np.isfinite(float(value)):
                    raise ValueError("training_noise_strengths must be finite for coherent_overrotation")
            elif not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"training_noise_strengths must lie in [0, 1] for noise_model_name='{canonical_noise_model}'"
                )

    @property
    def resolved_num_qubits_values(self) -> tuple[int, ...]:
        values = tuple(int(value) for value in self.num_qubits_values)
        if not self.odd_qubits_only:
            return values
        return tuple(value for value in values if value % 2 == 1)

    @property
    def resolved_symmetry_regularization_beta_values(self) -> tuple[float, ...]:
        if self.symmetry_regularization_beta_values:
            return tuple(float(value) for value in self.symmetry_regularization_beta_values)
        return (float(self.symmetry_regularization_weight),)


@dataclass(frozen=True)
class NoisyComparisonJob:
    index: int
    model_family: str
    num_qubits: int
    train_size: int
    epochs: int
    noise_strength: float
    noisy_qubit_index: int | None
    symmetry_regularization_beta: float
    seed: int


def enumerate_noisy_comparison_jobs(config: NoisyComparisonConfig) -> list[NoisyComparisonJob]:
    jobs: list[NoisyComparisonJob] = []
    for index, (
        model_family,
        num_qubits,
        train_size,
        epochs,
        noise_strength,
        noisy_qubit_index,
        symmetry_regularization_beta,
        seed,
    ) in enumerate(
        product(
            config.model_families,
            config.resolved_num_qubits_values,
            config.train_sizes,
            config.epochs_values,
            config.noise_strength_values,
            config.noisy_qubit_indices,
            config.resolved_symmetry_regularization_beta_values,
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
                symmetry_regularization_beta=float(symmetry_regularization_beta),
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
                        **_config_training_noise_aliases(config),
                        **_config_symmetry_regularization_aliases(config),
                        "resolved_num_qubits_values": list(config.resolved_num_qubits_values),
                        "resolved_symmetry_regularization_beta_values": list(
                            config.resolved_symmetry_regularization_beta_values
                        ),
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
            -1.0
            if _row_symmetry_regularization_beta(row) is None
            else float(_row_symmetry_regularization_beta(row)),
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
            _row_bool(row.get("noise_aware_training", False)),
            _hashable_key_value(_row_training_noise_strength_values(row)),
            _row_training_noise_sampling(row),
            _row_train_noise_sampling_mode(row),
            _row_train_noise_includes_zero(row),
            row.get("training_noise_seed"),
            _row_bool(row.get("symmetry_regularization", False)),
            _row_symmetry_regularization_enabled(row),
            _row_symmetry_regularization_beta(row),
            _row_symmetry_regularization_weight(row),
            row.get("num_symmetry_regularization_samples"),
            row.get("symmetry_regularization_frequency"),
            row.get("symmetry_regularization_state_samples"),
            row.get("symmetry_regularization_seed"),
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
            "eval_noise_strength": key[6],
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
            "noise_aware_training": key[23],
            "training_noise_strengths": _serialize_for_json(key[24]),
            "train_noise_strength_values": _serialize_for_json(key[24]),
            "training_noise_sampling": key[25],
            "train_noise_sampling_mode": key[26],
            "train_noise_includes_zero": key[27],
            "training_noise_seed": key[28],
            "symmetry_regularization": key[29],
            "symmetry_regularization_enabled": key[30],
            "symmetry_regularization_beta": key[31],
            "symmetry_regularization_weight": key[32],
            "num_symmetry_regularization_samples": key[33],
            "symmetry_regularization_frequency": key[34],
            "symmetry_regularization_state_samples": key[35],
            "symmetry_regularization_seed": key[36],
            "num_runs": len(rows),
            "symmetry_twirled_available": _consistent_metadata_value(rows, "symmetry_twirled_available"),
            "symmetry_twirled_note": _consistent_metadata_value(rows, "symmetry_twirled_note"),
            "num_symmetry_twirl_samples": _consistent_metadata_value(rows, "num_symmetry_twirl_samples"),
            "num_state_samples_for_twirled_evaluation": _consistent_metadata_value(
                rows,
                "num_state_samples_for_twirled_evaluation",
            ),
            "training_noise_effective_seed": _consistent_metadata_value(rows, "training_noise_effective_seed"),
            "training_noise_strength_counts": _consistent_metadata_value(rows, "training_noise_strength_counts"),
            "training_noise_defaulted_from_evaluation_grid": _consistent_metadata_value(
                rows,
                "training_noise_defaulted_from_evaluation_grid",
            ),
            "training_noise_note": _consistent_metadata_value(rows, "training_noise_note"),
            "symmetry_regularization_note": _consistent_metadata_value(rows, "symmetry_regularization_note"),
        }
        for flat_metric_name in (
            "mean_symmetry_penalty_history",
            "final_symmetry_penalty",
            "final_equivariance_error_mean",
            "final_equivariance_error_max",
        ):
            summary_row[flat_metric_name] = _mean_optional_metric(rows, flat_metric_name)
        for metric_name in (
            "train_accuracy",
            "test_accuracy",
            "train_loss",
            "test_loss",
            "classification_threshold",
            "test_equivariance_error_mean",
            "test_equivariance_error_max",
            "symmetry_twirled_test_accuracy",
            "symmetry_twirled_train_accuracy",
            "symmetry_twirled_raw_subset_accuracy",
            "symmetry_twirled_subset_size",
            "symmetry_twirled_num_correct_raw_subset",
            "symmetry_twirled_num_correct_twirled_subset",
            "symmetry_twirled_mean_abs_shift",
            "mean_symmetry_penalty_history",
            "final_symmetry_penalty",
            "final_equivariance_error_mean",
            "final_equivariance_error_max",
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
            str(row.get("noise_aware_training")),
            str(row.get("training_noise_strengths")),
            str(row.get("training_noise_sampling")),
            str(row.get("train_noise_sampling_mode")),
            str(row.get("train_noise_includes_zero")),
            -1 if row.get("training_noise_seed") is None else int(row["training_noise_seed"]),
            str(row.get("symmetry_regularization")),
            str(row.get("symmetry_regularization_enabled")),
            -1.0
            if row.get("symmetry_regularization_beta") is None
            else float(row["symmetry_regularization_beta"]),
            -1.0
            if row.get("symmetry_regularization_weight") is None
            else float(row["symmetry_regularization_weight"]),
            -1
            if row.get("num_symmetry_regularization_samples") is None
            else int(row["num_symmetry_regularization_samples"]),
            -1
            if row.get("symmetry_regularization_frequency") is None
            else int(row["symmetry_regularization_frequency"]),
            -1
            if row.get("symmetry_regularization_state_samples") is None
            else int(row["symmetry_regularization_state_samples"]),
            -1
            if row.get("symmetry_regularization_seed") is None
            else int(row["symmetry_regularization_seed"]),
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
            symmetry_regularization=config.symmetry_regularization,
            symmetry_regularization_weight=job.symmetry_regularization_beta,
            num_symmetry_regularization_samples=config.num_symmetry_regularization_samples,
            symmetry_regularization_frequency=config.symmetry_regularization_frequency,
            symmetry_regularization_state_samples=config.symmetry_regularization_state_samples,
            symmetry_regularization_seed=config.symmetry_regularization_seed,
        )
        training_noise_control = _build_training_noise_control(
            config,
            job,
            resolved_noise_config,
            backend,
        )

        (run_output_dir / "noisy_job_config.json").write_text(
            json.dumps(
                _serialize_for_json(
                    {
                        "job": asdict(job),
                        "config": {
                            **asdict(config),
                            **_config_training_noise_aliases(config),
                            **_config_symmetry_regularization_aliases(config),
                        },
                        "resolved_num_qubits_values": list(config.resolved_num_qubits_values),
                        "resolved_symmetry_regularization_beta_values": list(
                            config.resolved_symmetry_regularization_beta_values
                        ),
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
            training_epoch_callback=training_noise_control["epoch_callback"],
            post_training_callback=training_noise_control["restore_callback"],
        )

        symmetry_diagnostic = _symmetry_diagnostic_for_run(
            config,
            dataset.test.states,
            model_family=job.model_family,
            num_qubits=job.num_qubits,
            backend=backend,
            result=result,
        )
        symmetry_twirled_evaluation = _symmetry_twirled_evaluation_for_run(
            config,
            dataset.test.states,
            dataset.test.labels,
            model_family=job.model_family,
            num_qubits=job.num_qubits,
            backend=backend,
            result=result,
        )

    runtime_summary = run_profile.summary()
    runtime_seconds = float(runtime_summary["noisy.single_run"]["total_seconds"])
    runtime_breakdown = dict(result.get("runtime_breakdown", {}))
    noise_parameters = resolved_noise_config.parameter_metadata()
    training_noise_metadata = _training_noise_metadata(config, result, training_noise_control)
    symmetry_regularization_metadata = _symmetry_regularization_metadata(
        config,
        result,
        beta=job.symmetry_regularization_beta,
    )
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
        "eval_noise_strength": float(job.noise_strength),
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
        "symmetry_twirled_available": bool(symmetry_twirled_evaluation["available"]),
        "symmetry_twirled_note": str(symmetry_twirled_evaluation["note"]),
        "symmetry_twirled_test_accuracy": symmetry_twirled_evaluation["twirled_accuracy"],
        "symmetry_twirled_train_accuracy": None,
        "symmetry_twirled_raw_subset_accuracy": symmetry_twirled_evaluation["raw_accuracy"],
        "symmetry_twirled_subset_size": symmetry_twirled_evaluation["subset_size"],
        "symmetry_twirled_num_correct_raw_subset": symmetry_twirled_evaluation["num_correct_raw"],
        "symmetry_twirled_num_correct_twirled_subset": symmetry_twirled_evaluation["num_correct_twirled"],
        "symmetry_twirled_mean_abs_shift": symmetry_twirled_evaluation["mean_abs_twirling_shift"],
        "num_symmetry_twirl_samples": int(config.num_symmetry_twirl_samples),
        "num_state_samples_for_twirled_evaluation": symmetry_twirled_evaluation["num_state_samples"],
        "noise_aware_training": bool(training_noise_metadata["noise_aware_training"]),
        "training_noise_strengths": training_noise_metadata["training_noise_strengths"],
        "train_noise_strength_values": training_noise_metadata["train_noise_strength_values"],
        "training_noise_sampling": training_noise_metadata["training_noise_sampling"],
        "train_noise_sampling_mode": training_noise_metadata["train_noise_sampling_mode"],
        "train_noise_includes_zero": training_noise_metadata["train_noise_includes_zero"],
        "training_noise_seed": training_noise_metadata["training_noise_seed"],
        "training_noise_effective_seed": training_noise_metadata["training_noise_effective_seed"],
        "training_noise_schedule": training_noise_metadata["training_noise_schedule"],
        "training_noise_strength_counts": training_noise_metadata["training_noise_strength_counts"],
        "training_noise_defaulted_from_evaluation_grid": training_noise_metadata[
            "training_noise_defaulted_from_evaluation_grid"
        ],
        "training_noise_note": training_noise_metadata["training_noise_note"],
        "symmetry_regularization": bool(symmetry_regularization_metadata["symmetry_regularization"]),
        "symmetry_regularization_enabled": bool(
            symmetry_regularization_metadata["symmetry_regularization_enabled"]
        ),
        "symmetry_regularization_beta": symmetry_regularization_metadata["symmetry_regularization_beta"],
        "symmetry_regularization_weight": symmetry_regularization_metadata["symmetry_regularization_weight"],
        "num_symmetry_regularization_samples": symmetry_regularization_metadata[
            "num_symmetry_regularization_samples"
        ],
        "symmetry_regularization_frequency": symmetry_regularization_metadata[
            "symmetry_regularization_frequency"
        ],
        "symmetry_regularization_state_samples": symmetry_regularization_metadata[
            "symmetry_regularization_state_samples"
        ],
        "symmetry_regularization_seed": symmetry_regularization_metadata["symmetry_regularization_seed"],
        "mean_symmetry_penalty_history": symmetry_regularization_metadata["mean_symmetry_penalty_history"],
        "final_symmetry_penalty": symmetry_regularization_metadata["final_symmetry_penalty"],
        "final_equivariance_error_mean": symmetry_diagnostic["mean_error"],
        "final_equivariance_error_max": symmetry_diagnostic["max_error"],
        "symmetry_regularization_note": symmetry_regularization_metadata["symmetry_regularization_note"],
        "output_dir": str(run_output_dir.resolve()),
    }
    run_metadata = {
        "job": asdict(job),
        "config": {
            **asdict(config),
            **_config_training_noise_aliases(config),
            **_config_symmetry_regularization_aliases(config),
        },
        "resolved_num_qubits_values": list(config.resolved_num_qubits_values),
        "resolved_symmetry_regularization_beta_values": list(config.resolved_symmetry_regularization_beta_values),
        "noise_config": resolved_noise_config.to_dict(),
        "noise_metadata": resolved_noise_config.to_metadata(),
        "dataset_metadata": result.get("dataset_metadata"),
        "experiment_config": result.get("experiment_config"),
        "training_config": result.get("training_config"),
        "runtime_profile": result.get("runtime_profile"),
        "runtime_breakdown": runtime_breakdown,
        "symmetry_diagnostic": symmetry_diagnostic,
        "symmetry_twirled_evaluation": symmetry_twirled_evaluation,
        "training_noise": training_noise_metadata,
        "symmetry_regularization": symmetry_regularization_metadata,
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


def _build_training_noise_control(
    config: NoisyComparisonConfig,
    job: NoisyComparisonJob,
    resolved_noise_config: NoiseConfig,
    backend: object,
) -> dict[str, Any]:
    if not config.noise_aware_training:
        return {
            "schedule": [],
            "counts": {},
            "epoch_callback": None,
            "restore_callback": None,
            "sampling_mode": "none",
            "effective_seed": None,
            "note": "disabled",
        }

    if not hasattr(backend, "noise_config"):
        return {
            "schedule": [],
            "counts": {},
            "epoch_callback": None,
            "restore_callback": None,
            "sampling_mode": "none",
            "effective_seed": None,
            "note": "backend_does_not_expose_noise_config",
        }

    effective_seed = config.training_noise_seed if config.training_noise_seed is not None else int(job.seed)
    rng = np.random.default_rng(effective_seed)
    strengths = np.asarray(config.training_noise_strengths, dtype=np.float64)
    schedule = [float(value) for value in rng.choice(strengths, size=int(job.epochs), replace=True)]
    counts = _noise_strength_counts(schedule)
    if schedule:
        backend.noise_config = _training_noise_config_from_strength(
            config,
            resolved_noise_config,
            float(schedule[0]),
        )

    def epoch_callback(epoch: int, model: object) -> dict[str, Any]:
        del model
        strength = float(schedule[int(epoch) - 1])
        backend.noise_config = _training_noise_config_from_strength(
            config,
            resolved_noise_config,
            strength,
        )
        return {
            "epoch": int(epoch),
            "training_noise_strength": strength,
        }

    def restore_callback(model: object) -> None:
        del model
        backend.noise_config = resolved_noise_config

    note = "per_epoch_noise_config_sampling"
    if config.training_noise_defaulted_from_evaluation_grid:
        note += ";train_noise_strength_values_defaulted_from_eval_noise_grid"

    return {
        "schedule": schedule,
        "counts": counts,
        "epoch_callback": epoch_callback,
        "restore_callback": restore_callback,
        "sampling_mode": _training_noise_sampling_mode(config),
        "effective_seed": effective_seed,
        "note": note,
    }


def _training_noise_config_from_strength(
    config: NoisyComparisonConfig,
    resolved_noise_config: NoiseConfig,
    strength: float,
) -> NoiseConfig:
    return noise_config_from_strength(
        config.noise_model_name,
        float(strength),
        coherent_overrotation_mode=config.coherent_overrotation_mode,
        coherent_overrotation_probability=config.coherent_overrotation_probability,
        coherent_overrotation_angle_std=config.coherent_overrotation_angle_std,
        coherent_overrotation_seed=config.coherent_overrotation_seed,
        noise_application_scope=resolved_noise_config.noise_application_scope,
        noisy_qubits=resolved_noise_config.noisy_qubits,
        single_qubit_error_profile=config.single_qubit_error_profile,
    )


def _training_noise_metadata(
    config: NoisyComparisonConfig,
    result: dict[str, Any],
    training_noise_control: dict[str, Any],
) -> dict[str, Any]:
    history = result.get("history", {})
    callback_history = history.get("epoch_callback", []) if isinstance(history, dict) else []
    schedule = list(training_noise_control.get("schedule", []))
    if callback_history and not schedule:
        schedule = [float(item["training_noise_strength"]) for item in callback_history]
    train_noise_strength_values = list(config.training_noise_strengths)
    sampling_mode = str(training_noise_control.get("sampling_mode", _training_noise_sampling_mode(config)))
    return {
        "noise_aware_training": bool(config.noise_aware_training),
        "training_noise_strengths": train_noise_strength_values,
        "train_noise_strength_values": train_noise_strength_values,
        "training_noise_sampling": _legacy_training_noise_sampling(config.training_noise_sampling),
        "train_noise_sampling_mode": sampling_mode,
        "train_noise_includes_zero": _noise_strengths_include_zero(train_noise_strength_values),
        "training_noise_seed": config.training_noise_seed,
        "training_noise_effective_seed": training_noise_control.get("effective_seed"),
        "training_noise_schedule": schedule,
        "training_noise_strength_counts": dict(training_noise_control.get("counts", _noise_strength_counts(schedule))),
        "training_noise_defaulted_from_evaluation_grid": bool(
            config.training_noise_defaulted_from_evaluation_grid
        ),
        "training_noise_note": str(training_noise_control.get("note", "")),
    }


def _config_training_noise_aliases(config: NoisyComparisonConfig) -> dict[str, Any]:
    train_noise_strength_values = list(config.training_noise_strengths)
    return {
        "train_noise_strength_values": train_noise_strength_values,
        "train_noise_sampling_mode": _training_noise_sampling_mode(config),
        "train_noise_includes_zero": _noise_strengths_include_zero(train_noise_strength_values),
    }


def _config_symmetry_regularization_aliases(config: NoisyComparisonConfig) -> dict[str, Any]:
    return {
        "symmetry_regularization_beta_sweep": bool(config.symmetry_regularization_beta_values),
    }


def _noise_strength_counts(schedule: list[float]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in schedule:
        key = f"{float(value):.12g}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _noise_strengths_include_zero(values: list[float] | tuple[float, ...] | None) -> bool:
    if not values:
        return False
    return any(np.isclose(float(value), 0.0, rtol=0.0, atol=1e-12) for value in values)


def _training_noise_sampling_mode(config: NoisyComparisonConfig) -> str:
    if not config.noise_aware_training:
        return "none"
    if config.training_noise_sampling in {"per_epoch", "per_epoch_random_choice"}:
        return "per_epoch_random_choice"
    return str(config.training_noise_sampling)


def _legacy_training_noise_sampling(value: str | None) -> str | None:
    if value in {"per_epoch", "per_epoch_random_choice"}:
        return "per_epoch"
    return None if value is None else str(value)


def _mean_optional_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and np.isfinite(float(row[key]))
    ]
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _symmetry_regularization_metadata(
    config: NoisyComparisonConfig,
    result: dict[str, Any],
    *,
    beta: float,
) -> dict[str, Any]:
    history = result.get("history", {})
    raw_penalties = history.get("symmetry_penalty", []) if isinstance(history, dict) else []
    penalties = [
        float(value)
        for value in raw_penalties
        if value is not None and np.isfinite(float(value))
    ]
    note = "disabled"
    if config.symmetry_regularization:
        note = str(history.get("symmetry_regularization_note", "finite_difference_objective_regularizer"))
    beta_value = float(beta)
    return {
        "symmetry_regularization": bool(config.symmetry_regularization),
        "symmetry_regularization_enabled": bool(config.symmetry_regularization and beta_value > 0.0),
        "symmetry_regularization_beta": beta_value,
        "symmetry_regularization_weight": beta_value,
        "num_symmetry_regularization_samples": int(config.num_symmetry_regularization_samples),
        "symmetry_regularization_frequency": int(config.symmetry_regularization_frequency),
        "symmetry_regularization_state_samples": config.symmetry_regularization_state_samples,
        "symmetry_regularization_seed": config.symmetry_regularization_seed,
        "mean_symmetry_penalty_history": float(np.mean(penalties)) if penalties else None,
        "final_symmetry_penalty": float(penalties[-1]) if penalties else None,
        "symmetry_regularization_note": note,
    }


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
    if config.noise_aware_training:
        strengths_slug = "_".join(_noise_strength_slug(value) for value in config.training_noise_strengths)
        path = path / "mitigation_noise_aware" / f"train_noise_{strengths_slug}"
    if config.symmetry_regularization:
        path = (
            path
            / "mitigation_symmetry_regularized"
            / _symmetry_regularization_beta_path_component(config, job.symmetry_regularization_beta)
        )
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
    if config.noise_aware_training:
        name += "_noise_aware_training"
    if config.symmetry_regularization:
        name += f"_symreg_beta_{_noise_strength_slug(job.symmetry_regularization_beta)}"
    return f"{name}_seed{job.seed}"


def _symmetry_regularization_beta_path_component(config: NoisyComparisonConfig, beta: float) -> str:
    prefix = "symreg_beta" if config.symmetry_regularization_beta_values else "beta"
    return f"{prefix}_{_noise_strength_slug(beta)}"


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


def _row_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
    return bool(value)


def _row_training_noise_strength_values(row: dict[str, Any]) -> Any:
    if not _row_bool(row.get("noise_aware_training", False)):
        return []
    values = row.get("train_noise_strength_values")
    if values is None:
        values = row.get("training_noise_strengths")
    return [] if values is None else values


def _row_training_noise_sampling(row: dict[str, Any]) -> str | None:
    if not _row_bool(row.get("noise_aware_training", False)):
        return None
    sampling = row.get("training_noise_sampling")
    if sampling is not None:
        return _legacy_training_noise_sampling(str(sampling))
    sampling_mode = row.get("train_noise_sampling_mode")
    if sampling_mode == "per_epoch_random_choice":
        return "per_epoch"
    return None if sampling_mode is None else str(sampling_mode)


def _row_train_noise_sampling_mode(row: dict[str, Any]) -> str:
    sampling_mode = row.get("train_noise_sampling_mode")
    if sampling_mode is not None:
        return str(sampling_mode)
    if not _row_bool(row.get("noise_aware_training", False)):
        return "none"
    sampling = row.get("training_noise_sampling")
    if sampling in {"per_epoch", "per_epoch_random_choice"}:
        return "per_epoch_random_choice"
    return "" if sampling is None else str(sampling)


def _row_train_noise_includes_zero(row: dict[str, Any]) -> bool:
    if row.get("train_noise_includes_zero") is not None:
        return _row_bool(row["train_noise_includes_zero"])
    values = _row_training_noise_strength_values(row)
    if values is None:
        return False
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except json.JSONDecodeError:
            values = [float(part) for part in values.strip("[]()").split(",") if part.strip()]
    if isinstance(values, (int, float, np.generic)):
        values = [float(values)]
    return _noise_strengths_include_zero(list(values))


def _row_symmetry_regularization_beta(row: dict[str, Any]) -> float | None:
    beta = row.get("symmetry_regularization_beta")
    if beta is not None:
        return float(beta)
    weight = row.get("symmetry_regularization_weight")
    if weight is not None:
        return float(weight)
    return None


def _row_symmetry_regularization_weight(row: dict[str, Any]) -> float | None:
    weight = row.get("symmetry_regularization_weight")
    if weight is not None:
        return float(weight)
    return _row_symmetry_regularization_beta(row)


def _row_symmetry_regularization_enabled(row: dict[str, Any]) -> bool:
    if row.get("symmetry_regularization_enabled") is not None:
        return _row_bool(row["symmetry_regularization_enabled"])
    beta = _row_symmetry_regularization_beta(row)
    return _row_bool(row.get("symmetry_regularization", False)) and beta is not None and float(beta) > 0.0


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


def _symmetry_twirled_evaluation_for_run(
    config: NoisyComparisonConfig,
    test_states: np.ndarray,
    test_labels: np.ndarray,
    *,
    model_family: str,
    num_qubits: int,
    backend: object,
    result: dict[str, Any],
) -> dict[str, Any]:
    if not config.compute_symmetry_twirled_evaluation:
        return {
            "available": False,
            "note": "disabled",
            "twirled_accuracy": None,
            "raw_accuracy": None,
            "mean_abs_twirling_shift": None,
            "num_state_samples": 0,
            "subset_size": 0,
            "num_correct_raw": None,
            "num_correct_twirled": None,
            "symmetry_twirled_raw_subset_accuracy": None,
            "symmetry_twirled_test_accuracy": None,
            "symmetry_twirled_subset_size": 0,
            "symmetry_twirled_num_correct_raw_subset": None,
            "symmetry_twirled_num_correct_twirled_subset": None,
        }

    try:
        best_parameters = np.asarray(result["history"]["best_parameters"], dtype=np.float64)
        twirling_model = build_model(
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
        threshold = float(result["classification_threshold"])
        if hasattr(twirling_model, "set_classification_threshold"):
            twirling_model.set_classification_threshold(threshold)

        if config.num_state_samples_for_twirled_evaluation is None:
            subset_size = int(test_states.shape[0])
        else:
            subset_size = min(int(config.num_state_samples_for_twirled_evaluation), int(test_states.shape[0]))
        evaluation_states = np.asarray(test_states[:subset_size], dtype=np.complex128)
        evaluation_labels = np.asarray(test_labels[:subset_size], dtype=np.int64)
        evaluation = evaluate_with_symmetry_twirling(
            twirling_model,
            evaluation_states,
            parameters=best_parameters,
            backend=backend,
            labels=evaluation_labels,
            threshold=threshold,
            num_symmetry_samples=config.num_symmetry_twirl_samples,
            seed=config.symmetry_twirl_seed,
        )
        return {
            "available": bool(evaluation.get("symmetry_twirling_available", False)),
            "note": str(evaluation.get("symmetry_twirling_note", "")),
            "raw_probabilities": evaluation.get("raw_probabilities"),
            "twirled_probabilities": evaluation.get("twirled_probabilities"),
            "raw_accuracy": evaluation.get("raw_accuracy"),
            "twirled_accuracy": evaluation.get("twirled_accuracy"),
            "num_correct_raw": evaluation.get("num_correct_raw"),
            "num_correct_twirled": evaluation.get("num_correct_twirled"),
            "symmetry_twirled_raw_subset_accuracy": evaluation.get("symmetry_twirled_raw_subset_accuracy"),
            "symmetry_twirled_test_accuracy": evaluation.get("symmetry_twirled_test_accuracy"),
            "symmetry_twirled_subset_size": evaluation.get("symmetry_twirled_subset_size"),
            "symmetry_twirled_num_correct_raw_subset": evaluation.get(
                "symmetry_twirled_num_correct_raw_subset"
            ),
            "symmetry_twirled_num_correct_twirled_subset": evaluation.get(
                "symmetry_twirled_num_correct_twirled_subset"
            ),
            "mean_abs_twirling_shift": evaluation.get("mean_abs_twirling_shift"),
            "num_state_samples": evaluation.get("num_state_samples", 0),
            "subset_size": evaluation.get(
                "symmetry_twirled_subset_size",
                evaluation.get("num_state_samples", 0),
            ),
            "num_symmetry_samples": evaluation.get("num_symmetry_samples", config.num_symmetry_twirl_samples),
        }
    except Exception as exc:  # pragma: no cover - defensive path for optional evaluation
        return {
            "available": False,
            "note": f"not_available: {type(exc).__name__}: {exc}",
            "twirled_accuracy": None,
            "raw_accuracy": None,
            "mean_abs_twirling_shift": None,
            "num_state_samples": 0,
            "subset_size": 0,
            "num_correct_raw": None,
            "num_correct_twirled": None,
            "symmetry_twirled_raw_subset_accuracy": None,
            "symmetry_twirled_test_accuracy": None,
            "symmetry_twirled_subset_size": 0,
            "symmetry_twirled_num_correct_raw_subset": None,
            "symmetry_twirled_num_correct_twirled_subset": None,
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
        "eval_noise_strength",
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
        "noise_aware_training",
        "training_noise_strengths",
        "train_noise_strength_values",
        "training_noise_sampling",
        "train_noise_sampling_mode",
        "train_noise_includes_zero",
        "training_noise_seed",
        "training_noise_effective_seed",
        "training_noise_strength_counts",
        "training_noise_defaulted_from_evaluation_grid",
        "training_noise_note",
        "symmetry_regularization",
        "symmetry_regularization_enabled",
        "symmetry_regularization_beta",
        "symmetry_regularization_weight",
        "num_symmetry_regularization_samples",
        "symmetry_regularization_frequency",
        "symmetry_regularization_state_samples",
        "symmetry_regularization_seed",
        "symmetry_regularization_note",
        "mean_symmetry_penalty_history",
        "final_symmetry_penalty",
        "final_equivariance_error_mean",
        "final_equivariance_error_max",
        "num_runs",
        "symmetry_twirled_available",
        "symmetry_twirled_note",
        "num_symmetry_twirl_samples",
        "num_state_samples_for_twirled_evaluation",
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
        "mean_symmetry_twirled_test_accuracy",
        "variance_symmetry_twirled_test_accuracy",
        "mean_symmetry_twirled_train_accuracy",
        "variance_symmetry_twirled_train_accuracy",
        "mean_symmetry_twirled_raw_subset_accuracy",
        "variance_symmetry_twirled_raw_subset_accuracy",
        "mean_symmetry_twirled_subset_size",
        "variance_symmetry_twirled_subset_size",
        "mean_symmetry_twirled_num_correct_raw_subset",
        "variance_symmetry_twirled_num_correct_raw_subset",
        "mean_symmetry_twirled_num_correct_twirled_subset",
        "variance_symmetry_twirled_num_correct_twirled_subset",
        "mean_symmetry_twirled_mean_abs_shift",
        "variance_symmetry_twirled_mean_abs_shift",
        "mean_mean_symmetry_penalty_history",
        "variance_mean_symmetry_penalty_history",
        "mean_final_symmetry_penalty",
        "variance_final_symmetry_penalty",
        "mean_final_equivariance_error_mean",
        "variance_final_equivariance_error_mean",
        "mean_final_equivariance_error_max",
        "variance_final_equivariance_error_max",
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
