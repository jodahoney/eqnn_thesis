"""Experiment running and benchmark sweep utilities."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eqnn.datasets.heisenberg import DatasetBundle, HeisenbergDatasetConfig, generate_dataset
from eqnn.datasets.io import load_dataset_bundle, save_dataset_bundle
from eqnn.models import BaselineQCNN, BaselineQCNNConfig, QCNNConfig, SU2QCNN
from eqnn.training import Trainer, TrainingConfig
from eqnn.utils.timing import RuntimeProfile, timed


@dataclass(frozen=True)
class ExperimentConfig:
    model_family: str = "su2_qcnn"
    num_qubits: int = 4
    min_readout_qubits: int | None = None
    boundary: str = "open"
    parity_sequence: tuple[str, ...] = ("even", "odd")
    shared_convolution_parameter: bool = True
    pooling_mode: str = "partial_trace"
    pooling_keep: str = "left"
    readout_mode: str = "swap"

    def __post_init__(self) -> None:
        if self.model_family not in {"su2_qcnn", "baseline_qcnn"}:
            raise ValueError("model_family must be 'su2_qcnn' or 'baseline_qcnn'")


@dataclass(frozen=True)
class BenchmarkSweepConfig:
    num_qubits_values: tuple[int, ...] = (4,)
    model_families: tuple[str, ...] = ("su2_qcnn", "baseline_qcnn")
    labeling_strategies: tuple[str, ...] = ("ratio_threshold",)
    pooling_modes: tuple[str, ...] = ("partial_trace",)
    readout_modes: tuple[str, ...] = ("swap",)
    split_seeds: tuple[int, ...] = (0,)
    ratio_min: float = 0.4
    ratio_max: float = 1.6
    num_points: int = 9
    train_fraction: float = 0.8
    critical_ratio: float = 1.0
    exclusion_window: float = 0.05
    diagnostic_window: float = 0.05
    boundary: str = "open"
    eigensolver: str = "auto"
    partial_reflection_pairs: int | None = None
    training_config: TrainingConfig = field(default_factory=TrainingConfig)


def build_model(
    config: ExperimentConfig,
    parameters: np.ndarray | None = None,
) -> object:
    if config.model_family == "su2_qcnn":
        return SU2QCNN(
            QCNNConfig(
                num_qubits=config.num_qubits,
                min_readout_qubits=config.min_readout_qubits,
                boundary=config.boundary,
                parity_sequence=config.parity_sequence,
                shared_convolution_parameter=config.shared_convolution_parameter,
                pooling_mode=config.pooling_mode,
                pooling_keep=config.pooling_keep,
                readout_mode=config.readout_mode,
            ),
            parameters=parameters,
        )

    return BaselineQCNN(
        BaselineQCNNConfig(
            num_qubits=config.num_qubits,
            min_readout_qubits=config.min_readout_qubits,
            boundary=config.boundary,
            parity_sequence=config.parity_sequence,
            shared_convolution_parameter=config.shared_convolution_parameter,
            pooling_mode=config.pooling_mode,
            pooling_keep=config.pooling_keep,
            readout_mode=config.readout_mode,
        ),
        parameters=parameters,
    )


def load_or_generate_dataset(
    *,
    dataset_dir: str | Path | None = None,
    dataset_config: HeisenbergDatasetConfig | None = None,
    profile: RuntimeProfile | None = None,
) -> DatasetBundle:
    if dataset_dir is not None:
        with timed(profile, "dataset.load_bundle"):
            return load_dataset_bundle(dataset_dir, profile=profile)
    if dataset_config is None:
        raise ValueError("Either dataset_dir or dataset_config must be provided")
    with timed(profile, "dataset.generate_bundle"):
        return generate_dataset(dataset_config, profile=profile)


def run_training_experiment(
    dataset: DatasetBundle,
    experiment_config: ExperimentConfig,
    training_config: TrainingConfig,
    *,
    output_dir: str | Path | None = None,
    experiment_name: str | None = None,
    profile: RuntimeProfile | None = None,
) -> dict[str, Any]:
    with timed(profile, "experiment.build_model"):
        model = build_model(experiment_config)

    with timed(profile, "experiment.build_trainer"):
        trainer = Trainer(training_config)

    with timed(profile, "experiment.train_fit"):
        history = trainer.fit(model, dataset, profile=profile)

    with timed(profile, "experiment.final_train_evaluate"):
        train_metrics = trainer.evaluate(model, dataset.train, profile=profile)

    with timed(profile, "experiment.final_test_evaluate"):
        test_metrics = trainer.evaluate(model, dataset.test, profile=profile)

    with timed(profile, "experiment.train_predict"):
        train_probabilities = model.predict_batch(dataset.train.states)
        train_predicted_labels = (
            model.predict_labels_batch(dataset.train.states)
            if hasattr(model, "predict_labels_batch")
            else (train_probabilities >= 0.5).astype(np.int64)
        )

    with timed(profile, "experiment.test_predict"):
        test_probabilities = model.predict_batch(dataset.test.states)
        test_predicted_labels = (
            model.predict_labels_batch(dataset.test.states)
            if hasattr(model, "predict_labels_batch")
            else (test_probabilities >= 0.5).astype(np.int64)
        )

    result = {
        "experiment_name": experiment_name or _default_experiment_name(experiment_config),
        "experiment_config": asdict(experiment_config),
        "training_config": asdict(training_config),
        "dataset_metadata": dataset.metadata,
        "parameter_count": int(model.parameter_count),
        "classification_threshold": (
            float(model.get_classification_threshold())
            if hasattr(model, "get_classification_threshold")
            else 0.5
        ),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "history": _serialize_for_json(history),
    }

    output_path: Path | None = None
    if output_dir is not None:
        output_path = Path(output_dir)

        with timed(profile, "experiment.prepare_output_dir"):
            output_path.mkdir(parents=True, exist_ok=True)

        with timed(profile, "experiment.write_artifacts"):
            _save_experiment_artifacts(
                output_path=output_path,
                dataset=dataset,
                model=model,
                train_probabilities=train_probabilities,
                test_probabilities=test_probabilities,
                train_predicted_labels=train_predicted_labels,
                test_predicted_labels=test_predicted_labels,
            )

        result["output_dir"] = str(output_path.resolve())

    if profile is not None:
        result["runtime_profile"] = _serialize_for_json(profile.summary())

    if output_path is not None:
        (output_path / "metrics.json").write_text(
            json.dumps(_serialize_for_json(result), indent=2, sort_keys=True) + "\n"
        )

    return result


def run_benchmark_sweep(
    config: BenchmarkSweepConfig,
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    dataset_cache: dict[tuple[int, str, int], DatasetBundle] = {}

    for num_qubits in config.num_qubits_values:
        for labeling_strategy in config.labeling_strategies:
            for split_seed in config.split_seeds:
                dataset_key = (num_qubits, labeling_strategy, split_seed)
                if dataset_key not in dataset_cache:
                    dataset_bundle = generate_dataset(
                        HeisenbergDatasetConfig(
                            num_qubits=num_qubits,
                            ratio_min=config.ratio_min,
                            ratio_max=config.ratio_max,
                            num_points=config.num_points,
                            train_fraction=config.train_fraction,
                            critical_ratio=config.critical_ratio,
                            exclusion_window=config.exclusion_window,
                            boundary=config.boundary,
                            eigensolver=config.eigensolver,
                            labeling_strategy=labeling_strategy,
                            diagnostic_window=config.diagnostic_window,
                            partial_reflection_pairs=config.partial_reflection_pairs,
                            split_seed=split_seed,
                        )
                    )
                    dataset_cache[dataset_key] = dataset_bundle
                    save_dataset_bundle(
                        dataset_bundle,
                        output_path / "datasets" / _dataset_name(num_qubits, labeling_strategy, split_seed),
                    )
                dataset_bundle = dataset_cache[dataset_key]

                for model_family in config.model_families:
                    for pooling_mode in config.pooling_modes:
                        for readout_mode in config.readout_modes:
                            experiment_config = ExperimentConfig(
                                model_family=model_family,
                                num_qubits=num_qubits,
                                boundary=config.boundary,
                                pooling_mode=pooling_mode,
                                readout_mode=readout_mode,
                            )
                            experiment_id = _sweep_experiment_id(
                                model_family=model_family,
                                num_qubits=num_qubits,
                                labeling_strategy=labeling_strategy,
                                pooling_mode=pooling_mode,
                                readout_mode=readout_mode,
                                split_seed=split_seed,
                            )
                            experiment_output = output_path / "experiments" / experiment_id
                            result = run_training_experiment(
                                dataset_bundle,
                                experiment_config,
                                config.training_config,
                                output_dir=experiment_output,
                                experiment_name=experiment_id,
                            )
                            results.append(result)
                            summary_rows.append(
                                {
                                    "experiment_id": experiment_id,
                                    "model_family": model_family,
                                    "num_qubits": num_qubits,
                                    "labeling_strategy": labeling_strategy,
                                    "pooling_mode": pooling_mode,
                                    "readout_mode": readout_mode,
                                    "split_seed": split_seed,
                                    "train_loss": result["train_metrics"]["loss"],
                                    "train_accuracy": result["train_metrics"]["accuracy"],
                                    "test_loss": result["test_metrics"]["loss"],
                                    "test_accuracy": result["test_metrics"]["accuracy"],
                                    "output_dir": result["output_dir"],
                                }
                            )

    (output_path / "summary.json").write_text(
        json.dumps(_serialize_for_json(summary_rows), indent=2, sort_keys=True) + "\n"
    )
    _write_summary_csv(output_path / "summary.csv", summary_rows)
    return results


def _save_experiment_artifacts(
    *,
    output_path: Path,
    dataset: DatasetBundle,
    model: object,
    train_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    train_predicted_labels: np.ndarray,
    test_predicted_labels: np.ndarray,
) -> None:
    np.save(output_path / "best_parameters.npy", model.get_parameters())
    np.savez_compressed(
        output_path / "train_predictions.npz",
        probabilities=np.asarray(train_probabilities, dtype=np.float64),
        predicted_labels=np.asarray(train_predicted_labels, dtype=np.int64),
        labels=dataset.train.labels,
        coupling_ratios=dataset.train.coupling_ratios,
    )
    np.savez_compressed(
        output_path / "test_predictions.npz",
        probabilities=np.asarray(test_probabilities, dtype=np.float64),
        predicted_labels=np.asarray(test_predicted_labels, dtype=np.int64),
        labels=dataset.test.labels,
        coupling_ratios=dataset.test.coupling_ratios,
    )


def _serialize_for_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _serialize_for_json(asdict(value))
    if isinstance(value, dict):
        return {str(key): _serialize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_json(item) for item in value]
    return value


def _default_experiment_name(config: ExperimentConfig) -> str:
    return f"{config.model_family}_n{config.num_qubits}_{config.pooling_mode}_{config.readout_mode}"


def _dataset_name(num_qubits: int, labeling_strategy: str, split_seed: int) -> str:
    return f"n{num_qubits}_{labeling_strategy}_seed{split_seed}"


def _sweep_experiment_id(
    *,
    model_family: str,
    num_qubits: int,
    labeling_strategy: str,
    pooling_mode: str,
    readout_mode: str,
    split_seed: int,
) -> str:
    return (
        f"{model_family}_n{num_qubits}_{labeling_strategy}_"
        f"{pooling_mode}_{readout_mode}_seed{split_seed}"
    )


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "experiment_id",
        "model_family",
        "num_qubits",
        "labeling_strategy",
        "pooling_mode",
        "readout_mode",
        "split_seed",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "output_dir",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)