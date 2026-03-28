"""Locked paper-reproduction utilities for the SU(2)-EQCNN baseline."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eqnn.datasets.heisenberg import DatasetBundle, DatasetSplit
from eqnn.datasets.io import load_dataset_bundle, save_dataset_bundle
from eqnn.experiments.runner import ExperimentConfig, run_training_experiment
from eqnn.physics.heisenberg import BondAlternatingHeisenbergHamiltonian
from eqnn.physics.observables import alternating_singlet_means, dimerization_feature
from eqnn.physics.quantum import as_density_matrix
from eqnn.training import TrainingConfig
from eqnn.utils.timing import RuntimeProfile, timed


PAPER_DATASET_CACHE_NAMESPACE = "paper_reproduction_dataset_v1"


@dataclass(frozen=True)
class PaperDatasetConfig:
    """Dataset configuration for the locked paper-reproduction baseline.

    The training grid is chosen symmetrically around the critical point so that
    every training set contains samples on both sides of the transition.
    """

    num_qubits: int
    train_size: int
    critical_ratio: float = 1.0
    left_ratio_min: float = 0.0
    right_ratio_max: float = 2.0
    dense_test_points: int = 101
    boundary: str = "open"
    eigensolver: str = "auto"

    def __post_init__(self) -> None:
        if self.num_qubits < 2:
            raise ValueError("num_qubits must be at least 2")
        if self.train_size < 2 or self.train_size % 2 != 0:
            raise ValueError("train_size must be an even integer at least 2")
        if self.left_ratio_min >= self.critical_ratio:
            raise ValueError("left_ratio_min must lie below the critical ratio")
        if self.right_ratio_max <= self.critical_ratio:
            raise ValueError("right_ratio_max must lie above the critical ratio")
        if self.dense_test_points < 3:
            raise ValueError("dense_test_points must be at least 3")
        if self.boundary != "open":
            raise ValueError("paper reproduction is currently locked to open boundary conditions")
        if self.eigensolver not in {"auto", "dense", "sparse"}:
            raise ValueError("eigensolver must be 'auto', 'dense', or 'sparse'")


@dataclass(frozen=True)
class PaperReproductionConfig:
    """Frozen reproduction mode for the original SU(2)-EQCNN benchmark."""

    num_qubits: int
    train_sizes: tuple[int, ...] = (2, 4, 6, 8, 10, 12)
    random_seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 750
    learning_rate: float = 5e-2
    gradient_backend: str = "exact"
    initialization_strategy: str = "noisy_current"
    initialization_noise_scale: float = 5e-2
    critical_ratio: float = 1.0
    left_ratio_min: float = 0.0
    right_ratio_max: float = 2.0
    dense_test_points: int = 101
    eigensolver: str = "auto"

    def __post_init__(self) -> None:
        if not self.train_sizes:
            raise ValueError("train_sizes must not be empty")
        if not self.random_seeds:
            raise ValueError("random_seeds must not be empty")


def paper_training_ratios(config: PaperDatasetConfig) -> np.ndarray:
    """Return the symmetric training grid used for reproduction runs."""

    per_phase = config.train_size // 2
    left = np.linspace(
        config.left_ratio_min,
        config.critical_ratio,
        num=per_phase + 2,
        dtype=np.float64,
    )[1:-1]
    right = np.linspace(
        config.critical_ratio,
        config.right_ratio_max,
        num=per_phase + 2,
        dtype=np.float64,
    )[1:-1]
    return np.asarray(np.concatenate([left, right]), dtype=np.float64)


def paper_test_ratios(config: PaperDatasetConfig) -> np.ndarray:
    """Return the dense phase-diagram grid, excluding the exact critical point."""

    ratios = np.linspace(
        config.left_ratio_min,
        config.right_ratio_max,
        num=config.dense_test_points,
        dtype=np.float64,
    )
    mask = np.abs(ratios - config.critical_ratio) > 1e-12
    filtered = ratios[mask]
    if filtered.size < 2:
        raise ValueError("dense_test_points leaves too few usable test ratios after excluding the critical point")
    return filtered


def generate_paper_dataset(
    config: PaperDatasetConfig,
    *,
    profile: RuntimeProfile | None = None,
) -> DatasetBundle:
    """Generate the locked training/test splits for paper-style reproduction."""

    with timed(profile, "paper.dataset.total"):
        with timed(profile, "paper.dataset.ratio_grid"):
            train_ratios = paper_training_ratios(config)
            test_ratios = paper_test_ratios(config)

        hamiltonian = BondAlternatingHeisenbergHamiltonian(
            num_qubits=config.num_qubits,
            boundary=config.boundary,
        )

        with timed(profile, "paper.dataset.train_split"):
            train_split = _build_split_from_ratios(
                hamiltonian,
                config,
                train_ratios,
                profile=profile,
            )

        with timed(profile, "paper.dataset.test_split"):
            test_split = _build_split_from_ratios(
                hamiltonian,
                config,
                test_ratios,
                profile=profile,
            )

        metadata = {
            "dataset_name": "bond_alternating_heisenberg_paper_reproduction_v1",
            "protocol": "paper_reproduction_v1",
            "num_qubits": int(config.num_qubits),
            "boundary": config.boundary,
            "critical_ratio": float(config.critical_ratio),
            "left_ratio_min": float(config.left_ratio_min),
            "right_ratio_max": float(config.right_ratio_max),
            "train_size": int(config.train_size),
            "dense_test_points_requested": int(config.dense_test_points),
            "dense_test_points_used": int(test_ratios.size),
            "train_ratios": train_ratios.tolist(),
            "test_ratios": test_ratios.tolist(),
            "eigensolver_requested": config.eigensolver,
            "eigensolver_resolved": hamiltonian.resolve_ground_state_method(config.eigensolver),
            "phase_labels": {
                "1": "trivial phase (coupling_ratio < critical_ratio)",
                "0": "topological phase (coupling_ratio > critical_ratio)",
            },
            "label_convention": "paper_reproduction_v1_swap_aligned",
            "notes": [
                "This dataset is locked to the paper_reproduction_v1 protocol.",
                "The training grid is symmetric around the critical ratio so every train set spans both phases.",
                "The dense test grid omits the exact critical point to keep labels unambiguous.",
                "Paper reproduction labels are aligned with the SWAP readout direction: larger outputs map to the trivial phase.",
            ],
        }
        return DatasetBundle(train=train_split, test=test_split, metadata=metadata)


def run_paper_reproduction_suite(
    config: PaperReproductionConfig,
    output_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    force_rebuild: bool = False,
    profile: RuntimeProfile | None = None,
) -> dict[str, Any]:
    """Run the locked reproduction baseline across train sizes and seeds."""

    output_path = Path(output_dir)
    with timed(profile, "paper.output.prepare_root"):
        output_path.mkdir(parents=True, exist_ok=True)

    with timed(profile, "paper.write_config"):
        (output_path / "paper_reproduction_config.json").write_text(
            json.dumps(asdict(config), indent=2, sort_keys=True) + "\n"
        )

    summary_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for train_size in config.train_sizes:
        dataset_config = PaperDatasetConfig(
            num_qubits=config.num_qubits,
            train_size=train_size,
            critical_ratio=config.critical_ratio,
            left_ratio_min=config.left_ratio_min,
            right_ratio_max=config.right_ratio_max,
            dense_test_points=config.dense_test_points,
            eigensolver=config.eigensolver,
        )

        with timed(profile, "paper.dataset.acquire"):
            if cache_dir is not None:
                dataset, cache_path, cache_hit = _load_or_generate_cached_paper_dataset(
                    dataset_config,
                    cache_dir=cache_dir,
                    profile=profile,
                    force_rebuild=force_rebuild,
                )
            else:
                dataset = generate_paper_dataset(dataset_config, profile=profile)
                cache_path = None
                cache_hit = False

        dataset_dir = output_path / "datasets" / f"train_size_{train_size}"
        with timed(profile, "paper.dataset.save_output_copy"):
            save_dataset_bundle(dataset, dataset_dir, profile=profile)

        train_size_results: list[dict[str, Any]] = []
        experiment_root = output_path / "experiments" / f"train_size_{train_size}"

        for seed in config.random_seeds:
            training_config = TrainingConfig(
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                loss="mse",
                batch_size=2,
                gradient_backend=config.gradient_backend,
                optimizer="adam",
                initialization_strategy=config.initialization_strategy,
                initialization_noise_scale=config.initialization_noise_scale,
                num_restarts=1,
                random_seed=seed,
                classification_threshold=0.5,
                threshold_update="paper_nearest_critical",
                threshold_critical_ratio=config.critical_ratio,
            )
            experiment_name = (
                f"paper_reproduction_v1_n{config.num_qubits}_train{train_size}_seed{seed}"
            )
            experiment_output = experiment_root / f"seed_{seed}"

            with timed(profile, "paper.run_single_experiment"):
                result = run_training_experiment(
                    dataset,
                    _paper_experiment_config(config.num_qubits),
                    training_config,
                    output_dir=experiment_output,
                    experiment_name=experiment_name,
                    profile=profile,
                )

            result["train_size"] = train_size
            result["seed"] = seed
            if cache_path is not None:
                result["dataset_cache_dir"] = str(cache_path.resolve())
                result["dataset_cache_hit"] = bool(cache_hit)

            train_size_results.append(result)
            run_rows.append(
                {
                    "train_size": train_size,
                    "seed": seed,
                    "test_accuracy": float(result["test_metrics"]["accuracy"]),
                    "test_loss": float(result["test_metrics"]["loss"]),
                    "classification_threshold": float(result["classification_threshold"]),
                    "output_dir": str(experiment_output.resolve()),
                    "dataset_cache_hit": bool(cache_hit),
                    "dataset_cache_dir": str(cache_path.resolve()) if cache_path is not None else None,
                }
            )

        with timed(profile, "paper.aggregate_train_size"):
            summary_rows.append(
                _aggregate_reproduction_results(
                    train_size=train_size,
                    train_size_results=train_size_results,
                    experiment_root=experiment_root,
                    output_dir=output_path / "phase_diagrams",
                )
            )

    with timed(profile, "paper.write_summary_json"):
        (output_path / "summary.json").write_text(
            json.dumps(summary_rows, indent=2, sort_keys=True) + "\n"
        )
    with timed(profile, "paper.write_summary_csv"):
        _write_paper_summary_csv(output_path / "summary.csv", summary_rows)
    with timed(profile, "paper.write_runs_json"):
        (output_path / "runs.json").write_text(
            json.dumps(run_rows, indent=2, sort_keys=True) + "\n"
        )

    return {"summary": summary_rows, "runs": run_rows}


def _paper_experiment_config(num_qubits: int) -> ExperimentConfig:
    return ExperimentConfig(
        model_family="su2_qcnn",
        num_qubits=num_qubits,
        boundary="open",
        shared_convolution_parameter=True,
        pooling_mode="partial_trace",
        pooling_keep="left",
        readout_mode="swap",
    )


def _build_split_from_ratios(
    hamiltonian: BondAlternatingHeisenbergHamiltonian,
    config: PaperDatasetConfig,
    ratios: np.ndarray,
    *,
    profile: RuntimeProfile | None = None,
) -> DatasetSplit:
    states = np.zeros((ratios.size, hamiltonian.dimension), dtype=np.complex128)
    labels = np.zeros(ratios.size, dtype=np.int64)
    energies = np.zeros(ratios.size, dtype=np.float64)
    primary_singlet_means = np.zeros(ratios.size, dtype=np.float64)
    secondary_singlet_means = np.zeros(ratios.size, dtype=np.float64)
    dimerization_features = np.zeros(ratios.size, dtype=np.float64)

    for index, ratio in enumerate(ratios):
        with timed(profile, "paper.dataset.ground_state"):
            energy, state = hamiltonian.ground_state(float(ratio), method=config.eigensolver)

        states[index] = state
        energies[index] = energy
        labels[index] = 1 if float(ratio) < config.critical_ratio else 0

        with timed(profile, "paper.dataset.diagnostics"):
            density_matrix = as_density_matrix(state)
            primary_mean, secondary_mean = alternating_singlet_means(
                density_matrix,
                config.num_qubits,
                boundary=config.boundary,
            )
            primary_singlet_means[index] = primary_mean
            secondary_singlet_means[index] = secondary_mean
            dimerization_features[index] = dimerization_feature(
                density_matrix,
                config.num_qubits,
                boundary=config.boundary,
            )

    diagnostics = {
        "primary_singlet_mean": primary_singlet_means,
        "secondary_singlet_mean": secondary_singlet_means,
        "dimerization_feature": dimerization_features,
    }
    return DatasetSplit(
        states=states,
        labels=labels,
        coupling_ratios=ratios,
        ground_state_energies=energies,
        diagnostics=diagnostics,
    )


def _aggregate_reproduction_results(
    *,
    train_size: int,
    train_size_results: list[dict[str, Any]],
    experiment_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    thresholds = np.asarray(
        [float(result["classification_threshold"]) for result in train_size_results],
        dtype=np.float64,
    )
    train_accuracies = np.asarray(
        [float(result["train_metrics"]["accuracy"]) for result in train_size_results],
        dtype=np.float64,
    )
    test_accuracies = np.asarray(
        [float(result["test_metrics"]["accuracy"]) for result in train_size_results],
        dtype=np.float64,
    )
    train_losses = np.asarray(
        [float(result["train_metrics"]["loss"]) for result in train_size_results],
        dtype=np.float64,
    )
    test_losses = np.asarray(
        [float(result["test_metrics"]["loss"]) for result in train_size_results],
        dtype=np.float64,
    )

    probabilities_by_seed: list[np.ndarray] = []
    predicted_labels_by_seed: list[np.ndarray] = []
    coupling_ratios: np.ndarray | None = None
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed_index in range(len(train_size_results)):
        prediction_path = experiment_root / f"seed_{train_size_results[seed_index]['seed']}" / "test_predictions.npz"
        with np.load(prediction_path, allow_pickle=False) as data:
            probabilities_by_seed.append(np.asarray(data["probabilities"], dtype=np.float64))
            predicted_labels_by_seed.append(np.asarray(data["predicted_labels"], dtype=np.int64))
            if coupling_ratios is None:
                coupling_ratios = np.asarray(data["coupling_ratios"], dtype=np.float64)

    probability_stack = np.stack(probabilities_by_seed, axis=0)
    predicted_label_stack = np.stack(predicted_labels_by_seed, axis=0)
    phase_diagram_path = output_dir / f"train_size_{train_size}_phase_diagram.npz"
    np.savez_compressed(
        phase_diagram_path,
        coupling_ratios=np.asarray(coupling_ratios, dtype=np.float64),
        mean_probabilities=np.mean(probability_stack, axis=0),
        variance_probabilities=np.var(probability_stack, axis=0),
        mean_predicted_labels=np.mean(predicted_label_stack, axis=0),
        variance_predicted_labels=np.var(predicted_label_stack, axis=0),
    )

    return {
        "train_size": int(train_size),
        "num_runs": len(train_size_results),
        "mean_train_accuracy": float(np.mean(train_accuracies)),
        "variance_train_accuracy": float(np.var(train_accuracies)),
        "mean_test_accuracy": float(np.mean(test_accuracies)),
        "variance_test_accuracy": float(np.var(test_accuracies)),
        "mean_train_loss": float(np.mean(train_losses)),
        "variance_train_loss": float(np.var(train_losses)),
        "mean_test_loss": float(np.mean(test_losses)),
        "variance_test_loss": float(np.var(test_losses)),
        "mean_threshold": float(np.mean(thresholds)),
        "variance_threshold": float(np.var(thresholds)),
        "phase_diagram_path": str(phase_diagram_path.resolve()),
    }


def _write_paper_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
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
        "phase_diagram_path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _paper_dataset_cache_key(config: PaperDatasetConfig) -> str:
    payload = {
        "namespace": PAPER_DATASET_CACHE_NAMESPACE,
        "config": asdict(config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _load_or_generate_cached_paper_dataset(
    config: PaperDatasetConfig,
    *,
    cache_dir: str | Path,
    profile: RuntimeProfile | None = None,
    force_rebuild: bool = False,
) -> tuple[DatasetBundle, Path, bool]:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    target_dir = cache_root / _paper_dataset_cache_key(config)

    if target_dir.exists() and not force_rebuild:
        with timed(profile, "paper.cache.dataset_load"):
            bundle = load_dataset_bundle(target_dir, profile=profile)
        return bundle, target_dir, True

    with timed(profile, "paper.cache.dataset_generate"):
        bundle = generate_paper_dataset(config, profile=profile)

    with timed(profile, "paper.cache.dataset_save"):
        save_dataset_bundle(bundle, target_dir, profile=profile)

    return bundle, target_dir, False