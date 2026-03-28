"""Dataset generation for the bond-alternating Heisenberg benchmark."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np

from eqnn.physics.heisenberg import BondAlternatingHeisenbergHamiltonian
from eqnn.physics.observables import alternating_singlet_means, dimerization_feature
from eqnn.physics.quantum import as_density_matrix
from eqnn.physics.topology import (
    calibrated_partial_reflection_score,
    default_partial_reflection_pairs,
    normalized_partial_reflection_invariant,
)
from eqnn.types import ComplexArray, IntArray, RealArray
from eqnn.utils.timing import RuntimeProfile, timed


@dataclass(frozen=True)
class HeisenbergDatasetConfig:
    """Configuration for a small exact-diagonalization dataset."""

    num_qubits: int
    ratio_min: float = 0.4
    ratio_max: float = 1.6
    num_points: int = 31
    train_fraction: float = 0.8
    critical_ratio: float = 1.0
    exclusion_window: float = 0.05
    boundary: str = "open"
    eigensolver: str = "auto"
    labeling_strategy: str = "ratio_threshold"
    diagnostic_window: float = 0.05
    partial_reflection_pairs: int | None = None
    split_seed: int = 0

    def __post_init__(self) -> None:
        if self.num_qubits < 2:
            raise ValueError("num_qubits must be at least 2")
        if self.ratio_min >= self.ratio_max:
            raise ValueError("ratio_min must be smaller than ratio_max")
        if self.num_points < 3:
            raise ValueError("num_points must be at least 3")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must lie in (0, 1)")
        if self.exclusion_window < 0.0:
            raise ValueError("exclusion_window must be non-negative")
        if self.boundary not in {"open", "periodic"}:
            raise ValueError("boundary must be 'open' or 'periodic'")
        if self.eigensolver not in {"auto", "dense", "sparse"}:
            raise ValueError("eigensolver must be 'auto', 'dense', or 'sparse'")
        if self.labeling_strategy not in {"ratio_threshold", "partial_reflection"}:
            raise ValueError("labeling_strategy must be 'ratio_threshold' or 'partial_reflection'")
        if self.diagnostic_window < 0.0:
            raise ValueError("diagnostic_window must be non-negative")
        if self.partial_reflection_pairs is not None and self.partial_reflection_pairs < 1:
            raise ValueError("partial_reflection_pairs must be positive when provided")


@dataclass(frozen=True)
class DatasetSplit:
    """A dataset split storing statevectors and supervised labels."""

    states: ComplexArray
    labels: IntArray
    coupling_ratios: RealArray
    ground_state_energies: RealArray
    diagnostics: dict[str, RealArray] | None = None

    def __post_init__(self) -> None:
        num_examples = self.states.shape[0]
        if self.states.ndim != 2:
            raise ValueError("states must have shape (num_examples, hilbert_dimension)")
        if self.labels.shape != (num_examples,):
            raise ValueError("labels must align with states")
        if self.coupling_ratios.shape != (num_examples,):
            raise ValueError("coupling_ratios must align with states")
        if self.ground_state_energies.shape != (num_examples,):
            raise ValueError("ground_state_energies must align with states")
        if self.diagnostics is not None:
            for name, values in self.diagnostics.items():
                if values.shape != (num_examples,):
                    raise ValueError(f"diagnostic '{name}' must align with states")

    def __len__(self) -> int:
        return int(self.labels.shape[0])


@dataclass(frozen=True)
class DatasetBundle:
    """Train/test bundle plus dataset metadata."""

    train: DatasetSplit
    test: DatasetSplit
    metadata: dict[str, Any]


def phase_label_from_ratio(
    coupling_ratio: float,
    *,
    critical_ratio: float,
    exclusion_window: float,
) -> int:
    """Map a coupling ratio to a binary phase label."""

    if abs(coupling_ratio - critical_ratio) <= exclusion_window:
        raise ValueError(
            "Coupling ratio lies inside the excluded window around the critical point."
        )
    return 0 if coupling_ratio < critical_ratio else 1


def phase_label_from_partial_reflection(
    calibrated_score: float,
    *,
    diagnostic_window: float,
) -> int:
    """Map a calibrated partial-reflection score to a binary phase label."""

    if abs(calibrated_score) <= diagnostic_window:
        raise ValueError(
            "Partial-reflection score lies inside the excluded window around the phase boundary."
        )
    return 1 if calibrated_score > 0.0 else 0


def sample_coupling_ratios(config: HeisenbergDatasetConfig) -> RealArray:
    """Construct a reproducible grid of coupling ratios away from the transition."""

    ratios = np.linspace(
        config.ratio_min,
        config.ratio_max,
        num=config.num_points,
        dtype=np.float64,
    )
    mask = np.abs(ratios - config.critical_ratio) > config.exclusion_window
    filtered = ratios[mask]
    if filtered.size < 2:
        raise ValueError("No usable coupling ratios remain after applying the exclusion window.")
    return filtered


def generate_dataset(
    config: HeisenbergDatasetConfig,
    *,
    profile: RuntimeProfile | None = None,
) -> DatasetBundle:
    """Generate train and test splits from exact ground states."""
    with timed(profile, "dataset.total"):
        hamiltonian = BondAlternatingHeisenbergHamiltonian(
            num_qubits=config.num_qubits,
            boundary=config.boundary,
        )

        with timed(profile, "dataset.ratio_grid"):
            ratios = sample_coupling_ratios(config)

        states = np.zeros((ratios.size, hamiltonian.dimension), dtype=np.complex128)
        labels = np.zeros(ratios.size, dtype=np.int64)
        energies = np.zeros(ratios.size, dtype=np.float64)

        primary_singlet_means = np.zeros(ratios.size, dtype=np.float64)
        secondary_singlet_means = np.zeros(ratios.size, dtype=np.float64)
        dimerization_features = np.zeros(ratios.size, dtype=np.float64)
        partial_reflection_real = np.zeros(ratios.size, dtype=np.float64)
        partial_reflection_imag = np.zeros(ratios.size, dtype=np.float64)
        partial_reflection_calibrated = np.full(ratios.size, np.nan, dtype=np.float64)

        resolved_eigensolver = hamiltonian.resolve_ground_state_method(config.eigensolver)
        reflected_pairs = _resolve_partial_reflection_pairs(config)

        for index, ratio in enumerate(ratios):
            with timed(profile, "dataset.ground_state"):
                energy, state = hamiltonian.ground_state(float(ratio), method=config.eigensolver)

            states[index] = state
            energies[index] = energy

            with timed(profile, "dataset.diagnostics"):
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

                partial_reflection = normalized_partial_reflection_invariant(
                    state,
                    config.num_qubits,
                    num_reflected_pairs=reflected_pairs,
                )
                partial_reflection_real[index] = float(np.real(partial_reflection))
                partial_reflection_imag[index] = float(np.imag(partial_reflection))

                if config.num_qubits % 2 == 0:
                    partial_reflection_calibrated[index] = calibrated_partial_reflection_score(
                        state,
                        config.num_qubits,
                        num_reflected_pairs=reflected_pairs,
                    )

            with timed(profile, "dataset.label_assignment"):
                labels[index] = _phase_label_for_sample(
                    float(ratio),
                    config=config,
                    calibrated_partial_reflection_score_value=partial_reflection_calibrated[index],
                )

        diagnostics = {
            "primary_singlet_mean": primary_singlet_means,
            "secondary_singlet_mean": secondary_singlet_means,
            "dimerization_feature": dimerization_features,
            "partial_reflection_real": partial_reflection_real,
            "partial_reflection_imag": partial_reflection_imag,
            "partial_reflection_calibrated": partial_reflection_calibrated,
        }

        with timed(profile, "dataset.split"):
            train_indices, test_indices = _stratified_split_indices(
                labels=labels,
                train_fraction=config.train_fraction,
                seed=config.split_seed,
            )

        with timed(profile, "dataset.pack"):
            train = DatasetSplit(
                states=states[train_indices],
                labels=labels[train_indices],
                coupling_ratios=ratios[train_indices],
                ground_state_energies=energies[train_indices],
                diagnostics=_slice_diagnostics(diagnostics, train_indices),
            )
            test = DatasetSplit(
                states=states[test_indices],
                labels=labels[test_indices],
                coupling_ratios=ratios[test_indices],
                ground_state_energies=energies[test_indices],
                diagnostics=_slice_diagnostics(diagnostics, test_indices),
            )

        metadata = {
            "dataset_name": "bond_alternating_heisenberg",
            "num_qubits": int(config.num_qubits),
            "hilbert_dimension": int(hamiltonian.dimension),
            "boundary": config.boundary,
            "eigensolver_requested": config.eigensolver,
            "eigensolver_resolved": resolved_eigensolver,
            "labeling_strategy": config.labeling_strategy,
            "ratio_min": float(config.ratio_min),
            "ratio_max": float(config.ratio_max),
            "num_requested_points": int(config.num_points),
            "num_total_samples": int(ratios.size),
            "train_fraction": float(config.train_fraction),
            "critical_ratio": float(config.critical_ratio),
            "exclusion_window": float(config.exclusion_window),
            "diagnostic_window": float(config.diagnostic_window),
            "partial_reflection_pairs": int(reflected_pairs),
            "split_seed": int(config.split_seed),
            "train_size": len(train),
            "test_size": len(test),
            "train_label_histogram": _label_histogram(train.labels),
            "test_label_histogram": _label_histogram(test.labels),
            "phase_labels": {
                "0": "coupling_ratio < critical_ratio",
                "1": "coupling_ratio > critical_ratio",
            },
            "assumptions": [
                "The bond between sites 0 and 1 has coupling 1.0.",
                "The bond between sites 1 and 2 has coupling r, and the pattern alternates.",
                "Samples within exclusion_window of the critical ratio are omitted.",
            ],
            "diagnostics": {
                "primary_singlet_mean": _summary_statistics(primary_singlet_means),
                "secondary_singlet_mean": _summary_statistics(secondary_singlet_means),
                "dimerization_feature": _summary_statistics(dimerization_features),
                "partial_reflection_real": _summary_statistics(partial_reflection_real),
                "partial_reflection_imag": _summary_statistics(partial_reflection_imag),
                "partial_reflection_calibrated": _summary_statistics(
                    partial_reflection_calibrated[~np.isnan(partial_reflection_calibrated)]
                ),
            },
            "reference_phase_labels": _reference_phase_label_summary(
                ratios=ratios,
                partial_reflection_scores=partial_reflection_calibrated,
                critical_ratio=config.critical_ratio,
                exclusion_window=config.exclusion_window,
                diagnostic_window=config.diagnostic_window,
                supports_partial_reflection=(config.num_qubits % 2 == 0),
            ),
        }

        return DatasetBundle(train=train, test=test, metadata=metadata)


def _resolve_partial_reflection_pairs(config: HeisenbergDatasetConfig) -> int:
    if config.partial_reflection_pairs is not None:
        return int(config.partial_reflection_pairs)
    return default_partial_reflection_pairs(config.num_qubits)


def _phase_label_for_sample(
    coupling_ratio: float,
    *,
    config: HeisenbergDatasetConfig,
    calibrated_partial_reflection_score_value: float,
) -> int:
    if config.labeling_strategy == "ratio_threshold":
        return phase_label_from_ratio(
            coupling_ratio,
            critical_ratio=config.critical_ratio,
            exclusion_window=config.exclusion_window,
        )

    if config.num_qubits % 2 != 0:
        raise ValueError(
            "partial_reflection labeling is only supported for even num_qubits in the current finite-size implementation"
        )
    return phase_label_from_partial_reflection(
        calibrated_partial_reflection_score_value,
        diagnostic_window=config.diagnostic_window,
    )


def _slice_diagnostics(
    diagnostics: dict[str, RealArray],
    indices: IntArray,
) -> dict[str, RealArray]:
    return {
        name: np.asarray(values[indices], dtype=np.float64)
        for name, values in diagnostics.items()
    }


def _stratified_split_indices(
    *,
    labels: IntArray,
    train_fraction: float,
    seed: int,
) -> tuple[IntArray, IntArray]:
    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []

    for label in sorted(np.unique(labels).tolist()):
        label_indices = np.flatnonzero(labels == label)
        rng.shuffle(label_indices)

        if label_indices.size == 1:
            train_count = 1 if train_fraction >= 0.5 else 0
        else:
            train_count = int(round(train_fraction * label_indices.size))
            train_count = min(max(train_count, 1), label_indices.size - 1)

        train_indices.append(label_indices[:train_count])
        test_indices.append(label_indices[train_count:])

    train = np.concatenate(train_indices).astype(np.int64, copy=False)
    test = np.concatenate(test_indices).astype(np.int64, copy=False)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _label_histogram(labels: IntArray) -> dict[str, int]:
    counts = np.bincount(labels, minlength=2)
    return {str(index): int(count) for index, count in enumerate(counts.tolist())}


def _summary_statistics(values: RealArray) -> dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _reference_phase_label_summary(
    *,
    ratios: RealArray,
    partial_reflection_scores: RealArray,
    critical_ratio: float,
    exclusion_window: float,
    diagnostic_window: float,
    supports_partial_reflection: bool,
) -> dict[str, Any]:
    ratio_labels = np.asarray(
        [
            phase_label_from_ratio(
                float(ratio),
                critical_ratio=critical_ratio,
                exclusion_window=exclusion_window,
            )
            for ratio in ratios
        ],
        dtype=np.int64,
    )
    summary: dict[str, Any] = {
        "ratio_threshold": {
            "label_histogram": _label_histogram(ratio_labels),
        }
    }

    if not supports_partial_reflection:
        summary["partial_reflection"] = {
            "available": False,
            "reason": "finite-size partial-reflection calibration is only enabled for even num_qubits",
        }
        return summary

    valid_mask = np.abs(partial_reflection_scores) > diagnostic_window
    partial_labels = np.asarray(
        [
            phase_label_from_partial_reflection(
                float(score),
                diagnostic_window=diagnostic_window,
            )
            for score in partial_reflection_scores[valid_mask]
        ],
        dtype=np.int64,
    )
    ratio_on_valid = ratio_labels[valid_mask]
    agreement = float(np.mean(partial_labels == ratio_on_valid)) if partial_labels.size else 0.0
    summary["partial_reflection"] = {
        "available": True,
        "label_histogram": _label_histogram(partial_labels) if partial_labels.size else {"0": 0, "1": 0},
        "num_excluded_by_diagnostic_window": int(np.size(valid_mask) - np.count_nonzero(valid_mask)),
        "agreement_with_ratio_threshold_on_valid_samples": agreement,
    }
    return summary
