"""Dataset generation for the bond-alternating Heisenberg benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from eqnn.physics.heisenberg import BondAlternatingHeisenbergHamiltonian
from eqnn.types import ComplexArray, IntArray, RealArray


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


@dataclass(frozen=True)
class DatasetSplit:
    """A dataset split storing statevectors and supervised labels."""

    states: ComplexArray
    labels: IntArray
    coupling_ratios: RealArray
    ground_state_energies: RealArray

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


def generate_dataset(config: HeisenbergDatasetConfig) -> DatasetBundle:
    """Generate train and test splits from exact ground states."""

    hamiltonian = BondAlternatingHeisenbergHamiltonian(
        num_qubits=config.num_qubits,
        boundary=config.boundary,
    )
    ratios = sample_coupling_ratios(config)

    states = np.zeros((ratios.size, hamiltonian.dimension), dtype=np.complex128)
    labels = np.zeros(ratios.size, dtype=np.int64)
    energies = np.zeros(ratios.size, dtype=np.float64)

    for index, ratio in enumerate(ratios):
        energy, state = hamiltonian.ground_state(float(ratio))
        states[index] = state
        labels[index] = phase_label_from_ratio(
            float(ratio),
            critical_ratio=config.critical_ratio,
            exclusion_window=config.exclusion_window,
        )
        energies[index] = energy

    train_indices, test_indices = _stratified_split_indices(
        labels=labels,
        train_fraction=config.train_fraction,
        seed=config.split_seed,
    )

    train = DatasetSplit(
        states=states[train_indices],
        labels=labels[train_indices],
        coupling_ratios=ratios[train_indices],
        ground_state_energies=energies[train_indices],
    )
    test = DatasetSplit(
        states=states[test_indices],
        labels=labels[test_indices],
        coupling_ratios=ratios[test_indices],
        ground_state_energies=energies[test_indices],
    )

    metadata = {
        "dataset_name": "bond_alternating_heisenberg",
        "num_qubits": int(config.num_qubits),
        "hilbert_dimension": int(hamiltonian.dimension),
        "boundary": config.boundary,
        "ratio_min": float(config.ratio_min),
        "ratio_max": float(config.ratio_max),
        "num_requested_points": int(config.num_points),
        "num_total_samples": int(ratios.size),
        "train_fraction": float(config.train_fraction),
        "critical_ratio": float(config.critical_ratio),
        "exclusion_window": float(config.exclusion_window),
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
    }
    return DatasetBundle(train=train, test=test, metadata=metadata)


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
