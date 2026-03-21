"""SU(2)-invariant observables used by the QCNN readout."""

from __future__ import annotations

import numpy as np

from eqnn.physics.heisenberg import alternating_bond_groups
from eqnn.physics.quantum import expectation_value, reduced_density_matrix
from eqnn.types import ComplexArray

SINGLET_STATE = np.array(
    [0.0, 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0],
    dtype=np.complex128,
)
SINGLET_PROJECTOR = np.outer(SINGLET_STATE, np.conjugate(SINGLET_STATE))
SWAP_OPERATOR = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.complex128,
)


def singlet_fraction(
    density_matrix: ComplexArray,
    num_qubits: int,
    bond: tuple[int, int],
) -> float:
    """Return the singlet probability on a two-qubit bond."""

    left_site, right_site = bond
    reduced = reduced_density_matrix(density_matrix, num_qubits, (left_site, right_site))
    return expectation_value(reduced, SINGLET_PROJECTOR)


def alternating_singlet_means(
    density_matrix: ComplexArray,
    num_qubits: int,
    boundary: str = "open",
) -> tuple[float, float]:
    """Compute mean singlet fractions on the two alternating bond families."""

    primary_bonds, secondary_bonds = alternating_bond_groups(num_qubits, boundary)

    primary_values = [singlet_fraction(density_matrix, num_qubits, bond) for bond in primary_bonds]
    secondary_values = [singlet_fraction(density_matrix, num_qubits, bond) for bond in secondary_bonds]

    primary_mean = float(np.mean(primary_values)) if primary_values else 0.0
    secondary_mean = float(np.mean(secondary_values)) if secondary_values else 0.0
    return primary_mean, secondary_mean


def dimerization_feature(
    density_matrix: ComplexArray,
    num_qubits: int,
    boundary: str = "open",
) -> float:
    """Return the alternating-bond singlet imbalance used for classification."""

    primary_mean, secondary_mean = alternating_singlet_means(
        density_matrix,
        num_qubits,
        boundary=boundary,
    )
    return secondary_mean - primary_mean


def swap_expectation(density_matrix: ComplexArray) -> float:
    """Return <SWAP> on a two-qubit density matrix."""

    if density_matrix.shape != (4, 4):
        raise ValueError("swap_expectation expects a two-qubit density matrix with shape (4, 4)")
    return expectation_value(density_matrix, SWAP_OPERATOR)


def swap_probability(density_matrix: ComplexArray) -> float:
    """Return the paper-style QCNN readout f = (<SWAP> + 1) / 2."""

    return 0.5 * (swap_expectation(density_matrix) + 1.0)
