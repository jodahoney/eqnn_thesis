"""Topological diagnostics for bond-alternating spin chains."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from eqnn.physics.quantum import as_density_matrix, reduced_density_matrix
from eqnn.types import ComplexArray


def default_partial_reflection_pairs(num_qubits: int) -> int:
    """Choose a central reflected region size for small exact-diagonalization studies."""

    if num_qubits < 2:
        raise ValueError("num_qubits must be at least 2")
    if num_qubits <= 3:
        return 1
    if num_qubits % 2 == 0:
        return max(1, num_qubits // 2 - 1)
    return max(1, num_qubits // 2 - 1)


def central_reflection_sites(
    num_qubits: int,
    num_reflected_pairs: int,
) -> tuple[int, ...]:
    """Return the central sites used in the partial-reflection diagnostic."""

    if num_qubits < 2:
        raise ValueError("num_qubits must be at least 2")
    if num_reflected_pairs < 1:
        raise ValueError("num_reflected_pairs must be at least 1")
    if num_reflected_pairs > num_qubits // 2:
        raise ValueError("num_reflected_pairs cannot exceed floor(num_qubits / 2)")

    if num_qubits % 2 == 0:
        start = num_qubits // 2 - num_reflected_pairs
        return tuple(range(start, start + 2 * num_reflected_pairs))

    center_site = num_qubits // 2
    left_start = center_site - num_reflected_pairs
    left_sites = tuple(range(left_start, center_site))
    right_sites = tuple(range(center_site + 1, center_site + 1 + num_reflected_pairs))
    return left_sites + right_sites


@lru_cache(maxsize=None)
def reflection_permutation_operator(num_sites: int) -> ComplexArray:
    """Return the operator that reverses the order of num_sites qubits."""

    if num_sites < 1:
        raise ValueError("num_sites must be at least 1")

    dimension = 1 << num_sites
    permutation = np.zeros((dimension, dimension), dtype=np.complex128)
    for basis_index in range(dimension):
        bits = tuple((basis_index >> shift) & 1 for shift in reversed(range(num_sites)))
        reflected_index = 0
        for bit in reversed(bits):
            reflected_index = (reflected_index << 1) | bit
        permutation[reflected_index, basis_index] = 1.0
    return np.asarray(permutation, dtype=np.complex128)


def normalized_partial_reflection_invariant(
    state: ComplexArray,
    num_qubits: int,
    num_reflected_pairs: int | None = None,
) -> complex:
    """Compute the normalized partial-reflection many-body invariant."""

    reflected_pairs = (
        default_partial_reflection_pairs(num_qubits)
        if num_reflected_pairs is None
        else int(num_reflected_pairs)
    )
    kept_sites = central_reflection_sites(num_qubits, reflected_pairs)
    local_num_sites = len(kept_sites)

    density_matrix = as_density_matrix(state)
    reduced = reduced_density_matrix(density_matrix, num_qubits, kept_sites)
    reflection = reflection_permutation_operator(local_num_sites)

    left_sites = tuple(range(reflected_pairs))
    right_sites = tuple(range(reflected_pairs, local_num_sites))
    reduced_left = reduced_density_matrix(reduced, local_num_sites, left_sites)
    reduced_right = reduced_density_matrix(reduced, local_num_sites, right_sites)

    normalization = np.sqrt(
        (
            float(np.real(np.trace(reduced_left @ reduced_left)))
            + float(np.real(np.trace(reduced_right @ reduced_right)))
        )
        / 2.0
    )
    if normalization <= 0.0:
        raise ValueError("partial-reflection normalization vanished")

    value = np.trace(reduced @ reflection) / normalization
    return complex(value)


def calibrated_partial_reflection_score(
    state: ComplexArray,
    num_qubits: int,
    num_reflected_pairs: int | None = None,
) -> float:
    """Return a finite-size calibrated real score for even-length chains.

    The raw partial-reflection invariant carries an overall parity-dependent sign in
    these small even chains. Multiplying by (-1)^n_pairs yields a score that is
    negative in the trivial regime and positive in the topological regime for the
    benchmark systems used here.
    """

    if num_qubits % 2 != 0:
        raise ValueError(
            "calibrated_partial_reflection_score is only supported for even num_qubits"
        )

    reflected_pairs = (
        default_partial_reflection_pairs(num_qubits)
        if num_reflected_pairs is None
        else int(num_reflected_pairs)
    )
    invariant = normalized_partial_reflection_invariant(
        state,
        num_qubits,
        num_reflected_pairs=reflected_pairs,
    )
    return float(((-1) ** reflected_pairs) * np.real_if_close(invariant))
