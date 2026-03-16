"""General-purpose quantum state and density-matrix helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from eqnn.types import ComplexArray


def is_density_matrix(state: ComplexArray) -> bool:
    """Return True when the input looks like a square density matrix."""

    return state.ndim == 2 and state.shape[0] == state.shape[1]


def statevector_to_density_matrix(state: ComplexArray) -> ComplexArray:
    """Convert a normalized statevector into a rank-1 density matrix."""

    if state.ndim != 1:
        raise ValueError("statevector_to_density_matrix expects a one-dimensional statevector")
    return np.outer(state, np.conjugate(state))


def as_density_matrix(state: ComplexArray) -> ComplexArray:
    """Accept either a statevector or density matrix and return a density matrix."""

    if state.ndim == 1:
        return statevector_to_density_matrix(state)
    if is_density_matrix(state):
        return np.asarray(state, dtype=np.complex128)
    raise ValueError("State must be either a statevector or a square density matrix")


def partial_trace_density_matrix(
    density_matrix: ComplexArray,
    num_qubits: int,
    traced_out_sites: Iterable[int],
) -> ComplexArray:
    """Trace out a subset of qubits from an n-qubit density matrix."""

    traced_out = sorted(set(int(site) for site in traced_out_sites))
    for site in traced_out:
        if site < 0 or site >= num_qubits:
            raise ValueError(f"site index {site} is out of range for {num_qubits} qubits")

    expected_dimension = 1 << num_qubits
    if density_matrix.shape != (expected_dimension, expected_dimension):
        raise ValueError(
            "density_matrix shape does not match the provided num_qubits: "
            f"expected {(expected_dimension, expected_dimension)}, got {density_matrix.shape}"
        )

    tensor = density_matrix.reshape((2,) * num_qubits * 2)
    current_num_qubits = num_qubits
    for site in reversed(traced_out):
        tensor = np.trace(
            tensor,
            axis1=site,
            axis2=site + current_num_qubits,
        )
        current_num_qubits -= 1

    reduced_dimension = 1 << current_num_qubits
    return np.asarray(tensor.reshape(reduced_dimension, reduced_dimension), dtype=np.complex128)


def reduced_density_matrix(
    density_matrix: ComplexArray,
    num_qubits: int,
    kept_sites: Iterable[int],
) -> ComplexArray:
    """Return the reduced density matrix on the requested sites."""

    kept = sorted(set(int(site) for site in kept_sites))
    if not kept:
        raise ValueError("At least one kept site is required")

    traced_out = [site for site in range(num_qubits) if site not in kept]
    return partial_trace_density_matrix(density_matrix, num_qubits, traced_out)


def expectation_value(density_matrix: ComplexArray, operator: ComplexArray) -> float:
    """Compute the real expectation value of an observable."""

    value = np.trace(density_matrix @ operator)
    return float(np.real_if_close(value))
