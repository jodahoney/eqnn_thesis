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


def qubit_permutation_operator(
    num_qubits: int,
    permutation: Iterable[int],
) -> ComplexArray:
    """Return the basis permutation that reorders qubits.

    The returned operator maps

        |b_0 ... b_{n-1}> -> |b_{permutation[0]} ... b_{permutation[n-1]}>.
    """

    permutation_tuple = tuple(int(index) for index in permutation)
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1")
    if sorted(permutation_tuple) != list(range(num_qubits)):
        raise ValueError("permutation must be a rearrangement of range(num_qubits)")

    dimension = 1 << num_qubits
    operator = np.zeros((dimension, dimension), dtype=np.complex128)
    for input_index in range(dimension):
        input_bits = _index_to_bits(input_index, num_qubits)
        output_bits = [input_bits[site] for site in permutation_tuple]
        output_index = _bits_to_index(output_bits)
        operator[output_index, input_index] = 1.0
    return operator


def embed_operator_on_sites(
    operator: ComplexArray,
    num_qubits: int,
    sites: Iterable[int],
) -> ComplexArray:
    """Embed a k-qubit operator on selected sites of an n-qubit Hilbert space."""

    site_tuple = tuple(sorted(set(int(site) for site in sites)))
    if not site_tuple:
        raise ValueError("At least one site is required")
    if num_qubits < len(site_tuple):
        raise ValueError("num_qubits must be at least the number of embedded sites")
    for site in site_tuple:
        if site < 0 or site >= num_qubits:
            raise ValueError(f"site index {site} is out of range for {num_qubits} qubits")

    local_dimension = 1 << len(site_tuple)
    if operator.shape != (local_dimension, local_dimension):
        raise ValueError(
            "operator shape does not match the number of requested sites: "
            f"expected {(local_dimension, local_dimension)}, got {operator.shape}"
        )

    remainder_sites = tuple(site for site in range(num_qubits) if site not in site_tuple)
    permutation = site_tuple + remainder_sites
    permutation_operator = qubit_permutation_operator(num_qubits, permutation)
    embedded_front = np.kron(
        np.asarray(operator, dtype=np.complex128),
        np.eye(1 << len(remainder_sites), dtype=np.complex128),
    )
    return np.asarray(
        permutation_operator.conjugate().T @ embedded_front @ permutation_operator,
        dtype=np.complex128,
    )


def partial_trace_adjoint(
    reduced_operator: ComplexArray,
    num_qubits: int,
    traced_out_sites: Iterable[int],
) -> ComplexArray:
    """Return the adjoint map of partial trace acting on an observable."""

    traced_out = tuple(sorted(set(int(site) for site in traced_out_sites)))
    kept_sites = tuple(site for site in range(num_qubits) if site not in traced_out)
    return embed_operator_on_sites(reduced_operator, num_qubits, kept_sites)


def _index_to_bits(index: int, num_qubits: int) -> list[int]:
    return [int((index >> (num_qubits - 1 - site)) & 1) for site in range(num_qubits)]


def _bits_to_index(bits: Iterable[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value
