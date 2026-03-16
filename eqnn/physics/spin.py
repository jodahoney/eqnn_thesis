"""Spin-1/2 operator helpers."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from eqnn.types import ComplexArray

IDENTITY = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def kron_all(operators: list[ComplexArray]) -> ComplexArray:
    """Kronecker-product a list of local operators in site order."""

    result = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    for operator in operators:
        result = np.kron(result, operator)
    return np.asarray(result, dtype=np.complex128)


def embed_local_operators(
    num_qubits: int,
    site_operators: Mapping[int, ComplexArray],
) -> ComplexArray:
    """Embed single-site operators into an n-qubit Hilbert space."""

    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1")
    for site in site_operators:
        if site < 0 or site >= num_qubits:
            raise ValueError(f"site index {site} is out of range for {num_qubits} qubits")

    operators = [site_operators.get(site, IDENTITY) for site in range(num_qubits)]
    return kron_all(operators)
