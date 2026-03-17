"""Bond-alternating Heisenberg Hamiltonian construction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from eqnn.physics.spin import (
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    embed_local_operators,
    embed_local_operators_sparse,
    sparse,
)
from eqnn.types import ComplexArray

if sparse is not None:
    from scipy.sparse.linalg import eigsh
else:  # pragma: no cover - exercised indirectly when scipy is unavailable.
    eigsh = None  # type: ignore[assignment]


def nearest_neighbor_bonds(num_qubits: int, boundary: str = "open") -> tuple[tuple[int, int], ...]:
    """List nearest-neighbor bonds for the requested boundary condition."""

    if num_qubits < 2:
        raise ValueError("num_qubits must be at least 2")
    if boundary not in {"open", "periodic"}:
        raise ValueError("boundary must be 'open' or 'periodic'")

    bonds = tuple((site, site + 1) for site in range(num_qubits - 1))
    if boundary == "periodic":
        bonds = bonds + ((num_qubits - 1, 0),)
    return bonds


def alternating_bond_groups(
    num_qubits: int,
    boundary: str = "open",
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Split bonds into the two alternating coupling families."""

    bonds = nearest_neighbor_bonds(num_qubits, boundary)
    return bonds[::2], bonds[1::2]


def heisenberg_exchange_term(num_qubits: int, left_site: int, right_site: int) -> ComplexArray:
    """Construct S_i . S_j for a given pair of qubits."""

    term = np.zeros((1 << num_qubits, 1 << num_qubits), dtype=np.complex128)
    for pauli in (PAULI_X, PAULI_Y, PAULI_Z):
        term += embed_local_operators(
            num_qubits,
            {
                left_site: pauli,
                right_site: pauli,
            },
        )
    return 0.25 * term


def heisenberg_exchange_term_sparse(num_qubits: int, left_site: int, right_site: int) -> Any:
    """Sparse version of S_i . S_j for a given pair of qubits."""

    if sparse is None:
        raise ImportError("scipy is required for sparse Heisenberg operators")

    pauli_x = sparse.csr_matrix(PAULI_X)
    pauli_y = sparse.csr_matrix(PAULI_Y)
    pauli_z = sparse.csr_matrix(PAULI_Z)

    term = sparse.csr_matrix((1 << num_qubits, 1 << num_qubits), dtype=np.complex128)
    for pauli in (pauli_x, pauli_y, pauli_z):
        term = term + embed_local_operators_sparse(
            num_qubits,
            {
                left_site: pauli,
                right_site: pauli,
            },
        )
    return 0.25 * term


def fix_global_phase(state: ComplexArray, atol: float = 1e-12) -> ComplexArray:
    """Choose a reproducible global phase for a statevector."""

    for amplitude in state:
        if abs(amplitude) > atol:
            return state * np.exp(-1.0j * np.angle(amplitude))
    return state


@dataclass(frozen=True)
class BondAlternatingHeisenbergHamiltonian:
    """Hamiltonian for small bond-alternating Heisenberg chains."""

    num_qubits: int
    boundary: str = "open"

    def __post_init__(self) -> None:
        if self.num_qubits < 2:
            raise ValueError("num_qubits must be at least 2")
        if self.boundary not in {"open", "periodic"}:
            raise ValueError("boundary must be 'open' or 'periodic'")

    @property
    def dimension(self) -> int:
        return 1 << self.num_qubits

    @cached_property
    def primary_bonds(self) -> tuple[tuple[int, int], ...]:
        return alternating_bond_groups(self.num_qubits, self.boundary)[0]

    @cached_property
    def secondary_bonds(self) -> tuple[tuple[int, int], ...]:
        return alternating_bond_groups(self.num_qubits, self.boundary)[1]

    @cached_property
    def primary_operator(self) -> ComplexArray:
        return self._sum_bond_terms(self.primary_bonds)

    @cached_property
    def secondary_operator(self) -> ComplexArray:
        return self._sum_bond_terms(self.secondary_bonds)

    @cached_property
    def primary_operator_sparse(self) -> Any:
        return self._sum_bond_terms_sparse(self.primary_bonds)

    @cached_property
    def secondary_operator_sparse(self) -> Any:
        return self._sum_bond_terms_sparse(self.secondary_bonds)

    def matrix(self, coupling_ratio: float) -> ComplexArray:
        """Return H(r) = H_primary + r H_secondary."""

        return self.primary_operator + float(coupling_ratio) * self.secondary_operator

    def sparse_matrix(self, coupling_ratio: float) -> Any:
        """Return a sparse H(r) when scipy is available."""

        if sparse is None:
            raise ImportError("scipy is required for sparse Hamiltonian construction")
        return self.primary_operator_sparse + float(coupling_ratio) * self.secondary_operator_sparse

    def resolve_ground_state_method(self, method: str = "auto") -> str:
        """Pick a concrete eigensolver strategy."""

        if method not in {"auto", "dense", "sparse"}:
            raise ValueError("method must be 'auto', 'dense', or 'sparse'")
        if method == "auto":
            if sparse is not None and self.num_qubits >= 7:
                return "sparse"
            return "dense"
        if method == "sparse" and sparse is None:
            raise ImportError("scipy is required for sparse ground-state solves")
        return method

    def ground_state(self, coupling_ratio: float, method: str = "auto") -> tuple[float, ComplexArray]:
        """Return the ground-state energy and normalized statevector."""

        resolved_method = self.resolve_ground_state_method(method)
        if resolved_method == "dense":
            eigenvalues, eigenvectors = np.linalg.eigh(self.matrix(coupling_ratio))
            ground_energy = float(np.real(eigenvalues[0]))
            ground_state = np.asarray(eigenvectors[:, 0], dtype=np.complex128)
        else:
            if eigsh is None:
                raise ImportError("scipy is required for sparse ground-state solves")
            eigenvalues, eigenvectors = eigsh(
                self.sparse_matrix(coupling_ratio),
                k=1,
                which="SA",
            )
            ground_energy = float(np.real(eigenvalues[0]))
            ground_state = np.asarray(eigenvectors[:, 0], dtype=np.complex128)

        ground_state = fix_global_phase(ground_state)
        ground_state = ground_state / np.linalg.norm(ground_state)
        return ground_energy, ground_state

    def _sum_bond_terms(self, bonds: tuple[tuple[int, int], ...]) -> ComplexArray:
        operator = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        for left_site, right_site in bonds:
            operator += heisenberg_exchange_term(self.num_qubits, left_site, right_site)
        return operator

    def _sum_bond_terms_sparse(self, bonds: tuple[tuple[int, int], ...]) -> Any:
        if sparse is None:
            raise ImportError("scipy is required for sparse Heisenberg operators")
        operator = sparse.csr_matrix((self.dimension, self.dimension), dtype=np.complex128)
        for left_site, right_site in bonds:
            operator = operator + heisenberg_exchange_term_sparse(
                self.num_qubits,
                left_site,
                right_site,
            )
        return operator
