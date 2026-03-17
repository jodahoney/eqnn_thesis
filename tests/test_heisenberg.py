from __future__ import annotations

import unittest

import numpy as np

from eqnn.physics.heisenberg import BondAlternatingHeisenbergHamiltonian
from eqnn.physics.spin import sparse

SCIPY_AVAILABLE = sparse is not None


class BondAlternatingHeisenbergHamiltonianTests(unittest.TestCase):
    def test_matrix_is_hermitian(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=4, boundary="open")
        matrix = hamiltonian.matrix(0.7)

        self.assertEqual(matrix.shape, (16, 16))
        np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-12)

    def test_ground_state_is_normalized_eigenvector(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=4, boundary="open")
        energy, state = hamiltonian.ground_state(1.3)

        self.assertAlmostEqual(float(np.linalg.norm(state)), 1.0, places=12)
        expectation = np.vdot(state, hamiltonian.matrix(1.3) @ state).real
        self.assertAlmostEqual(energy, float(expectation), places=12)

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy is required for sparse tests")
    def test_sparse_ground_state_matches_dense_ground_state(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=4, boundary="open")
        dense_energy, dense_state = hamiltonian.ground_state(1.1, method="dense")
        sparse_energy, sparse_state = hamiltonian.ground_state(1.1, method="sparse")

        self.assertAlmostEqual(dense_energy, sparse_energy, places=10)
        self.assertAlmostEqual(abs(np.vdot(dense_state, sparse_state)), 1.0, places=8)

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy is required for sparse tests")
    def test_auto_solver_prefers_sparse_for_larger_systems(self) -> None:
        small_hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=4, boundary="open")
        large_hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=7, boundary="open")

        self.assertEqual(small_hamiltonian.resolve_ground_state_method("auto"), "dense")
        self.assertEqual(large_hamiltonian.resolve_ground_state_method("auto"), "sparse")
