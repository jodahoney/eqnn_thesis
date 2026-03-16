from __future__ import annotations

import unittest

import numpy as np

from eqnn.physics.heisenberg import BondAlternatingHeisenbergHamiltonian


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
