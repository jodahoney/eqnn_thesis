from __future__ import annotations

import unittest

import numpy as np

from eqnn.physics.heisenberg import BondAlternatingHeisenbergHamiltonian
from eqnn.physics.topology import (
    calibrated_partial_reflection_score,
    central_reflection_sites,
    default_partial_reflection_pairs,
    normalized_partial_reflection_invariant,
    reflection_permutation_operator,
)


class PartialReflectionTopologyTests(unittest.TestCase):
    def test_reflection_operator_is_unitary_and_an_involution(self) -> None:
        operator = reflection_permutation_operator(4)
        np.testing.assert_allclose(operator.conjugate().T @ operator, np.eye(16), atol=1e-12)
        np.testing.assert_allclose(operator @ operator, np.eye(16), atol=1e-12)

    def test_central_reflection_sites_cover_the_expected_windows(self) -> None:
        self.assertEqual(central_reflection_sites(6, 2), (1, 2, 3, 4))
        self.assertEqual(central_reflection_sites(7, 2), (1, 2, 4, 5))

    def test_default_partial_reflection_pairs_match_small_chain_convention(self) -> None:
        self.assertEqual(default_partial_reflection_pairs(4), 1)
        self.assertEqual(default_partial_reflection_pairs(6), 2)
        self.assertEqual(default_partial_reflection_pairs(8), 3)

    def test_even_chain_calibrated_partial_reflection_separates_the_two_regimes(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=6)
        _, trivial_state = hamiltonian.ground_state(0.6)
        _, topological_state = hamiltonian.ground_state(1.4)

        trivial_score = calibrated_partial_reflection_score(trivial_state, num_qubits=6)
        topological_score = calibrated_partial_reflection_score(topological_state, num_qubits=6)

        self.assertLess(trivial_score, -0.05)
        self.assertGreater(topological_score, 0.05)

    def test_partial_reflection_invariant_is_real_for_the_even_chain_benchmark(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=6)
        _, state = hamiltonian.ground_state(1.2)
        invariant = normalized_partial_reflection_invariant(state, num_qubits=6)
        self.assertAlmostEqual(float(np.imag(invariant)), 0.0, places=10)

    def test_calibrated_score_rejects_odd_length_chains(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=5)
        _, state = hamiltonian.ground_state(1.2)
        with self.assertRaises(ValueError):
            calibrated_partial_reflection_score(state, num_qubits=5)
