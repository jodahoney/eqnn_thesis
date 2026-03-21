from __future__ import annotations

import unittest

import numpy as np

from eqnn.models.qcnn import QCNNConfig, SU2QCNN
from eqnn.physics.heisenberg import BondAlternatingHeisenbergHamiltonian
from eqnn.physics.observables import SINGLET_STATE
from eqnn.verification.equivariance import model_invariance_error


class SU2QCNNTests(unittest.TestCase):
    def test_forward_output_is_probabilistic_and_interpretable(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=4)
        _, state = hamiltonian.ground_state(0.6)
        model = SU2QCNN(
            QCNNConfig(num_qubits=4),
            parameters=(0.2, -0.15, 0.1),
        )

        forward = model.forward(state)

        self.assertGreaterEqual(forward.probability, 0.0)
        self.assertLessEqual(forward.probability, 1.0)
        self.assertEqual(forward.final_num_qubits, 2)
        self.assertEqual(forward.readout_mode, "swap")
        self.assertIsNotNone(forward.swap_expectation)
        self.assertIsNone(forward.logit)
        self.assertAlmostEqual(
            forward.probability,
            0.5 * (float(forward.swap_expectation) + 1.0),
        )

    def test_swap_readout_matches_known_two_qubit_limits(self) -> None:
        singlet_model = SU2QCNN(QCNNConfig(num_qubits=2))
        triplet_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

        singlet_forward = singlet_model.forward(SINGLET_STATE)
        triplet_forward = singlet_model.forward(triplet_state)

        self.assertAlmostEqual(float(singlet_forward.swap_expectation), -1.0, places=12)
        self.assertAlmostEqual(singlet_forward.probability, 0.0, places=12)
        self.assertAlmostEqual(float(triplet_forward.swap_expectation), 1.0, places=12)
        self.assertAlmostEqual(triplet_forward.probability, 1.0, places=12)

    def test_swap_readout_requires_two_terminal_qubits(self) -> None:
        with self.assertRaises(ValueError):
            QCNNConfig(num_qubits=4, min_readout_qubits=4, readout_mode="swap")

    def test_swap_readout_tracks_small_chain_coupling_ordering(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=4)
        _, low_ratio_state = hamiltonian.ground_state(0.4)
        _, high_ratio_state = hamiltonian.ground_state(1.6)
        model = SU2QCNN(QCNNConfig(num_qubits=4))

        low_forward = model.forward(low_ratio_state)
        high_forward = model.forward(high_ratio_state)

        self.assertLess(low_forward.swap_expectation, high_forward.swap_expectation)
        self.assertLess(low_forward.probability, high_forward.probability)

    def test_legacy_dimerization_readout_tracks_bond_dimerization(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=4)
        _, low_ratio_state = hamiltonian.ground_state(0.4)
        _, high_ratio_state = hamiltonian.ground_state(1.6)
        model = SU2QCNN(
            QCNNConfig(
                num_qubits=4,
                min_readout_qubits=4,
                readout_mode="dimerization",
            )
        )

        low_forward = model.forward(low_ratio_state)
        high_forward = model.forward(high_ratio_state)

        self.assertGreater(low_forward.primary_singlet_mean, low_forward.secondary_singlet_mean)
        self.assertGreater(high_forward.primary_singlet_mean, high_forward.secondary_singlet_mean)
        self.assertLess(low_forward.dimerization_feature, high_forward.dimerization_feature)
        self.assertGreater(
            high_forward.secondary_singlet_mean - low_forward.secondary_singlet_mean,
            0.25,
        )

    def test_model_prediction_is_su2_invariant(self) -> None:
        model = SU2QCNN(
            QCNNConfig(num_qubits=4),
            parameters=(0.27, -0.19, 0.11),
        )
        summary = model_invariance_error(model, num_trials=5, seed=19)
        self.assertLess(summary["max_error"], 1e-10)

    def test_equivariant_pooling_model_matches_partial_trace_limit(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=6)
        _, state = hamiltonian.ground_state(1.1)
        partial_trace_model = SU2QCNN(
            QCNNConfig(num_qubits=6, min_readout_qubits=2, pooling_mode="partial_trace"),
            parameters=(0.1, -0.2, 0.05, 0.3, -0.17),
        )
        equivariant_model = SU2QCNN(
            QCNNConfig(num_qubits=6, min_readout_qubits=2, pooling_mode="equivariant"),
            parameters=(0.1, -0.2, 0.05, 0.3, -0.17, -20.0, 20.0, 0.0, -20.0, 20.0, 0.0),
        )

        partial_forward = partial_trace_model.forward(state)
        equivariant_forward = equivariant_model.forward(state)

        np.testing.assert_allclose(
            equivariant_forward.final_density_matrix,
            partial_forward.final_density_matrix,
            atol=1e-6,
        )
        self.assertAlmostEqual(equivariant_forward.probability, partial_forward.probability, places=6)

    def test_equivariant_pooling_model_is_su2_invariant(self) -> None:
        model = SU2QCNN(
            QCNNConfig(num_qubits=6, min_readout_qubits=2, pooling_mode="equivariant"),
            parameters=(0.1, -0.2, 0.05, 0.3, -0.17, 0.7, -0.1, 0.2, -0.3, 0.4, -0.2),
        )
        summary = model_invariance_error(model, num_trials=5, seed=27)
        self.assertLess(summary["max_error"], 1e-10)

    def test_equivariant_pooling_preserves_the_dataset_dimerization_ordering(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=6)
        _, low_ratio_state = hamiltonian.ground_state(0.5)
        _, high_ratio_state = hamiltonian.ground_state(1.5)
        model = SU2QCNN(
            QCNNConfig(
                num_qubits=6,
                min_readout_qubits=4,
                pooling_mode="equivariant",
                readout_mode="dimerization",
            )
        )

        low_forward = model.forward(low_ratio_state)
        high_forward = model.forward(high_ratio_state)

        self.assertLess(low_forward.dimerization_feature, high_forward.dimerization_feature)
