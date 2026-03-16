from __future__ import annotations

import unittest

from eqnn.models.qcnn import QCNNConfig, SU2QCNN
from eqnn.physics.heisenberg import BondAlternatingHeisenbergHamiltonian
from eqnn.verification.equivariance import model_invariance_error


class SU2QCNNTests(unittest.TestCase):
    def test_forward_output_is_probabilistic_and_interpretable(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=4)
        _, state = hamiltonian.ground_state(0.6)
        model = SU2QCNN(
            QCNNConfig(num_qubits=4, min_readout_qubits=4),
            parameters=(0.2, -0.15, 2.0, 0.1),
        )

        forward = model.forward(state)

        self.assertGreaterEqual(forward.probability, 0.0)
        self.assertLessEqual(forward.probability, 1.0)
        self.assertEqual(forward.final_num_qubits, 4)

    def test_readout_feature_tracks_bond_dimerization(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=4)
        _, low_ratio_state = hamiltonian.ground_state(0.4)
        _, high_ratio_state = hamiltonian.ground_state(1.6)
        model = SU2QCNN(QCNNConfig(num_qubits=4, min_readout_qubits=4))

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
            QCNNConfig(num_qubits=4, min_readout_qubits=4),
            parameters=(0.27, -0.19, 3.0, -0.2),
        )
        summary = model_invariance_error(model, num_trials=5, seed=19)
        self.assertLess(summary["max_error"], 1e-10)
