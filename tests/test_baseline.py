from __future__ import annotations

import unittest

import numpy as np

from eqnn.layers import AnisotropicConvolution, AnisotropicConvolutionConfig
from eqnn.models import BaselineQCNN, BaselineQCNNConfig
from eqnn.physics.heisenberg import BondAlternatingHeisenbergHamiltonian
from eqnn.verification.equivariance import model_invariance_error


class BaselineLayerTests(unittest.TestCase):
    def test_anisotropic_convolution_gate_is_unitary(self) -> None:
        gate = AnisotropicConvolution.gate(0.2, -0.3, 0.1)
        identity = np.eye(4, dtype=np.complex128)
        np.testing.assert_allclose(gate.conjugate().T @ gate, identity, atol=1e-12)

    def test_anisotropic_convolution_parameter_count_tracks_active_parities(self) -> None:
        layer = AnisotropicConvolution(AnisotropicConvolutionConfig(num_qubits=4))
        self.assertEqual(layer.parameter_count, 6)


class BaselineQCNNTests(unittest.TestCase):
    def test_baseline_forward_output_is_probabilistic(self) -> None:
        hamiltonian = BondAlternatingHeisenbergHamiltonian(num_qubits=4)
        _, state = hamiltonian.ground_state(0.6)
        model = BaselineQCNN(
            BaselineQCNNConfig(num_qubits=4),
            parameters=(0.2, -0.4, 0.1, 0.3, 0.05, -0.2, -0.1, 0.25, 0.15),
        )

        forward = model.forward(state)

        self.assertGreaterEqual(forward.probability, 0.0)
        self.assertLessEqual(forward.probability, 1.0)
        self.assertEqual(forward.final_num_qubits, 2)

    def test_baseline_model_breaks_su2_invariance(self) -> None:
        model = BaselineQCNN(
            BaselineQCNNConfig(num_qubits=4),
            parameters=(0.2, -0.4, 0.1, 0.3, 0.05, -0.2, -0.1, 0.25, 0.15),
        )

        summary = model_invariance_error(model, num_trials=5, seed=7)

        self.assertGreater(summary["max_error"], 1e-3)
