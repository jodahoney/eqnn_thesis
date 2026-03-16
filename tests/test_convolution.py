from __future__ import annotations

import unittest

import numpy as np

from eqnn.layers.convolution import SU2SwapConvolution, SU2SwapConvolutionConfig
from eqnn.verification.equivariance import convolution_equivariance_error


class SU2SwapConvolutionTests(unittest.TestCase):
    def test_gate_is_unitary(self) -> None:
        gate = SU2SwapConvolution.gate(0.37)
        np.testing.assert_allclose(gate.conj().T @ gate, np.eye(4), atol=1e-12)

    def test_full_block_is_unitary(self) -> None:
        layer = SU2SwapConvolution(
            SU2SwapConvolutionConfig(num_qubits=4),
            parameters=(0.23, -0.41),
        )
        unitary = layer.unitary()
        np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(16), atol=1e-12)

    def test_convolution_is_globally_su2_equivariant(self) -> None:
        layer = SU2SwapConvolution(
            SU2SwapConvolutionConfig(num_qubits=5),
            parameters=(0.29, -0.17),
        )
        summary = convolution_equivariance_error(layer, num_trials=5, seed=7)
        self.assertLess(summary["max_error"], 1e-10)
