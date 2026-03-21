from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from eqnn.layers.convolution import SU2SwapConvolution, SU2SwapConvolutionConfig
from eqnn.physics.spin import IDENTITY, PAULI_Z
from eqnn.verification.equivariance import (
    convolution_equivariance_error,
    convolution_operator_equivariance_error,
)


class _BrokenLocalZLayer:
    def __init__(self, num_qubits: int) -> None:
        self.config = SimpleNamespace(num_qubits=num_qubits)
        self._unitary = np.kron(PAULI_Z, IDENTITY)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(self._unitary @ state, dtype=np.complex128)


class SU2SwapConvolutionTests(unittest.TestCase):
    def test_gate_is_unitary(self) -> None:
        gate = SU2SwapConvolution.gate(0.37)
        np.testing.assert_allclose(gate.conj().T @ gate, np.eye(4), atol=1e-12)

    def test_gate_has_expected_singlet_triplet_eigenphases(self) -> None:
        theta = 0.37
        gate = SU2SwapConvolution.gate(theta)

        singlet = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.complex128) / np.sqrt(2.0)
        triplet_plus = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        triplet_zero = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.complex128) / np.sqrt(2.0)
        triplet_minus = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex128)

        np.testing.assert_allclose(gate @ singlet, np.exp(1.0j * theta) * singlet, atol=1e-12)
        np.testing.assert_allclose(gate @ triplet_plus, np.exp(-1.0j * theta) * triplet_plus, atol=1e-12)
        np.testing.assert_allclose(gate @ triplet_zero, np.exp(-1.0j * theta) * triplet_zero, atol=1e-12)
        np.testing.assert_allclose(gate @ triplet_minus, np.exp(-1.0j * theta) * triplet_minus, atol=1e-12)

    def test_gate_known_limits_match_identity_and_swap_up_to_phase(self) -> None:
        np.testing.assert_allclose(
            SU2SwapConvolution.gate(0.0),
            np.eye(4, dtype=np.complex128),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            SU2SwapConvolution.gate(np.pi / 2.0),
            -1.0j * SU2SwapConvolution.swap_operator(),
            atol=1e-12,
        )

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

    def test_convolution_commutes_with_global_rotations_at_operator_level(self) -> None:
        layer = SU2SwapConvolution(
            SU2SwapConvolutionConfig(num_qubits=5),
            parameters=(0.29, -0.17),
        )
        summary = convolution_operator_equivariance_error(layer, num_trials=5, seed=9)
        self.assertLess(summary["max_error"], 1e-10)

    def test_equivariance_checker_detects_a_symmetry_breaking_layer(self) -> None:
        broken_layer = _BrokenLocalZLayer(num_qubits=2)
        summary = convolution_equivariance_error(broken_layer, num_trials=5, seed=31)
        self.assertGreater(summary["max_error"], 1e-3)
