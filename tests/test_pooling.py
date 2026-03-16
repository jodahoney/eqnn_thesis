from __future__ import annotations

import unittest

import numpy as np

from eqnn.layers.pooling import PartialTracePooling, PartialTracePoolingConfig
from eqnn.verification.equivariance import pooling_equivariance_error, random_complex_statevector


class PartialTracePoolingTests(unittest.TestCase):
    def test_pooling_preserves_density_matrix_properties(self) -> None:
        state = random_complex_statevector(num_qubits=4, seed=13)
        pooling = PartialTracePooling(PartialTracePoolingConfig(num_qubits=4, keep="left"))

        reduced = pooling(state)

        self.assertEqual(reduced.shape, (4, 4))
        self.assertAlmostEqual(float(np.trace(reduced).real), 1.0, places=12)
        np.testing.assert_allclose(reduced, reduced.conj().T, atol=1e-12)
        eigenvalues = np.linalg.eigvalsh(reduced)
        self.assertGreaterEqual(float(np.min(eigenvalues)), -1e-12)

    def test_odd_qubit_pooling_keeps_the_unpaired_qubit(self) -> None:
        pooling = PartialTracePooling(PartialTracePoolingConfig(num_qubits=5, keep="left"))
        self.assertEqual(pooling.trace_out_sites(), (1, 3))
        self.assertEqual(pooling.output_num_qubits, 3)

    def test_pooling_is_globally_su2_equivariant(self) -> None:
        pooling = PartialTracePooling(PartialTracePoolingConfig(num_qubits=4, keep="left"))
        summary = pooling_equivariance_error(pooling, num_trials=5, seed=11)
        self.assertLess(summary["max_error"], 1e-10)
