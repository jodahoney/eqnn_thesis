from __future__ import annotations

import unittest

import numpy as np

from eqnn.layers.pooling import (
    PartialTracePooling,
    PartialTracePoolingConfig,
    SU2EquivariantPooling,
    SU2EquivariantPoolingConfig,
)
from eqnn.verification.equivariance import pooling_equivariance_error, random_complex_statevector


def _trace_output_of_local_choi(choi_matrix: np.ndarray) -> np.ndarray:
    return np.trace(choi_matrix.reshape(4, 2, 4, 2), axis1=1, axis2=3)


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


class SU2EquivariantPoolingTests(unittest.TestCase):
    def test_equivariant_pooling_preserves_density_matrix_properties(self) -> None:
        state = random_complex_statevector(num_qubits=4, seed=21)
        pooling = SU2EquivariantPooling(
            SU2EquivariantPoolingConfig(num_qubits=4),
            parameters=(0.3, -0.1, 0.2),
        )

        reduced = pooling(state)

        self.assertEqual(reduced.shape, (4, 4))
        self.assertAlmostEqual(float(np.trace(reduced).real), 1.0, places=12)
        np.testing.assert_allclose(reduced, reduced.conj().T, atol=1e-12)
        eigenvalues = np.linalg.eigvalsh(reduced)
        self.assertGreaterEqual(float(np.min(eigenvalues)), -1e-12)

    def test_parameter_map_stays_inside_the_exact_cptp_region(self) -> None:
        pooling = SU2EquivariantPooling(
            SU2EquivariantPoolingConfig(num_qubits=4),
            parameters=(0.8, -1.4, 0.7),
        )
        x_coordinate, y_coordinate, z_coordinate = pooling.physical_coordinates()

        self.assertLessEqual(y_coordinate + z_coordinate, 1.0 + 1e-12)
        self.assertGreaterEqual(0.5 + y_coordinate + z_coordinate, -1e-12)
        self.assertLessEqual(
            3.0 * x_coordinate**2 + y_coordinate**2 - y_coordinate * z_coordinate + z_coordinate**2,
            0.25 * (1.0 + y_coordinate + z_coordinate) ** 2 + 1e-12,
        )

    def test_local_choi_matrix_is_positive_and_trace_preserving(self) -> None:
        pooling = SU2EquivariantPooling(
            SU2EquivariantPoolingConfig(num_qubits=4),
            parameters=(0.4, -0.2, 0.1),
        )
        choi_matrix = pooling.local_choi_matrix()

        eigenvalues = np.linalg.eigvalsh(choi_matrix)
        self.assertGreaterEqual(float(np.min(eigenvalues)), -1e-12)
        np.testing.assert_allclose(_trace_output_of_local_choi(choi_matrix), np.eye(4), atol=1e-12)

    def test_local_channel_matches_the_kraus_realization(self) -> None:
        state = random_complex_statevector(num_qubits=2, seed=25)
        density_matrix = np.outer(state, np.conjugate(state))
        pooling = SU2EquivariantPooling(
            SU2EquivariantPoolingConfig(num_qubits=2),
            parameters=(0.2, -0.3, 0.5),
        )

        reference = pooling.apply_local_channel(density_matrix)
        realized = np.zeros((2, 2), dtype=np.complex128)
        for kraus in pooling.local_kraus_operators():
            realized += kraus @ density_matrix @ kraus.conjugate().T

        np.testing.assert_allclose(realized, reference, atol=1e-12)

    def test_exact_basis_recovers_reference_channels(self) -> None:
        state = random_complex_statevector(num_qubits=2, seed=29)
        density_matrix = np.outer(state, np.conjugate(state))

        maximally_mixed_output = SU2EquivariantPooling.local_channel_from_coordinates(
            density_matrix,
            physical_coordinates=(0.0, 0.0, 0.0),
        )
        left_trace_output = SU2EquivariantPooling.local_channel_from_coordinates(
            density_matrix,
            physical_coordinates=(0.0, 1.0, 0.0),
        )
        right_trace_output = SU2EquivariantPooling.local_channel_from_coordinates(
            density_matrix,
            physical_coordinates=(0.0, 0.0, 1.0),
        )

        np.testing.assert_allclose(maximally_mixed_output, 0.5 * np.eye(2), atol=1e-12)
        np.testing.assert_allclose(
            left_trace_output,
            PartialTracePooling(PartialTracePoolingConfig(num_qubits=2, keep="left"))(density_matrix),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            right_trace_output,
            PartialTracePooling(PartialTracePoolingConfig(num_qubits=2, keep="right"))(density_matrix),
            atol=1e-12,
        )

    def test_default_warm_start_tracks_the_requested_partial_trace_orientation(self) -> None:
        left_warm_start = SU2EquivariantPooling(
            SU2EquivariantPoolingConfig(num_qubits=2, warm_start="left")
        )
        right_warm_start = SU2EquivariantPooling(
            SU2EquivariantPoolingConfig(num_qubits=2, warm_start="right")
        )

        left_x, left_y, left_z = left_warm_start.physical_coordinates()
        right_x, right_y, right_z = right_warm_start.physical_coordinates()

        self.assertLess(abs(left_x), 1e-10)
        self.assertGreater(left_y, 0.99)
        self.assertLess(abs(left_z), 1e-2)

        self.assertLess(abs(right_x), 1e-10)
        self.assertGreater(right_z, 0.99)
        self.assertLess(abs(right_y), 1e-2)

    def test_invalid_physical_coordinates_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SU2EquivariantPooling.validate_physical_coordinates((0.5, 0.0, 0.0))

    def test_equivariant_pooling_is_globally_su2_equivariant(self) -> None:
        pooling = SU2EquivariantPooling(
            SU2EquivariantPoolingConfig(num_qubits=4),
            parameters=(0.2, -0.3, 0.1),
        )
        summary = pooling_equivariance_error(pooling, num_trials=5, seed=17)
        self.assertLess(summary["max_error"], 1e-10)

    def test_equivariant_pooling_parameter_gradient_matches_finite_difference(self) -> None:
        state = random_complex_statevector(num_qubits=4, seed=31)
        density_matrix = np.outer(state, np.conjugate(state))
        pooling = SU2EquivariantPooling(
            SU2EquivariantPoolingConfig(num_qubits=4),
            parameters=(0.2, -0.3, 0.1),
        )
        rng = np.random.default_rng(7)
        random_matrix = rng.normal(size=(4, 4)) + 1.0j * rng.normal(size=(4, 4))
        observable = random_matrix + random_matrix.conjugate().T

        exact_gradient = pooling.parameter_gradient(density_matrix, observable)

        eps = 1e-6
        finite_difference_gradient = np.zeros(pooling.parameter_count, dtype=np.float64)
        base_parameters = pooling.get_parameters()
        for index in range(pooling.parameter_count):
            offset = np.zeros(pooling.parameter_count, dtype=np.float64)
            offset[index] = eps
            output_plus = pooling.apply(density_matrix, parameters=base_parameters + offset)
            output_minus = pooling.apply(density_matrix, parameters=base_parameters - offset)
            value_plus = float(np.real_if_close(np.trace(observable @ output_plus)))
            value_minus = float(np.real_if_close(np.trace(observable @ output_minus)))
            finite_difference_gradient[index] = (value_plus - value_minus) / (2.0 * eps)

        np.testing.assert_allclose(exact_gradient, finite_difference_gradient, atol=1e-5)
