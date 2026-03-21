"""Pooling layers for SU(2)-equivariant QCNNs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from eqnn.physics.quantum import (
    as_density_matrix,
    partial_trace_adjoint,
    partial_trace_density_matrix,
    qubit_permutation_operator,
)
from eqnn.types import ComplexArray


_PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_PAIR_PHI5_OPERATORS = (
    np.kron(_PAULI_Y, _PAULI_Z) - np.kron(_PAULI_Z, _PAULI_Y),
    np.kron(_PAULI_Z, _PAULI_X) - np.kron(_PAULI_X, _PAULI_Z),
    np.kron(_PAULI_X, _PAULI_Y) - np.kron(_PAULI_Y, _PAULI_X),
)
_PAIR_PHI5_OUTPUTS = (_PAULI_X, _PAULI_Y, _PAULI_Z)
_SQRT_THREE = float(np.sqrt(3.0))
_CHOI_EIGENVALUE_TOLERANCE = 1e-12
_PHYSICAL_COORDINATE_TOLERANCE = 1e-10


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponent = np.exp(-value)
        return float(1.0 / (1.0 + exponent))

    exponent = np.exp(value)
    return float(exponent / (1.0 + exponent))


def _map_plane_to_open_unit_disk(horizontal: float, vertical: float) -> np.ndarray:
    radius = float(np.hypot(horizontal, vertical))
    if radius < 1e-15:
        return np.zeros(2, dtype=np.float64)

    scale = float(np.tanh(radius) / radius)
    return np.asarray((scale * horizontal, scale * vertical), dtype=np.float64)


def _left_pair_partial_trace(density_matrix: ComplexArray) -> ComplexArray:
    return np.asarray(
        partial_trace_density_matrix(density_matrix, num_qubits=2, traced_out_sites=(1,)),
        dtype=np.complex128,
    )


def _right_pair_partial_trace(density_matrix: ComplexArray) -> ComplexArray:
    return np.asarray(
        partial_trace_density_matrix(density_matrix, num_qubits=2, traced_out_sites=(0,)),
        dtype=np.complex128,
    )


@dataclass(frozen=True)
class PartialTracePoolingConfig:
    num_qubits: int
    keep: str = "left"

    def __post_init__(self) -> None:
        if self.num_qubits < 2:
            raise ValueError("num_qubits must be at least 2")
        if self.keep not in {"left", "right"}:
            raise ValueError("keep must be either 'left' or 'right'")


class PartialTracePooling:
    """Pool each adjacent pair by tracing out one qubit."""

    def __init__(self, config: PartialTracePoolingConfig) -> None:
        self.config = config
        self.parameters = np.zeros(0, dtype=np.float64)

    def __call__(self, state: ComplexArray) -> ComplexArray:
        return self.apply(state)

    def apply(
        self,
        state: ComplexArray,
        parameters: Iterable[float] | None = None,
    ) -> ComplexArray:
        density_matrix = as_density_matrix(state)
        return np.asarray(
            partial_trace_density_matrix(
                density_matrix,
                self.config.num_qubits,
                self.trace_out_sites(),
            ),
            dtype=np.complex128,
        )

    @property
    def parameter_count(self) -> int:
        return 0

    @property
    def output_num_qubits(self) -> int:
        return self.config.num_qubits - len(self.trace_out_sites())

    def trace_out_sites(self) -> tuple[int, ...]:
        """Return the sites removed by this pooling layer."""

        if self.config.keep == "left":
            return tuple(range(1, self.config.num_qubits, 2))
        if self.config.num_qubits % 2 == 0:
            return tuple(range(0, self.config.num_qubits, 2))
        return tuple(range(0, self.config.num_qubits - 1, 2))

    def kept_sites(self) -> tuple[int, ...]:
        traced_out = set(self.trace_out_sites())
        return tuple(site for site in range(self.config.num_qubits) if site not in traced_out)

    def adjoint_apply(
        self,
        observable: ComplexArray,
        parameters: Iterable[float] | None = None,
    ) -> ComplexArray:
        """Apply the adjoint map of partial-trace pooling to an observable."""

        return np.asarray(
            partial_trace_adjoint(
                as_density_matrix(observable),
                self.config.num_qubits,
                self.trace_out_sites(),
            ),
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class SU2EquivariantPoolingConfig:
    num_qubits: int
    warm_start: str = "left"

    def __post_init__(self) -> None:
        if self.num_qubits < 2:
            raise ValueError("num_qubits must be at least 2")
        if self.warm_start not in {"left", "right"}:
            raise ValueError("warm_start must be either 'left' or 'right'")


class SU2EquivariantPooling:
    """Paper-derived SU(2)-equivariant 2 -> 1 pooling on adjacent pairs.

    The local channel uses the complete three-parameter trace-preserving
    SU(2)-equivariant family described in the EQNN theory paper:

    Phi = phi_1 + x phi_5 + y (phi_3 - phi_1) + z (phi_4 - phi_1).

    The exposed optimization parameters live in R^3 and are mapped into the
    exact CPTP region through the Choi-matrix block decomposition. That keeps
    training unconstrained while preserving the full local channel family.
    """

    def __init__(
        self,
        config: SU2EquivariantPoolingConfig,
        parameters: Iterable[float] | None = None,
    ) -> None:
        self.config = config
        self.parameters = self._initialize_parameters(parameters)

    def __call__(self, state: ComplexArray) -> ComplexArray:
        return self.apply(state)

    def apply(
        self,
        state: ComplexArray,
        parameters: Iterable[float] | None = None,
    ) -> ComplexArray:
        density_matrix = as_density_matrix(state)
        parameter_array = self.parameters if parameters is None else self._validate_parameters(parameters)
        current_density = density_matrix
        current_num_qubits = self.config.num_qubits
        kraus_operators = self.local_kraus_operators(parameter_array)

        for pair_start in reversed(range(0, current_num_qubits - 1, 2)):
            current_density = self._apply_local_channel_to_pair(
                current_density,
                current_num_qubits,
                pair_start,
                kraus_operators,
            )
            current_num_qubits -= 1

        return np.asarray(current_density, dtype=np.complex128)

    def adjoint_apply(
        self,
        observable: ComplexArray,
        parameters: Iterable[float] | None = None,
    ) -> ComplexArray:
        current_observable = as_density_matrix(observable)
        parameter_array = self.parameters if parameters is None else self._validate_parameters(parameters)
        current_num_qubits = self.output_num_qubits
        kraus_operators = self.local_kraus_operators(parameter_array)

        forward_steps: list[tuple[int, int]] = []
        forward_num_qubits = self.config.num_qubits
        for pair_start in reversed(range(0, forward_num_qubits - 1, 2)):
            forward_steps.append((forward_num_qubits, pair_start))
            forward_num_qubits -= 1

        for step_num_qubits, pair_start in reversed(forward_steps):
            current_observable = self._apply_local_adjoint_to_pair(
                current_observable,
                step_num_qubits,
                pair_start,
                kraus_operators,
            )
            current_num_qubits += 1

        return np.asarray(current_observable, dtype=np.complex128)

    @property
    def parameter_count(self) -> int:
        return 3

    @property
    def output_num_qubits(self) -> int:
        return (self.config.num_qubits + 1) // 2

    def get_parameters(self) -> np.ndarray:
        return self.parameters.copy()

    def set_parameters(self, parameters: Iterable[float]) -> None:
        self.parameters = self._validate_parameters(parameters)

    def physical_coordinates(
        self,
        parameters: Iterable[float] | None = None,
    ) -> np.ndarray:
        parameter_array = self.parameters if parameters is None else self._validate_parameters(parameters)
        return self._physical_coordinates_from_parameters(parameter_array)

    def physical_coordinate_jacobian(
        self,
        parameters: Iterable[float] | None = None,
    ) -> np.ndarray:
        parameter_array = self.parameters if parameters is None else self._validate_parameters(parameters)
        return self._physical_coordinate_jacobian_from_parameters(parameter_array)

    def local_kraus_operators(
        self,
        parameters: Iterable[float] | None = None,
    ) -> list[ComplexArray]:
        choi_matrix = self.local_choi_matrix(parameters=parameters)
        eigenvalues, eigenvectors = np.linalg.eigh(choi_matrix)

        kraus_operators: list[ComplexArray] = []
        for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors.T):
            clipped_eigenvalue = 0.0 if eigenvalue < _CHOI_EIGENVALUE_TOLERANCE else float(eigenvalue)
            if clipped_eigenvalue == 0.0:
                continue

            kraus = np.sqrt(clipped_eigenvalue) * eigenvector.reshape((4, 2), order="C").T
            kraus_operators.append(np.asarray(kraus, dtype=np.complex128))
        return kraus_operators

    def local_choi_matrix(
        self,
        parameters: Iterable[float] | None = None,
        *,
        physical_coordinates: Iterable[float] | None = None,
    ) -> ComplexArray:
        if physical_coordinates is not None:
            coordinates = self.validate_physical_coordinates(physical_coordinates)
        else:
            coordinates = self.physical_coordinates(parameters=parameters)
        return self.local_choi_matrix_from_coordinates(coordinates)

    def apply_local_channel(
        self,
        density_matrix: ComplexArray,
        parameters: Iterable[float] | None = None,
        *,
        physical_coordinates: Iterable[float] | None = None,
    ) -> ComplexArray:
        local_density = as_density_matrix(density_matrix)
        if local_density.shape != (4, 4):
            raise ValueError(
                "apply_local_channel expects a two-qubit density matrix with shape (4, 4)"
            )

        if physical_coordinates is not None:
            coordinates = self.validate_physical_coordinates(physical_coordinates)
        else:
            coordinates = self.physical_coordinates(parameters=parameters)
        return self.local_channel_from_coordinates(local_density, coordinates)

    def local_channel_parameter_derivatives(
        self,
        density_matrix: ComplexArray,
        parameters: Iterable[float] | None = None,
    ) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
        local_density = as_density_matrix(density_matrix)
        if local_density.shape != (4, 4):
            raise ValueError(
                "local_channel_parameter_derivatives expects a two-qubit density matrix with shape (4, 4)"
            )

        jacobian = self.physical_coordinate_jacobian(parameters=parameters)
        phi_one = np.trace(local_density) * np.eye(2, dtype=np.complex128) / 2.0
        phi_three = _left_pair_partial_trace(local_density)
        phi_four = _right_pair_partial_trace(local_density)
        phi_five = sum(
            np.trace(local_density @ input_operator) * output_operator
            for input_operator, output_operator in zip(_PAIR_PHI5_OPERATORS, _PAIR_PHI5_OUTPUTS)
        )
        basis_outputs = (
            phi_five,
            phi_three - phi_one,
            phi_four - phi_one,
        )
        derivatives: list[ComplexArray] = []
        for parameter_index in range(3):
            derivative = sum(
                jacobian[coordinate_index, parameter_index] * basis_output
                for coordinate_index, basis_output in enumerate(basis_outputs)
            )
            derivatives.append(np.asarray(derivative, dtype=np.complex128))
        return tuple(derivatives)  # type: ignore[return-value]

    def parameter_gradient(
        self,
        density_matrix: ComplexArray,
        output_adjoint: ComplexArray,
        parameters: Iterable[float] | None = None,
    ) -> np.ndarray:
        parameter_array = self.parameters if parameters is None else self._validate_parameters(parameters)
        current_density = as_density_matrix(density_matrix)
        current_num_qubits = self.config.num_qubits

        forward_steps: list[tuple[int, int, ComplexArray]] = []
        for pair_start in reversed(range(0, current_num_qubits - 1, 2)):
            forward_steps.append((current_num_qubits, pair_start, current_density))
            current_density = self._apply_local_map_to_pair(
                current_density,
                current_num_qubits,
                pair_start,
                lambda local_density: self.apply_local_channel(local_density, parameters=parameter_array),
            )
            current_num_qubits -= 1

        gradient = np.zeros(3, dtype=np.float64)
        adjoint = as_density_matrix(output_adjoint)
        kraus_operators = self.local_kraus_operators(parameter_array)
        for step_num_qubits, pair_start, step_density in reversed(forward_steps):
            for parameter_index in range(self.parameter_count):
                derivative_density = self._apply_local_map_to_pair(
                    step_density,
                    step_num_qubits,
                    pair_start,
                    lambda local_density, parameter_index=parameter_index: (
                        self.local_channel_parameter_derivatives(local_density, parameters=parameter_array)[parameter_index]
                    ),
                )
                gradient[parameter_index] += float(np.real_if_close(np.trace(adjoint @ derivative_density)))
            adjoint = self._apply_local_adjoint_to_pair(
                adjoint,
                step_num_qubits,
                pair_start,
                kraus_operators,
            )

        return np.asarray(gradient, dtype=np.float64)

    def _initialize_parameters(self, parameters: Iterable[float] | None) -> np.ndarray:
        if parameters is None:
            if self.config.warm_start == "left":
                return np.asarray((-8.0, 8.0, 0.0), dtype=np.float64)
            return np.asarray((-8.0, -8.0, 0.0), dtype=np.float64)
        return self._validate_parameters(parameters)

    def _validate_parameters(self, parameters: Iterable[float]) -> np.ndarray:
        parameter_array = np.asarray(list(parameters), dtype=np.float64)
        if parameter_array.shape != (self.parameter_count,):
            raise ValueError(
                f"Expected {self.parameter_count} parameters, got shape {parameter_array.shape}"
            )
        return parameter_array

    @classmethod
    def validate_physical_coordinates(cls, physical_coordinates: Iterable[float]) -> np.ndarray:
        coordinate_array = np.asarray(list(physical_coordinates), dtype=np.float64)
        if coordinate_array.shape != (3,):
            raise ValueError(f"Expected physical coordinates of shape (3,), got {coordinate_array.shape}")

        x_coordinate, y_coordinate, z_coordinate = coordinate_array
        singlet_weight = 0.5 * (1.0 - y_coordinate - z_coordinate)
        triplet_diagonal = 0.5 + y_coordinate + z_coordinate
        triplet_offdiagonal = 0.5 * _SQRT_THREE * (
            (y_coordinate - z_coordinate) - 2.0j * x_coordinate
        )
        determinant = 0.5 * triplet_diagonal - abs(triplet_offdiagonal) ** 2

        if singlet_weight < -_PHYSICAL_COORDINATE_TOLERANCE:
            raise ValueError("physical coordinates violate the spin-3/2 positivity bound")
        if triplet_diagonal < -_PHYSICAL_COORDINATE_TOLERANCE:
            raise ValueError("physical coordinates violate the spin-1/2 positivity bound")
        if determinant < -_PHYSICAL_COORDINATE_TOLERANCE:
            raise ValueError("physical coordinates lie outside the SU(2)-equivariant CPTP region")

        return coordinate_array

    @classmethod
    def local_choi_matrix_from_coordinates(
        cls,
        physical_coordinates: Iterable[float],
    ) -> ComplexArray:
        coordinate_array = cls.validate_physical_coordinates(physical_coordinates)
        choi_matrix = np.zeros((8, 8), dtype=np.complex128)
        for row_index in range(4):
            for column_index in range(4):
                basis_operator = np.zeros((4, 4), dtype=np.complex128)
                basis_operator[row_index, column_index] = 1.0
                channel_output = cls.local_channel_from_coordinates(basis_operator, coordinate_array)
                choi_basis_operator = np.zeros((4, 4), dtype=np.complex128)
                choi_basis_operator[row_index, column_index] = 1.0
                choi_matrix += np.kron(choi_basis_operator, channel_output)
        return np.asarray(choi_matrix, dtype=np.complex128)

    @classmethod
    def local_channel_from_coordinates(
        cls,
        density_matrix: ComplexArray,
        physical_coordinates: Iterable[float],
    ) -> ComplexArray:
        coordinate_array = cls.validate_physical_coordinates(physical_coordinates)
        local_density = as_density_matrix(density_matrix)
        if local_density.shape != (4, 4):
            raise ValueError(
                "local_channel_from_coordinates expects a two-qubit density matrix with shape (4, 4)"
            )

        x_coordinate, y_coordinate, z_coordinate = coordinate_array
        phi_one = np.trace(local_density) * np.eye(2, dtype=np.complex128) / 2.0
        phi_three = _left_pair_partial_trace(local_density)
        phi_four = _right_pair_partial_trace(local_density)
        phi_five = sum(
            np.trace(local_density @ input_operator) * output_operator
            for input_operator, output_operator in zip(_PAIR_PHI5_OPERATORS, _PAIR_PHI5_OUTPUTS)
        )
        output = (
            (1.0 - y_coordinate - z_coordinate) * phi_one
            + y_coordinate * phi_three
            + z_coordinate * phi_four
            + x_coordinate * phi_five
        )
        return np.asarray(output, dtype=np.complex128)

    @staticmethod
    def _physical_coordinates_from_parameters(parameters: np.ndarray) -> np.ndarray:
        singlet_weight = 0.75 * _sigmoid(float(parameters[0]))
        open_disk_point = _map_plane_to_open_unit_disk(float(parameters[1]), float(parameters[2]))
        offdiagonal_radius = np.sqrt(max(0.0, 0.75 - singlet_weight))
        triplet_offdiagonal = offdiagonal_radius * (
            open_disk_point[0] + 1.0j * open_disk_point[1]
        )

        coupling_sum = 1.0 - 2.0 * singlet_weight
        coupling_difference = 2.0 * np.real(triplet_offdiagonal) / _SQRT_THREE
        x_coordinate = -np.imag(triplet_offdiagonal) / _SQRT_THREE
        y_coordinate = 0.5 * (coupling_sum + coupling_difference)
        z_coordinate = 0.5 * (coupling_sum - coupling_difference)
        return np.asarray((x_coordinate, y_coordinate, z_coordinate), dtype=np.float64)

    @staticmethod
    def _physical_coordinate_jacobian_from_parameters(parameters: np.ndarray) -> np.ndarray:
        sigmoid_value = _sigmoid(float(parameters[0]))
        singlet_weight = 0.75 * sigmoid_value
        singlet_derivative = 0.75 * sigmoid_value * (1.0 - sigmoid_value)

        disk_point, disk_jacobian = SU2EquivariantPooling._disk_point_and_jacobian(
            float(parameters[1]),
            float(parameters[2]),
        )
        offdiagonal_radius = np.sqrt(max(0.0, 0.75 - singlet_weight))
        radius_derivative = 0.0
        if offdiagonal_radius > 1e-15:
            radius_derivative = -0.5 * singlet_derivative / offdiagonal_radius

        triplet_offdiagonal = offdiagonal_radius * (
            disk_point[0] + 1.0j * disk_point[1]
        )
        triplet_derivatives = (
            radius_derivative * (disk_point[0] + 1.0j * disk_point[1]),
            offdiagonal_radius * (disk_jacobian[0, 0] + 1.0j * disk_jacobian[1, 0]),
            offdiagonal_radius * (disk_jacobian[0, 1] + 1.0j * disk_jacobian[1, 1]),
        )

        coordinate_jacobian = np.zeros((3, 3), dtype=np.float64)
        for parameter_index, triplet_derivative in enumerate(triplet_derivatives):
            coupling_sum_derivative = -2.0 * singlet_derivative if parameter_index == 0 else 0.0
            coupling_difference_derivative = 2.0 * np.real(triplet_derivative) / _SQRT_THREE
            coordinate_jacobian[0, parameter_index] = -np.imag(triplet_derivative) / _SQRT_THREE
            coordinate_jacobian[1, parameter_index] = 0.5 * (
                coupling_sum_derivative + coupling_difference_derivative
            )
            coordinate_jacobian[2, parameter_index] = 0.5 * (
                coupling_sum_derivative - coupling_difference_derivative
            )
        return coordinate_jacobian

    @staticmethod
    def _disk_point_and_jacobian(horizontal: float, vertical: float) -> tuple[np.ndarray, np.ndarray]:
        radius = float(np.hypot(horizontal, vertical))
        if radius < 1e-15:
            return np.zeros(2, dtype=np.float64), np.eye(2, dtype=np.float64)

        scale = float(np.tanh(radius) / radius)
        scale_radius_derivative = float(
            (radius / np.cosh(radius) ** 2 - np.tanh(radius)) / (radius**2)
        )
        scale_horizontal_derivative = scale_radius_derivative * horizontal / radius
        scale_vertical_derivative = scale_radius_derivative * vertical / radius

        point = np.asarray((scale * horizontal, scale * vertical), dtype=np.float64)
        jacobian = np.asarray(
            [
                [
                    scale + horizontal * scale_horizontal_derivative,
                    horizontal * scale_vertical_derivative,
                ],
                [
                    vertical * scale_horizontal_derivative,
                    scale + vertical * scale_vertical_derivative,
                ],
            ],
            dtype=np.float64,
        )
        return point, jacobian

    def _apply_local_map_to_pair(
        self,
        density_matrix: ComplexArray,
        num_qubits: int,
        pair_start: int,
        local_map: Callable[[ComplexArray], ComplexArray],
    ) -> ComplexArray:
        pair_sites = (pair_start, pair_start + 1)
        remaining_sites = tuple(site for site in range(num_qubits) if site not in pair_sites)
        input_permutation = pair_sites + remaining_sites
        input_permutation_operator = qubit_permutation_operator(num_qubits, input_permutation)
        reordered_density = (
            input_permutation_operator
            @ density_matrix
            @ input_permutation_operator.conjugate().T
        )

        environment_dimension = 1 << (num_qubits - 2)
        reordered_tensor = reordered_density.reshape(4, environment_dimension, 4, environment_dimension)
        output_tensor = np.zeros((2, environment_dimension, 2, environment_dimension), dtype=np.complex128)
        for left_environment_index in range(environment_dimension):
            for right_environment_index in range(environment_dimension):
                local_density = reordered_tensor[
                    :,
                    left_environment_index,
                    :,
                    right_environment_index,
                ].reshape(4, 4)
                output_tensor[:, left_environment_index, :, right_environment_index] = local_map(local_density)

        reordered_output = output_tensor.reshape(2 * environment_dimension, 2 * environment_dimension)
        if num_qubits == 2:
            return reordered_output

        current_labels = [pair_start] + list(remaining_sites)
        target_labels = [site for site in range(num_qubits) if site != pair_start + 1]
        output_permutation = tuple(current_labels.index(label) for label in target_labels)
        output_permutation_operator = qubit_permutation_operator(num_qubits - 1, output_permutation)
        return np.asarray(
            output_permutation_operator
            @ reordered_output
            @ output_permutation_operator.conjugate().T,
            dtype=np.complex128,
        )

    def _apply_local_channel_to_pair(
        self,
        density_matrix: ComplexArray,
        num_qubits: int,
        pair_start: int,
        kraus_operators: list[ComplexArray],
    ) -> ComplexArray:
        output_dimension = 1 << (num_qubits - 1)
        updated_density = np.zeros((output_dimension, output_dimension), dtype=np.complex128)
        for local_kraus in kraus_operators:
            embedded_kraus = self._embed_local_kraus(local_kraus, num_qubits, pair_start)
            updated_density += embedded_kraus @ density_matrix @ embedded_kraus.conjugate().T
        return updated_density

    def _apply_local_adjoint_to_pair(
        self,
        observable: ComplexArray,
        num_qubits: int,
        pair_start: int,
        kraus_operators: list[ComplexArray],
    ) -> ComplexArray:
        output_dimension = 1 << num_qubits
        updated_observable = np.zeros((output_dimension, output_dimension), dtype=np.complex128)
        for local_kraus in kraus_operators:
            embedded_kraus = self._embed_local_kraus(local_kraus, num_qubits, pair_start)
            updated_observable += embedded_kraus.conjugate().T @ observable @ embedded_kraus
        return updated_observable

    @staticmethod
    def _embed_local_kraus(local_kraus: ComplexArray, num_qubits: int, pair_start: int) -> ComplexArray:
        left_dimension = 1 << pair_start
        right_dimension = 1 << (num_qubits - pair_start - 2)

        embedded = local_kraus
        if left_dimension > 1:
            embedded = np.kron(np.eye(left_dimension, dtype=np.complex128), embedded)
        if right_dimension > 1:
            embedded = np.kron(embedded, np.eye(right_dimension, dtype=np.complex128))
        return np.asarray(embedded, dtype=np.complex128)
