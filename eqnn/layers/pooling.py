"""Pooling layers for SU(2)-equivariant QCNNs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from eqnn.physics.quantum import as_density_matrix, partial_trace_density_matrix
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
