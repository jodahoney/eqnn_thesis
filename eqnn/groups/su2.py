"""SU(2) representation utilities for global symmetry checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eqnn.physics.spin import IDENTITY, PAULI_X, PAULI_Y, PAULI_Z, kron_all
from eqnn.types import ComplexArray


@dataclass(frozen=True)
class SU2Group:
    name: str = "SU(2)"

    @staticmethod
    def single_qubit_rotation(axis: tuple[float, float, float], angle: float) -> ComplexArray:
        """Return the spin-1/2 representation of an SU(2) rotation."""

        axis_array = np.asarray(axis, dtype=np.float64)
        norm = np.linalg.norm(axis_array)
        if norm == 0.0:
            raise ValueError("Rotation axis must be non-zero")
        axis_array = axis_array / norm

        generator = (
            axis_array[0] * PAULI_X
            + axis_array[1] * PAULI_Y
            + axis_array[2] * PAULI_Z
        )
        half_angle = 0.5 * float(angle)
        return np.asarray(
            np.cos(half_angle) * IDENTITY - 1.0j * np.sin(half_angle) * generator,
            dtype=np.complex128,
        )

    def global_rotation(
        self,
        num_qubits: int,
        axis: tuple[float, float, float],
        angle: float,
    ) -> ComplexArray:
        """Return the global SU(2) action U^{⊗n} on n qubits."""

        local_rotation = self.single_qubit_rotation(axis, angle)
        return kron_all([local_rotation for _ in range(num_qubits)])

    def representation(self, num_qubits: int, *parameters: float) -> ComplexArray:
        if len(parameters) != 4:
            raise ValueError(
                "SU2Group.representation expects parameters (axis_x, axis_y, axis_z, angle)"
            )
        axis = (parameters[0], parameters[1], parameters[2])
        angle = parameters[3]
        return self.global_rotation(num_qubits, axis, angle)
