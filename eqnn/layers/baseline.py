"""Symmetry-agnostic convolution layers for baseline QCNN models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from eqnn.physics.spin import IDENTITY, PAULI_X, PAULI_Y, PAULI_Z, kron_all
from eqnn.types import ComplexArray, RealArray

_TWO_QUBIT_XX = np.kron(PAULI_X, PAULI_X)
_TWO_QUBIT_YY = np.kron(PAULI_Y, PAULI_Y)
_TWO_QUBIT_ZZ = np.kron(PAULI_Z, PAULI_Z)


@dataclass(frozen=True)
class AnisotropicConvolutionConfig:
    num_qubits: int
    parity_sequence: tuple[str, ...] = ("even", "odd")
    shared_parameter: bool = True

    def __post_init__(self) -> None:
        if self.num_qubits < 2:
            raise ValueError("num_qubits must be at least 2")
        for parity in self.parity_sequence:
            if parity not in {"even", "odd"}:
                raise ValueError("parity_sequence entries must be 'even' or 'odd'")


class AnisotropicConvolution:
    """Brickwork baseline layer built from anisotropic XX/YY/ZZ couplings.

    Each two-qubit gate is

        exp(-i theta_x X⊗X) exp(-i theta_y Y⊗Y) exp(-i theta_z Z⊗Z),

    which is generally not SU(2)-equivariant when the three angles differ.
    """

    def __init__(
        self,
        config: AnisotropicConvolutionConfig,
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
        unitary = self.unitary(parameters=parameters)
        if state.ndim == 1:
            return np.asarray(unitary @ state, dtype=np.complex128)
        if state.ndim == 2:
            return np.asarray(unitary @ state @ unitary.conjugate().T, dtype=np.complex128)
        raise ValueError("State must be either a statevector or a density matrix")

    @staticmethod
    def gate(theta_x: float, theta_y: float, theta_z: float) -> ComplexArray:
        gate, _ = AnisotropicConvolution.gate_and_derivatives(theta_x, theta_y, theta_z)
        return gate

    @staticmethod
    def gate_and_derivatives(
        theta_x: float,
        theta_y: float,
        theta_z: float,
    ) -> tuple[ComplexArray, tuple[ComplexArray, ComplexArray, ComplexArray]]:
        identity = np.eye(4, dtype=np.complex128)
        x_gate = np.cos(theta_x) * identity - 1.0j * np.sin(theta_x) * _TWO_QUBIT_XX
        y_gate = np.cos(theta_y) * identity - 1.0j * np.sin(theta_y) * _TWO_QUBIT_YY
        z_gate = np.cos(theta_z) * identity - 1.0j * np.sin(theta_z) * _TWO_QUBIT_ZZ
        dx_gate = -np.sin(theta_x) * identity - 1.0j * np.cos(theta_x) * _TWO_QUBIT_XX
        dy_gate = -np.sin(theta_y) * identity - 1.0j * np.cos(theta_y) * _TWO_QUBIT_YY
        dz_gate = -np.sin(theta_z) * identity - 1.0j * np.cos(theta_z) * _TWO_QUBIT_ZZ
        gate = np.asarray(z_gate @ y_gate @ x_gate, dtype=np.complex128)
        derivatives = (
            np.asarray(z_gate @ y_gate @ dx_gate, dtype=np.complex128),
            np.asarray(z_gate @ dy_gate @ x_gate, dtype=np.complex128),
            np.asarray(dz_gate @ y_gate @ x_gate, dtype=np.complex128),
        )
        return gate, derivatives

    @property
    def parameter_count(self) -> int:
        if self.config.shared_parameter:
            return 3 * len(self.active_parities())
        return 3 * sum(len(self.pairs_for_parity(parity)) for parity in self.config.parity_sequence)

    def get_parameters(self) -> RealArray:
        return self.parameters.copy()

    def set_parameters(self, parameters: Iterable[float]) -> None:
        self.parameters = self._validate_parameters(parameters)

    def pairs_for_parity(self, parity: str) -> tuple[tuple[int, int], ...]:
        start = 0 if parity == "even" else 1
        return tuple((site, site + 1) for site in range(start, self.config.num_qubits - 1, 2))

    def active_parities(self) -> tuple[str, ...]:
        return tuple(
            parity for parity in self.config.parity_sequence if self.pairs_for_parity(parity)
        )

    def unitary(self, parameters: Iterable[float] | None = None) -> ComplexArray:
        return self.unitary_and_gradients(parameters=parameters)[0]

    def unitary_and_gradients(
        self,
        parameters: Iterable[float] | None = None,
    ) -> tuple[ComplexArray, list[ComplexArray]]:
        parameter_array = self.parameters if parameters is None else self._validate_parameters(parameters)

        sublayer_unitaries: list[ComplexArray] = []
        sublayer_gradients: list[list[ComplexArray]] = []
        parameter_offset = 0
        for parity in self.active_parities():
            pairs = self.pairs_for_parity(parity)
            if self.config.shared_parameter:
                parity_parameters = parameter_array[parameter_offset : parameter_offset + 3]
            else:
                parity_parameters = parameter_array[
                    parameter_offset : parameter_offset + 3 * len(pairs)
                ]
            sublayer_unitary, parity_gradients = self._sublayer_unitary_and_gradients(
                parity,
                parity_parameters,
            )
            sublayer_unitaries.append(sublayer_unitary)
            sublayer_gradients.append(parity_gradients)
            parameter_offset += len(parity_parameters)

        prefix_products: list[ComplexArray] = []
        current_prefix = np.eye(1 << self.config.num_qubits, dtype=np.complex128)
        for sublayer_unitary in sublayer_unitaries:
            prefix_products.append(current_prefix)
            current_prefix = sublayer_unitary @ current_prefix
        total_unitary = current_prefix

        suffix_products: list[ComplexArray] = [np.eye(1 << self.config.num_qubits, dtype=np.complex128)] * len(
            sublayer_unitaries
        )
        current_suffix = np.eye(1 << self.config.num_qubits, dtype=np.complex128)
        for index in range(len(sublayer_unitaries) - 1, -1, -1):
            suffix_products[index] = current_suffix
            current_suffix = current_suffix @ sublayer_unitaries[index]

        gradients: list[ComplexArray] = []
        for sublayer_index, parity_gradients in enumerate(sublayer_gradients):
            for parity_gradient in parity_gradients:
                gradients.append(
                    np.asarray(
                        suffix_products[sublayer_index]
                        @ parity_gradient
                        @ prefix_products[sublayer_index],
                        dtype=np.complex128,
                    )
                )
        return total_unitary, gradients

    def _initialize_parameters(self, parameters: Iterable[float] | None) -> RealArray:
        if parameters is None:
            return np.zeros(self.parameter_count, dtype=np.float64)
        return self._validate_parameters(parameters)

    def _validate_parameters(self, parameters: Iterable[float]) -> RealArray:
        parameter_array = np.asarray(list(parameters), dtype=np.float64)
        if parameter_array.shape != (self.parameter_count,):
            raise ValueError(
                f"Expected {self.parameter_count} parameters, got shape {parameter_array.shape}"
            )
        return parameter_array

    def _sublayer_unitary(
        self,
        parity: str,
        parity_parameters: RealArray,
    ) -> ComplexArray:
        return self._sublayer_unitary_and_gradients(parity, parity_parameters)[0]

    def _sublayer_unitary_and_gradients(
        self,
        parity: str,
        parity_parameters: RealArray,
    ) -> tuple[ComplexArray, list[ComplexArray]]:
        pair_starts = {left_site for left_site, _ in self.pairs_for_parity(parity)}
        local_factors: list[ComplexArray] = []
        local_derivatives: list[tuple[ComplexArray, ComplexArray, ComplexArray] | None] = []

        pair_index = 0
        site = 0
        while site < self.config.num_qubits:
            if site in pair_starts:
                if self.config.shared_parameter:
                    gate_parameters = parity_parameters[:3]
                else:
                    gate_parameters = parity_parameters[3 * pair_index : 3 * (pair_index + 1)]
                gate, gate_derivatives = self.gate_and_derivatives(
                    float(gate_parameters[0]),
                    float(gate_parameters[1]),
                    float(gate_parameters[2]),
                )
                local_factors.append(gate)
                local_derivatives.append(gate_derivatives)
                pair_index += 1
                site += 2
            else:
                local_factors.append(IDENTITY)
                local_derivatives.append(None)
                site += 1

        unitary = kron_all(local_factors)
        gradients: list[ComplexArray] = []

        if self.config.shared_parameter:
            shared_gradients = [np.zeros_like(unitary) for _ in range(3)]
            for factor_index, derivatives in enumerate(local_derivatives):
                if derivatives is None:
                    continue
                for axis_index, axis_derivative in enumerate(derivatives):
                    derivative_factors = list(local_factors)
                    derivative_factors[factor_index] = axis_derivative
                    shared_gradients[axis_index] += kron_all(derivative_factors)
            return unitary, [np.asarray(gradient, dtype=np.complex128) for gradient in shared_gradients]

        for factor_index, derivatives in enumerate(local_derivatives):
            if derivatives is None:
                continue
            for axis_derivative in derivatives:
                derivative_factors = list(local_factors)
                derivative_factors[factor_index] = axis_derivative
                gradients.append(np.asarray(kron_all(derivative_factors), dtype=np.complex128))
        return unitary, gradients
