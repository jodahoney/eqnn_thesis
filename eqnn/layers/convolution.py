"""SU(2)-equivariant convolution layers built from exp(-i theta SWAP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from eqnn.physics.spin import IDENTITY, kron_all
from eqnn.types import ComplexArray, RealArray


@dataclass(frozen=True)
class SU2SwapConvolutionConfig:
    num_qubits: int
    parity_sequence: tuple[str, ...] = ("even", "odd")
    shared_parameter: bool = True

    def __post_init__(self) -> None:
        if self.num_qubits < 2:
            raise ValueError("num_qubits must be at least 2")
        for parity in self.parity_sequence:
            if parity not in {"even", "odd"}:
                raise ValueError("parity_sequence entries must be 'even' or 'odd'")


class SU2SwapConvolution:
    """Brickwork layer of two-qubit SU(2)-equivariant SWAP rotations."""

    def __init__(
        self,
        config: SU2SwapConvolutionConfig,
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
    def swap_operator() -> ComplexArray:
        """Return the two-qubit SWAP operator."""

        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.complex128,
        )

    @classmethod
    def gate(cls, theta: float) -> ComplexArray:
        """Return exp(-i theta SWAP)."""

        swap = cls.swap_operator()
        identity = np.eye(4, dtype=np.complex128)
        return np.asarray(
            np.cos(theta) * identity - 1.0j * np.sin(theta) * swap,
            dtype=np.complex128,
        )

    @classmethod
    def gate_derivative(cls, theta: float) -> ComplexArray:
        """Return d/dtheta exp(-i theta SWAP)."""

        swap = cls.swap_operator()
        identity = np.eye(4, dtype=np.complex128)
        return np.asarray(
            -np.sin(theta) * identity - 1.0j * np.cos(theta) * swap,
            dtype=np.complex128,
        )

    @property
    def parameter_count(self) -> int:
        if self.config.shared_parameter:
            return len(self.active_parities())
        return sum(len(self.pairs_for_parity(parity)) for parity in self.config.parity_sequence)

    def get_parameters(self) -> RealArray:
        return self.parameters.copy()

    def set_parameters(self, parameters: Iterable[float]) -> None:
        self.parameters = self._validate_parameters(parameters)

    def pairs_for_parity(self, parity: str) -> tuple[tuple[int, int], ...]:
        """Return the disjoint adjacent pairs used by a brickwork sublayer."""

        start = 0 if parity == "even" else 1
        return tuple(
            (site, site + 1)
            for site in range(start, self.config.num_qubits - 1, 2)
        )

    def active_parities(self) -> tuple[str, ...]:
        """Return the parities that contribute at least one gate."""

        return tuple(
            parity for parity in self.config.parity_sequence if self.pairs_for_parity(parity)
        )

    def unitary(self, parameters: Iterable[float] | None = None) -> ComplexArray:
        """Return the full-system unitary for the convolution block."""

        return self.unitary_and_gradients(parameters=parameters)[0]

    def unitary_and_gradients(
        self,
        parameters: Iterable[float] | None = None,
    ) -> tuple[ComplexArray, list[ComplexArray]]:
        """Return the full-system unitary and its parameter derivatives."""

        parameter_array = self.parameters if parameters is None else self._validate_parameters(parameters)

        sublayer_unitaries: list[ComplexArray] = []
        sublayer_gradients: list[list[ComplexArray]] = []
        parameter_offset = 0
        for parity in self.active_parities():
            pairs = self.pairs_for_parity(parity)
            if self.config.shared_parameter:
                parity_parameters = parameter_array[parameter_offset : parameter_offset + 1]
            else:
                parity_parameters = parameter_array[parameter_offset : parameter_offset + len(pairs)]
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
        gate_factor_indices: list[int] = []
        gate_angles: list[float] = []

        pair_index = 0
        site = 0
        while site < self.config.num_qubits:
            if site in pair_starts:
                theta = (
                    float(parity_parameters[0])
                    if self.config.shared_parameter
                    else float(parity_parameters[pair_index])
                )
                local_factors.append(self.gate(theta))
                gate_factor_indices.append(len(local_factors) - 1)
                gate_angles.append(theta)
                pair_index += 1
                site += 2
            else:
                local_factors.append(IDENTITY)
                site += 1

        unitary = kron_all(local_factors)
        if self.config.shared_parameter:
            derivative = np.zeros_like(unitary)
            for factor_index, theta in zip(gate_factor_indices, gate_angles):
                derivative_factors = list(local_factors)
                derivative_factors[factor_index] = self.gate_derivative(theta)
                derivative += kron_all(derivative_factors)
            return unitary, [np.asarray(derivative, dtype=np.complex128)]

        gradients: list[ComplexArray] = []
        for factor_index, theta in zip(gate_factor_indices, gate_angles):
            derivative_factors = list(local_factors)
            derivative_factors[factor_index] = self.gate_derivative(theta)
            gradients.append(np.asarray(kron_all(derivative_factors), dtype=np.complex128))
        return unitary, gradients
