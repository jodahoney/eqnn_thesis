"""HEA-inspired convolution layers for QCNN baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from eqnn.physics.spin import IDENTITY, kron_all
from eqnn.types import ComplexArray, RealArray

_CZ_GATE = np.diag((1.0, 1.0, 1.0, -1.0)).astype(np.complex128)


def _coerce_block_parameters(parameters: tuple[float, ...] | Iterable[float]) -> RealArray:
    if len(parameters) == 1 and not np.isscalar(parameters[0]):  # type: ignore[index]
        return np.asarray(list(parameters[0]), dtype=np.float64)  # type: ignore[index]
    return np.asarray(parameters, dtype=np.float64)


def _ry(theta: float) -> ComplexArray:
    half = 0.5 * theta
    return np.asarray(
        (
            (np.cos(half), -np.sin(half)),
            (np.sin(half), np.cos(half)),
        ),
        dtype=np.complex128,
    )


def _dry(theta: float) -> ComplexArray:
    half = 0.5 * theta
    return 0.5 * np.asarray(
        (
            (-np.sin(half), -np.cos(half)),
            (np.cos(half), -np.sin(half)),
        ),
        dtype=np.complex128,
    )


def _rz(theta: float) -> ComplexArray:
    half = 0.5 * theta
    return np.asarray(
        (
            (np.exp(-1.0j * half), 0.0),
            (0.0, np.exp(1.0j * half)),
        ),
        dtype=np.complex128,
    )


def _drz(theta: float) -> ComplexArray:
    half = 0.5 * theta
    return np.asarray(
        (
            (-0.5j * np.exp(-1.0j * half), 0.0),
            (0.0, 0.5j * np.exp(1.0j * half)),
        ),
        dtype=np.complex128,
    )


def _single_qubit_layer(theta_y: float, theta_z: float) -> tuple[ComplexArray, tuple[ComplexArray, ComplexArray]]:
    """Return the circuit-ordered single-qubit layer Ry -> Rz and its derivatives."""

    ry_gate = _ry(theta_y)
    rz_gate = _rz(theta_z)
    gate = np.asarray(rz_gate @ ry_gate, dtype=np.complex128)
    derivatives = (
        np.asarray(rz_gate @ _dry(theta_y), dtype=np.complex128),
        np.asarray(_drz(theta_z) @ ry_gate, dtype=np.complex128),
    )
    return gate, derivatives


@dataclass(frozen=True)
class HEAConvolutionConfig:
    num_qubits: int
    parity_sequence: tuple[str, ...] = ("even", "odd")
    shared_parameter: bool = True
    entangler: str = "cz"

    def __post_init__(self) -> None:
        if self.num_qubits < 2:
            raise ValueError("num_qubits must be at least 2")
        for parity in self.parity_sequence:
            if parity not in {"even", "odd"}:
                raise ValueError("parity_sequence entries must be 'even' or 'odd'")
        if self.entangler != "cz":
            raise ValueError("Only entangler='cz' is currently supported")


class HEAConvolution:
    """Brickwork convolution built from shallow two-qubit HEA blocks.

    Each active pair uses the local circuit

        (Ry Rz)^{⊗2} -> CZ -> (Ry Rz)^{⊗2},

    interpreted in circuit order as Ry then Rz on each qubit in both local
    layers. Each pair therefore carries eight trainable parameters.
    """

    def __init__(
        self,
        config: HEAConvolutionConfig,
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
    def entangler() -> ComplexArray:
        return _CZ_GATE.copy()

    @classmethod
    def gate(cls, *parameters: float) -> ComplexArray:
        gate, _ = cls.gate_and_derivatives(*parameters)
        return gate

    @classmethod
    def gate_and_derivatives(
        cls,
        *parameters: float,
    ) -> tuple[ComplexArray, tuple[ComplexArray, ...]]:
        parameter_array = _coerce_block_parameters(parameters)
        if parameter_array.shape != (8,):
            raise ValueError(f"Expected 8 HEA block parameters, got shape {parameter_array.shape}")

        (
            pre_q0_y,
            pre_q0_z,
            pre_q1_y,
            pre_q1_z,
            post_q0_y,
            post_q0_z,
            post_q1_y,
            post_q1_z,
        ) = parameter_array.tolist()

        pre_q0, (dpre_q0_y, dpre_q0_z) = _single_qubit_layer(pre_q0_y, pre_q0_z)
        pre_q1, (dpre_q1_y, dpre_q1_z) = _single_qubit_layer(pre_q1_y, pre_q1_z)
        post_q0, (dpost_q0_y, dpost_q0_z) = _single_qubit_layer(post_q0_y, post_q0_z)
        post_q1, (dpost_q1_y, dpost_q1_z) = _single_qubit_layer(post_q1_y, post_q1_z)

        pre_layer = kron_all([pre_q0, pre_q1])
        post_layer = kron_all([post_q0, post_q1])
        entangler = cls.entangler()
        gate = np.asarray(post_layer @ entangler @ pre_layer, dtype=np.complex128)

        pre_derivatives = (
            kron_all([dpre_q0_y, pre_q1]),
            kron_all([dpre_q0_z, pre_q1]),
            kron_all([pre_q0, dpre_q1_y]),
            kron_all([pre_q0, dpre_q1_z]),
        )
        post_derivatives = (
            kron_all([dpost_q0_y, post_q1]),
            kron_all([dpost_q0_z, post_q1]),
            kron_all([post_q0, dpost_q1_y]),
            kron_all([post_q0, dpost_q1_z]),
        )

        derivatives = tuple(
            np.asarray(post_layer @ entangler @ derivative, dtype=np.complex128)
            for derivative in pre_derivatives
        ) + tuple(
            np.asarray(derivative @ entangler @ pre_layer, dtype=np.complex128)
            for derivative in post_derivatives
        )
        return gate, derivatives

    @property
    def parameter_count(self) -> int:
        if self.config.shared_parameter:
            return 8 * len(self.active_parities())
        return 8 * sum(len(self.pairs_for_parity(parity)) for parity in self.config.parity_sequence)

    def get_parameters(self) -> RealArray:
        return self.parameters.copy()

    def set_parameters(self, parameters: Iterable[float]) -> None:
        self.parameters = self._validate_parameters(parameters)

    def pairs_for_parity(self, parity: str) -> tuple[tuple[int, int], ...]:
        start = 0 if parity == "even" else 1
        return tuple((site, site + 1) for site in range(start, self.config.num_qubits - 1, 2))

    def active_parities(self) -> tuple[str, ...]:
        return tuple(parity for parity in self.config.parity_sequence if self.pairs_for_parity(parity))

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
                parity_parameters = parameter_array[parameter_offset : parameter_offset + 8]
            else:
                parity_parameters = parameter_array[
                    parameter_offset : parameter_offset + 8 * len(pairs)
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

    def _sublayer_unitary_and_gradients(
        self,
        parity: str,
        parity_parameters: RealArray,
    ) -> tuple[ComplexArray, list[ComplexArray]]:
        pair_starts = {left_site for left_site, _ in self.pairs_for_parity(parity)}
        local_factors: list[ComplexArray] = []
        local_derivatives: list[tuple[ComplexArray, ...] | None] = []

        pair_index = 0
        site = 0
        while site < self.config.num_qubits:
            if site in pair_starts:
                if self.config.shared_parameter:
                    gate_parameters = parity_parameters[:8]
                else:
                    gate_parameters = parity_parameters[8 * pair_index : 8 * (pair_index + 1)]
                gate, gate_derivatives = self.gate_and_derivatives(gate_parameters)
                local_factors.append(gate)
                local_derivatives.append(gate_derivatives)
                pair_index += 1
                site += 2
            else:
                local_factors.append(IDENTITY)
                local_derivatives.append(None)
                site += 1

        unitary = kron_all(local_factors)

        if self.config.shared_parameter:
            shared_gradients = [np.zeros_like(unitary) for _ in range(8)]
            for factor_index, derivatives in enumerate(local_derivatives):
                if derivatives is None:
                    continue
                for parameter_index, derivative in enumerate(derivatives):
                    derivative_factors = list(local_factors)
                    derivative_factors[factor_index] = derivative
                    shared_gradients[parameter_index] += kron_all(derivative_factors)
            return unitary, [np.asarray(gradient, dtype=np.complex128) for gradient in shared_gradients]

        gradients: list[ComplexArray] = []
        for factor_index, derivatives in enumerate(local_derivatives):
            if derivatives is None:
                continue
            for derivative in derivatives:
                derivative_factors = list(local_factors)
                derivative_factors[factor_index] = derivative
                gradients.append(np.asarray(kron_all(derivative_factors), dtype=np.complex128))
        return unitary, gradients
