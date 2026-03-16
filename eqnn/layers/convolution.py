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

        parameter_array = self.parameters if parameters is None else self._validate_parameters(parameters)

        unitary = np.eye(1 << self.config.num_qubits, dtype=np.complex128)
        parameter_offset = 0
        for parity in self.active_parities():
            pairs = self.pairs_for_parity(parity)
            if self.config.shared_parameter:
                parity_parameters = parameter_array[parameter_offset : parameter_offset + 1]
            else:
                parity_parameters = parameter_array[parameter_offset : parameter_offset + len(pairs)]
            unitary = self._sublayer_unitary(parity, parity_parameters) @ unitary
            parameter_offset += len(parity_parameters)
        return unitary

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
        pair_starts = {left_site for left_site, _ in self.pairs_for_parity(parity)}
        local_factors: list[ComplexArray] = []

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
                pair_index += 1
                site += 2
            else:
                local_factors.append(IDENTITY)
                site += 1

        return kron_all(local_factors)
