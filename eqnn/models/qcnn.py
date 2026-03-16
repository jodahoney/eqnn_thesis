"""Full QCNN forward pass built from convolution, pooling, and invariant readout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from eqnn.layers.convolution import SU2SwapConvolution, SU2SwapConvolutionConfig
from eqnn.layers.pooling import PartialTracePooling, PartialTracePoolingConfig
from eqnn.physics.observables import alternating_singlet_means, dimerization_feature
from eqnn.physics.quantum import as_density_matrix
from eqnn.types import ComplexArray


@dataclass(frozen=True)
class QCNNConfig:
    num_qubits: int
    min_readout_qubits: int | None = None
    boundary: str = "open"
    parity_sequence: tuple[str, ...] = ("even", "odd")
    shared_convolution_parameter: bool = True
    pooling_keep: str = "left"
    symmetry_group: str = "SU(2)"

    def __post_init__(self) -> None:
        min_readout_qubits = (
            min(4, self.num_qubits)
            if self.min_readout_qubits is None
            else self.min_readout_qubits
        )
        object.__setattr__(self, "min_readout_qubits", min_readout_qubits)

        if self.num_qubits < 2:
            raise ValueError("num_qubits must be at least 2")
        if min_readout_qubits < 2 or min_readout_qubits > self.num_qubits:
            raise ValueError("min_readout_qubits must lie in [2, num_qubits]")
        if self.boundary not in {"open", "periodic"}:
            raise ValueError("boundary must be 'open' or 'periodic'")


@dataclass(frozen=True)
class QCNNForwardPass:
    """Interpretable outputs from the QCNN forward pass."""

    final_density_matrix: ComplexArray
    final_num_qubits: int
    primary_singlet_mean: float
    secondary_singlet_mean: float
    dimerization_feature: float
    logit: float
    probability: float


class SU2QCNN:
    """End-to-end SU(2)-equivariant QCNN simulator."""

    def __init__(
        self,
        config: QCNNConfig,
        parameters: Iterable[float] | None = None,
    ) -> None:
        self.config = config
        self.block_num_qubits = self._build_block_num_qubits()
        self.convolutions = [
            SU2SwapConvolution(
                SU2SwapConvolutionConfig(
                    num_qubits=num_qubits,
                    parity_sequence=self.config.parity_sequence,
                    shared_parameter=self.config.shared_convolution_parameter,
                )
            )
            for num_qubits in self.block_num_qubits
        ]
        self.poolings = [
            PartialTracePooling(
                PartialTracePoolingConfig(num_qubits=num_qubits, keep=self.config.pooling_keep)
            )
            for num_qubits in self.block_num_qubits[:-1]
        ]
        self._convolution_slices = self._build_convolution_slices()
        self.parameters = self._initialize_parameters(parameters)

    @property
    def parameter_count(self) -> int:
        return self._convolution_slices[-1].stop + 2

    def get_parameters(self) -> np.ndarray:
        return self.parameters.copy()

    def set_parameters(self, parameters: Iterable[float]) -> None:
        self.parameters = self._validate_parameters(parameters)

    def predict(self, state: ComplexArray, parameters: Iterable[float] | None = None) -> float:
        return self.forward(state, parameters=parameters).probability

    def predict_batch(
        self,
        states: ComplexArray,
        parameters: Iterable[float] | None = None,
    ) -> np.ndarray:
        return np.asarray(
            [self.predict(state, parameters=parameters) for state in states],
            dtype=np.float64,
        )

    def binary_cross_entropy(
        self,
        states: ComplexArray,
        labels: np.ndarray,
        parameters: Iterable[float] | None = None,
    ) -> float:
        probabilities = np.clip(self.predict_batch(states, parameters=parameters), 1e-8, 1.0 - 1e-8)
        labels_array = np.asarray(labels, dtype=np.float64)
        loss = -np.mean(
            labels_array * np.log(probabilities) + (1.0 - labels_array) * np.log(1.0 - probabilities)
        )
        return float(loss)

    def forward(
        self,
        state: ComplexArray,
        parameters: Iterable[float] | None = None,
    ) -> QCNNForwardPass:
        parameter_array = self.parameters if parameters is None else self._validate_parameters(parameters)

        current_density = as_density_matrix(state)

        for block_index, convolution in enumerate(self.convolutions):
            convolution_parameters = parameter_array[self._convolution_slices[block_index]]
            current_density = convolution.apply(current_density, parameters=convolution_parameters)
            if block_index < len(self.poolings):
                current_density = self.poolings[block_index](current_density)

        return self._finalize_forward_pass(
            current_density,
            self.block_num_qubits[-1],
            parameter_array[-2],
            parameter_array[-1],
        )

    def _build_block_num_qubits(self) -> list[int]:
        block_num_qubits: list[int] = []
        current_num_qubits = self.config.num_qubits

        while True:
            block_num_qubits.append(current_num_qubits)
            if current_num_qubits <= int(self.config.min_readout_qubits):
                break
            current_num_qubits = (current_num_qubits + 1) // 2
        return block_num_qubits

    def _build_convolution_slices(self) -> list[slice]:
        slices: list[slice] = []
        start = 0
        for convolution in self.convolutions:
            stop = start + convolution.parameter_count
            slices.append(slice(start, stop))
            start = stop
        if not slices:
            slices.append(slice(0, 0))
        return slices

    def _initialize_parameters(self, parameters: Iterable[float] | None) -> np.ndarray:
        if parameters is None:
            return np.zeros(self.parameter_count, dtype=np.float64)
        return self._validate_parameters(parameters)

    def _validate_parameters(self, parameters: Iterable[float]) -> np.ndarray:
        parameter_array = np.asarray(list(parameters), dtype=np.float64)
        if parameter_array.shape != (self.parameter_count,):
            raise ValueError(
                f"Expected {self.parameter_count} parameters, got shape {parameter_array.shape}"
            )
        return parameter_array

    def _finalize_forward_pass(
        self,
        density_matrix: ComplexArray,
        num_qubits: int,
        readout_weight: float,
        readout_bias: float,
    ) -> QCNNForwardPass:
        primary_mean, secondary_mean = alternating_singlet_means(
            density_matrix,
            num_qubits,
            boundary=self.config.boundary,
        )
        feature = dimerization_feature(
            density_matrix,
            num_qubits,
            boundary=self.config.boundary,
        )
        logit = float(readout_weight * feature + readout_bias)
        probability = float(1.0 / (1.0 + np.exp(-logit)))

        return QCNNForwardPass(
            final_density_matrix=density_matrix,
            final_num_qubits=num_qubits,
            primary_singlet_mean=primary_mean,
            secondary_singlet_mean=secondary_mean,
            dimerization_feature=feature,
            logit=logit,
            probability=probability,
        )
