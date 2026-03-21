"""Full QCNN forward pass built from convolution, pooling, and invariant readout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from eqnn.layers.convolution import SU2SwapConvolution, SU2SwapConvolutionConfig
from eqnn.layers.pooling import (
    PartialTracePooling,
    PartialTracePoolingConfig,
    SU2EquivariantPooling,
    SU2EquivariantPoolingConfig,
)
from eqnn.physics.heisenberg import alternating_bond_groups
from eqnn.physics.observables import (
    SINGLET_PROJECTOR,
    SWAP_OPERATOR,
    alternating_singlet_means,
    dimerization_feature,
    swap_expectation,
)
from eqnn.physics.quantum import as_density_matrix, embed_operator_on_sites
from eqnn.types import ComplexArray


@dataclass(frozen=True)
class QCNNConfig:
    num_qubits: int
    min_readout_qubits: int | None = None
    boundary: str = "open"
    parity_sequence: tuple[str, ...] = ("even", "odd")
    shared_convolution_parameter: bool = True
    pooling_mode: str = "partial_trace"
    pooling_keep: str = "left"
    readout_mode: str = "swap"
    symmetry_group: str = "SU(2)"

    def __post_init__(self) -> None:
        if self.readout_mode not in {"swap", "dimerization"}:
            raise ValueError("readout_mode must be 'swap' or 'dimerization'")
        min_readout_qubits = (
            (2 if self.readout_mode == "swap" else min(4, self.num_qubits))
            if self.min_readout_qubits is None
            else self.min_readout_qubits
        )
        object.__setattr__(self, "min_readout_qubits", min_readout_qubits)

        if self.num_qubits < 2:
            raise ValueError("num_qubits must be at least 2")
        if min_readout_qubits < 2 or min_readout_qubits > self.num_qubits:
            raise ValueError("min_readout_qubits must lie in [2, num_qubits]")
        if self.readout_mode == "swap" and min_readout_qubits != 2:
            raise ValueError("swap readout requires min_readout_qubits=2")
        if self.boundary not in {"open", "periodic"}:
            raise ValueError("boundary must be 'open' or 'periodic'")
        if self.pooling_mode not in {"partial_trace", "equivariant"}:
            raise ValueError("pooling_mode must be 'partial_trace' or 'equivariant'")


@dataclass(frozen=True)
class QCNNForwardPass:
    """Interpretable outputs from the QCNN forward pass."""

    final_density_matrix: ComplexArray
    final_num_qubits: int
    readout_mode: str
    primary_singlet_mean: float
    secondary_singlet_mean: float
    dimerization_feature: float
    swap_expectation: float | None
    logit: float | None
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
            self._build_pooling(num_qubits)
            for num_qubits in self.block_num_qubits[:-1]
        ]
        self._convolution_slices = self._build_convolution_slices()
        self._pooling_slices = self._build_pooling_slices()
        self._readout_slice = self._build_readout_slice()
        self.parameters = self._initialize_parameters(parameters)
        self.classification_threshold = 0.5

    @property
    def parameter_count(self) -> int:
        return self._readout_slice.stop

    def get_parameters(self) -> np.ndarray:
        return self.parameters.copy()

    def set_parameters(self, parameters: Iterable[float]) -> None:
        self.parameters = self._validate_parameters(parameters)

    def get_classification_threshold(self) -> float:
        return float(self.classification_threshold)

    def set_classification_threshold(self, threshold: float) -> None:
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("classification_threshold must lie in [0, 1]")
        self.classification_threshold = float(threshold)

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

    def predict_labels_batch(
        self,
        states: ComplexArray,
        parameters: Iterable[float] | None = None,
        *,
        threshold: float | None = None,
    ) -> np.ndarray:
        threshold_value = self.classification_threshold if threshold is None else float(threshold)
        return (self.predict_batch(states, parameters=parameters) >= threshold_value).astype(np.int64)

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

    def mean_squared_error(
        self,
        states: ComplexArray,
        labels: np.ndarray,
        parameters: Iterable[float] | None = None,
    ) -> float:
        probabilities = self.predict_batch(states, parameters=parameters)
        labels_array = np.asarray(labels, dtype=np.float64)
        return float(np.mean((probabilities - labels_array) ** 2))

    def loss(
        self,
        states: ComplexArray,
        labels: np.ndarray,
        parameters: Iterable[float] | None = None,
        *,
        loss_name: str = "bce",
    ) -> float:
        if loss_name == "mse":
            return self.mean_squared_error(states, labels, parameters=parameters)
        if loss_name != "bce":
            raise ValueError("loss_name must be 'bce' or 'mse'")
        return self.binary_cross_entropy(states, labels, parameters=parameters)

    def loss_gradient(
        self,
        states: ComplexArray,
        labels: np.ndarray,
        parameters: Iterable[float] | None = None,
        *,
        finite_difference_eps: float = 1e-3,
        loss_name: str = "bce",
    ) -> np.ndarray:
        if loss_name not in {"bce", "mse"}:
            raise ValueError("loss_name must be 'bce' or 'mse'")
        parameter_array = self.parameters if parameters is None else self._validate_parameters(parameters)
        states_array = np.asarray(states, dtype=np.complex128)
        labels_array = np.asarray(labels, dtype=np.float64)
        if states_array.ndim != 2:
            raise ValueError("states must have shape (num_examples, hilbert_dimension)")
        if labels_array.shape != (states_array.shape[0],):
            raise ValueError("labels must align with states")

        gradient = np.zeros_like(parameter_array)
        unsupported_slices = [
            np.arange(pooling_slice.start, pooling_slice.stop, dtype=np.int64)
            for pooling_slice, pooling in zip(self._pooling_slices, self.poolings)
            if pooling.parameter_count > 0 and not hasattr(pooling, "parameter_gradient")
        ]
        unsupported_pooling_indices = (
            np.concatenate(unsupported_slices).astype(np.int64, copy=False)
            if unsupported_slices
            else np.zeros(0, dtype=np.int64)
        )

        for state, label in zip(states_array, labels_array):
            gradient += self._sample_loss_gradient(
                state,
                float(label),
                parameter_array,
                unsupported_pooling_indices=unsupported_pooling_indices,
                loss_name=loss_name,
            )

        gradient /= float(labels_array.size)

        for index in unsupported_pooling_indices.tolist():
            offset = np.zeros_like(parameter_array)
            offset[index] = finite_difference_eps
            loss_plus = self.binary_cross_entropy(states_array, labels_array, parameters=parameter_array + offset)
            loss_minus = self.binary_cross_entropy(states_array, labels_array, parameters=parameter_array - offset)
            if loss_name == "mse":
                loss_plus = self.mean_squared_error(states_array, labels_array, parameters=parameter_array + offset)
                loss_minus = self.mean_squared_error(states_array, labels_array, parameters=parameter_array - offset)
            gradient[index] = (loss_plus - loss_minus) / (2.0 * finite_difference_eps)

        return np.asarray(gradient, dtype=np.float64)

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
                pooling = self.poolings[block_index]
                pooling_parameters = parameter_array[self._pooling_slices[block_index]]
                current_density = pooling.apply(current_density, parameters=pooling_parameters)

        return self._finalize_forward_pass(
            current_density,
            self.block_num_qubits[-1],
            parameter_array[self._readout_slice],
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
        return slices

    def _build_pooling_slices(self) -> list[slice]:
        slices: list[slice] = []
        start = self._convolution_slices[-1].stop if self._convolution_slices else 0
        for pooling in self.poolings:
            stop = start + pooling.parameter_count
            slices.append(slice(start, stop))
            start = stop
        return slices

    def _build_readout_slice(self) -> slice:
        start = self._pooling_slices[-1].stop if self._pooling_slices else (
            self._convolution_slices[-1].stop if self._convolution_slices else 0
        )
        return slice(start, start + self._readout_parameter_count())

    def _build_pooling(self, num_qubits: int) -> PartialTracePooling | SU2EquivariantPooling:
        if self.config.pooling_mode == "partial_trace":
            return PartialTracePooling(
                PartialTracePoolingConfig(num_qubits=num_qubits, keep=self.config.pooling_keep)
            )
        return SU2EquivariantPooling(
            SU2EquivariantPoolingConfig(
                num_qubits=num_qubits,
                warm_start=self.config.pooling_keep,
            )
        )

    def _initialize_parameters(self, parameters: Iterable[float] | None) -> np.ndarray:
        if parameters is None:
            default_parameters = [convolution.get_parameters() for convolution in self.convolutions]
            default_parameters.extend(
                np.asarray(pooling.parameters, dtype=np.float64).copy() for pooling in self.poolings
            )
            default_parameters.append(np.zeros(self._readout_parameter_count(), dtype=np.float64))
            return np.asarray(np.concatenate(default_parameters), dtype=np.float64)
        return self._validate_parameters(parameters)

    def _readout_parameter_count(self) -> int:
        if self.config.readout_mode == "swap":
            return 0
        return 2

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
        readout_parameters: np.ndarray,
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
        if self.config.readout_mode == "swap":
            if num_qubits != 2:
                raise ValueError("swap readout expects the QCNN to terminate on exactly 2 qubits")
            swap_value = swap_expectation(density_matrix)
            logit = None
            probability = float(np.clip(0.5 * (swap_value + 1.0), 0.0, 1.0))
        else:
            if readout_parameters.shape != (2,):
                raise ValueError("dimerization readout expects exactly two readout parameters")
            readout_weight = float(readout_parameters[0])
            readout_bias = float(readout_parameters[1])
            swap_value = None
            logit = float(readout_weight * feature + readout_bias)
            probability = float(1.0 / (1.0 + np.exp(-logit)))

        return QCNNForwardPass(
            final_density_matrix=density_matrix,
            final_num_qubits=num_qubits,
            readout_mode=self.config.readout_mode,
            primary_singlet_mean=primary_mean,
            secondary_singlet_mean=secondary_mean,
            dimerization_feature=feature,
            swap_expectation=swap_value,
            logit=logit,
            probability=probability,
        )

    def _sample_loss_gradient(
        self,
        state: ComplexArray,
        label: float,
        parameter_array: np.ndarray,
        *,
        unsupported_pooling_indices: np.ndarray,
        loss_name: str,
    ) -> np.ndarray:
        final_density, caches = self._forward_with_cache(state, parameter_array)
        adjoint, readout_gradient = self._readout_loss_gradient(
            final_density,
            self.block_num_qubits[-1],
            parameter_array[self._readout_slice],
            label,
            loss_name=loss_name,
        )

        sample_gradient = np.zeros_like(parameter_array)
        if self._readout_slice.stop > self._readout_slice.start:
            sample_gradient[self._readout_slice] = readout_gradient

        unsupported_index_set = set(int(index) for index in unsupported_pooling_indices.tolist())

        for block_index in range(len(caches) - 1, -1, -1):
            cache = caches[block_index]
            pooling = cache["pooling"]
            if pooling is not None:
                pooling_slice = self._pooling_slices[block_index]
                if pooling_slice.stop > pooling_slice.start and hasattr(pooling, "parameter_gradient"):
                    sample_gradient[pooling_slice] = np.asarray(
                        pooling.parameter_gradient(
                            cache["density_after_convolution"],
                            adjoint,
                            cache["pooling_parameters"],
                        ),
                        dtype=np.float64,
                    )
                adjoint = self._apply_pooling_adjoint(
                    pooling,
                    adjoint,
                    cache["pooling_parameters"],
                )

            density_in = cache["density_in"]
            unitary = cache["unitary"]
            unitary_dagger = unitary.conjugate().T
            convolution_slice = self._convolution_slices[block_index]
            for local_index, derivative_unitary in enumerate(cache["unitary_gradients"]):
                parameter_index = convolution_slice.start + local_index
                if parameter_index in unsupported_index_set:
                    continue
                density_derivative = (
                    derivative_unitary @ density_in @ unitary_dagger
                    + unitary @ density_in @ derivative_unitary.conjugate().T
                )
                sample_gradient[parameter_index] = float(
                    np.real_if_close(np.trace(adjoint @ density_derivative))
                )

            adjoint = unitary_dagger @ adjoint @ unitary

        return sample_gradient

    def _forward_with_cache(
        self,
        state: ComplexArray,
        parameter_array: np.ndarray,
    ) -> tuple[ComplexArray, list[dict[str, object]]]:
        current_density = as_density_matrix(state)
        caches: list[dict[str, object]] = []

        for block_index, convolution in enumerate(self.convolutions):
            if not hasattr(convolution, "unitary_and_gradients"):
                raise NotImplementedError("Convolution layer does not expose exact unitary derivatives")

            density_in = current_density
            convolution_parameters = parameter_array[self._convolution_slices[block_index]]
            unitary, unitary_gradients = convolution.unitary_and_gradients(parameters=convolution_parameters)
            current_density = unitary @ density_in @ unitary.conjugate().T

            cache: dict[str, object] = {
                "density_in": density_in,
                "density_after_convolution": current_density,
                "unitary": unitary,
                "unitary_gradients": unitary_gradients,
                "pooling": None,
                "pooling_parameters": np.zeros(0, dtype=np.float64),
            }

            if block_index < len(self.poolings):
                pooling = self.poolings[block_index]
                pooling_parameters = parameter_array[self._pooling_slices[block_index]]
                current_density = pooling.apply(current_density, parameters=pooling_parameters)
                cache["pooling"] = pooling
                cache["pooling_parameters"] = pooling_parameters

            caches.append(cache)

        return np.asarray(current_density, dtype=np.complex128), caches

    def _readout_loss_gradient(
        self,
        density_matrix: ComplexArray,
        num_qubits: int,
        readout_parameters: np.ndarray,
        label: float,
        *,
        loss_name: str,
    ) -> tuple[ComplexArray, np.ndarray]:
        if loss_name not in {"bce", "mse"}:
            raise ValueError("loss_name must be 'bce' or 'mse'")
        probability_floor = 1e-8
        if self.config.readout_mode == "swap":
            probability = float(np.clip(0.5 * (swap_expectation(density_matrix) + 1.0), 0.0, 1.0))
            if loss_name == "mse":
                loss_probability_gradient = 2.0 * (probability - label)
            else:
                clipped_probability = float(np.clip(probability, probability_floor, 1.0 - probability_floor))
                if probability <= probability_floor or probability >= 1.0 - probability_floor:
                    loss_probability_gradient = 0.0
                else:
                    loss_probability_gradient = float(
                        (clipped_probability - label) / (clipped_probability * (1.0 - clipped_probability))
                    )
            observable_gradient = 0.5 * loss_probability_gradient * SWAP_OPERATOR
            return np.asarray(observable_gradient, dtype=np.complex128), np.zeros(0, dtype=np.float64)

        feature_operator = self._dimerization_operator(num_qubits)
        feature_value = float(np.real_if_close(np.trace(density_matrix @ feature_operator)))
        readout_weight = float(readout_parameters[0])
        readout_bias = float(readout_parameters[1])
        logit = float(readout_weight * feature_value + readout_bias)
        probability = float(1.0 / (1.0 + np.exp(-logit)))
        if loss_name == "mse":
            loss_logit_gradient = 2.0 * (probability - label) * probability * (1.0 - probability)
        else:
            if probability <= probability_floor or probability >= 1.0 - probability_floor:
                loss_logit_gradient = 0.0
            else:
                loss_logit_gradient = probability - label

        observable_gradient = loss_logit_gradient * readout_weight * feature_operator
        readout_gradient = np.asarray(
            (loss_logit_gradient * feature_value, loss_logit_gradient),
            dtype=np.float64,
        )
        return np.asarray(observable_gradient, dtype=np.complex128), readout_gradient

    def _dimerization_operator(self, num_qubits: int) -> ComplexArray:
        primary_bonds, secondary_bonds = alternating_bond_groups(num_qubits, self.config.boundary)
        operator = np.zeros((1 << num_qubits, 1 << num_qubits), dtype=np.complex128)

        if primary_bonds:
            primary_scale = -1.0 / float(len(primary_bonds))
            for bond in primary_bonds:
                operator += primary_scale * embed_operator_on_sites(
                    SINGLET_PROJECTOR,
                    num_qubits,
                    bond,
                )

        if secondary_bonds:
            secondary_scale = 1.0 / float(len(secondary_bonds))
            for bond in secondary_bonds:
                operator += secondary_scale * embed_operator_on_sites(
                    SINGLET_PROJECTOR,
                    num_qubits,
                    bond,
                )

        return np.asarray(operator, dtype=np.complex128)

    def _apply_pooling_adjoint(
        self,
        pooling: PartialTracePooling | SU2EquivariantPooling,
        observable: ComplexArray,
        pooling_parameters: np.ndarray,
    ) -> ComplexArray:
        if hasattr(pooling, "adjoint_apply"):
            return np.asarray(pooling.adjoint_apply(observable, parameters=pooling_parameters), dtype=np.complex128)
        raise NotImplementedError("Pooling layer does not expose an adjoint map")
