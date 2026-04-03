"""NumPy pure-state backend for current QCNN simulations."""

from __future__ import annotations

from typing import Any

import numpy as np

from eqnn.backends.base import BackendCompatibleQCNN
from eqnn.physics.quantum import as_density_matrix
from eqnn.types import ComplexArray


class NumpyPureStateBackend:
    """Reference NumPy backend for pure-state QCNN simulations."""

    @property
    def supports_exact_gradients(self) -> bool:
        return True

    def forward(
        self,
        model: BackendCompatibleQCNN,
        state: ComplexArray,
        parameters: np.ndarray,
    ):
        parameter_array = np.asarray(parameters, dtype=np.float64)
        current_density = as_density_matrix(state)

        for block_index, convolution in enumerate(model.convolutions):
            convolution_parameters = parameter_array[model.convolution_slices[block_index]]
            current_density = convolution.apply(current_density, parameters=convolution_parameters)
            if block_index < len(model.poolings):
                pooling = model.poolings[block_index]
                pooling_parameters = parameter_array[model.pooling_slices[block_index]]
                current_density = pooling.apply(current_density, parameters=pooling_parameters)

        return model.finalize_forward_pass(
            np.asarray(current_density, dtype=np.complex128),
            int(model.block_num_qubits[-1]),
            parameter_array[model.readout_slice],
        )

    def predict_batch(
        self,
        model: BackendCompatibleQCNN,
        states: ComplexArray,
        parameters: np.ndarray,
    ) -> np.ndarray:
        states_array = np.asarray(states, dtype=np.complex128)
        if states_array.ndim != 2:
            raise ValueError("states must have shape (num_examples, hilbert_dimension)")
        parameter_array = np.asarray(parameters, dtype=np.float64)
        return np.asarray(
            [self.forward(model, state, parameter_array).probability for state in states_array],
            dtype=np.float64,
        )

    def evaluate_batch(
        self,
        model: BackendCompatibleQCNN,
        states: ComplexArray,
        labels: np.ndarray,
        parameters: np.ndarray,
        *,
        loss_name: str,
        threshold: float,
    ) -> dict[str, np.ndarray | float]:
        probabilities = self.predict_batch(model, states, parameters)
        labels_array = np.asarray(labels, dtype=np.float64)
        predictions = (probabilities >= float(threshold)).astype(np.int64)
        accuracy = float(np.mean(predictions == labels_array.astype(np.int64)))

        if loss_name == "mse":
            loss = float(np.mean((probabilities - labels_array) ** 2))
        elif loss_name == "bce":
            clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
            loss = float(
                -np.mean(labels_array * np.log(clipped) + (1.0 - labels_array) * np.log(1.0 - clipped))
            )
        else:
            raise ValueError("loss_name must be 'bce' or 'mse'")

        return {
            "probabilities": probabilities,
            "predictions": predictions,
            "loss": loss,
            "accuracy": accuracy,
        }

    def loss_gradient(
        self,
        model: BackendCompatibleQCNN,
        states: ComplexArray,
        labels: np.ndarray,
        parameters: np.ndarray,
        *,
        loss_name: str,
        finite_difference_eps: float = 1e-3,
    ) -> np.ndarray:
        if loss_name not in {"bce", "mse"}:
            raise ValueError("loss_name must be 'bce' or 'mse'")

        parameter_array = np.asarray(parameters, dtype=np.float64)
        states_array = np.asarray(states, dtype=np.complex128)
        labels_array = np.asarray(labels, dtype=np.float64)

        if states_array.ndim != 2:
            raise ValueError("states must have shape (num_examples, hilbert_dimension)")
        if labels_array.shape != (states_array.shape[0],):
            raise ValueError("labels must align with states")

        gradient = np.zeros_like(parameter_array)
        unsupported_pooling_indices = self._unsupported_pooling_indices(model)

        for state, label in zip(states_array, labels_array):
            gradient += self._sample_loss_gradient(
                model,
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
            loss_plus = model.loss(
                states_array,
                labels_array,
                parameters=parameter_array + offset,
                loss_name=loss_name,
            )
            loss_minus = model.loss(
                states_array,
                labels_array,
                parameters=parameter_array - offset,
                loss_name=loss_name,
            )
            gradient[index] = (loss_plus - loss_minus) / (2.0 * finite_difference_eps)

        return np.asarray(gradient, dtype=np.float64)

    def _unsupported_pooling_indices(
        self,
        model: BackendCompatibleQCNN,
    ) -> np.ndarray:
        unsupported_slices = [
            np.arange(pooling_slice.start, pooling_slice.stop, dtype=np.int64)
            for pooling_slice, pooling in zip(model.pooling_slices, model.poolings)
            if getattr(pooling, "parameter_count", 0) > 0 and not hasattr(pooling, "parameter_gradient")
        ]
        if not unsupported_slices:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(unsupported_slices).astype(np.int64, copy=False)

    def _sample_loss_gradient(
        self,
        model: BackendCompatibleQCNN,
        state: ComplexArray,
        label: float,
        parameter_array: np.ndarray,
        *,
        unsupported_pooling_indices: np.ndarray,
        loss_name: str,
    ) -> np.ndarray:
        final_density, caches = self._forward_with_cache(model, state, parameter_array)
        adjoint, readout_gradient = model.readout_loss_gradient(
            final_density,
            int(model.block_num_qubits[-1]),
            parameter_array[model.readout_slice],
            label,
            loss_name=loss_name,
        )

        sample_gradient = np.zeros_like(parameter_array)
        if model.readout_slice.stop > model.readout_slice.start:
            sample_gradient[model.readout_slice] = readout_gradient

        unsupported_index_set = set(int(index) for index in unsupported_pooling_indices.tolist())

        for block_index in range(len(caches) - 1, -1, -1):
            cache = caches[block_index]
            pooling = cache["pooling"]
            if pooling is not None:
                pooling_slice = model.pooling_slices[block_index]
                if pooling_slice.stop > pooling_slice.start and hasattr(pooling, "parameter_gradient"):
                    sample_gradient[pooling_slice] = np.asarray(
                        pooling.parameter_gradient(
                            cache["density_after_convolution"],
                            adjoint,
                            cache["pooling_parameters"],
                        ),
                        dtype=np.float64,
                    )
                adjoint = model.apply_pooling_adjoint(
                    pooling,
                    adjoint,
                    cache["pooling_parameters"],
                )

            density_in = cache["density_in"]
            unitary = cache["unitary"]
            unitary_dagger = unitary.conjugate().T
            convolution_slice = model.convolution_slices[block_index]
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
        model: BackendCompatibleQCNN,
        state: ComplexArray,
        parameter_array: np.ndarray,
    ) -> tuple[ComplexArray, list[dict[str, Any]]]:
        current_density = as_density_matrix(state)
        caches: list[dict[str, Any]] = []

        for block_index, convolution in enumerate(model.convolutions):
            if not hasattr(convolution, "unitary_and_gradients"):
                raise NotImplementedError("Convolution layer does not expose exact unitary derivatives")

            density_in = current_density
            convolution_parameters = parameter_array[model.convolution_slices[block_index]]
            unitary, unitary_gradients = convolution.unitary_and_gradients(parameters=convolution_parameters)
            current_density = unitary @ density_in @ unitary.conjugate().T

            cache: dict[str, Any] = {
                "density_in": density_in,
                "density_after_convolution": current_density,
                "unitary": unitary,
                "unitary_gradients": unitary_gradients,
                "pooling": None,
                "pooling_parameters": np.zeros(0, dtype=np.float64),
            }

            if block_index < len(model.poolings):
                pooling = model.poolings[block_index]
                pooling_parameters = parameter_array[model.pooling_slices[block_index]]
                current_density = pooling.apply(current_density, parameters=pooling_parameters)
                cache["pooling"] = pooling
                cache["pooling_parameters"] = pooling_parameters

            caches.append(cache)

        return np.asarray(current_density, dtype=np.complex128), caches
