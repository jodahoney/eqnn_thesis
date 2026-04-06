"""Qiskit-backed pure-state parity backend for small QCNN simulations."""

from __future__ import annotations

import numpy as np

from eqnn.backends.base import BackendCompatibleQCNN
from eqnn.circuits.qiskit_builders import (
    QISKIT_AVAILABLE,
    build_convolution_circuit,
    qiskit_density_to_repo,
    repo_density_to_qiskit,
)
from eqnn.types import ComplexArray

try:  # pragma: no cover - exercised when qiskit is installed
    from qiskit.quantum_info import DensityMatrix
except ImportError:  # pragma: no cover - local environments may not have qiskit
    DensityMatrix = None  # type: ignore[assignment]


class QiskitPureStateBackend:
    """Noiseless Qiskit backend for small pure-state parity checks."""

    def __init__(self) -> None:
        if not QISKIT_AVAILABLE or DensityMatrix is None:  # pragma: no cover - guarded in tests
            raise ImportError(
                "QiskitPureStateBackend requires the optional 'qiskit' dependency. "
                "Install it with `pip install 'eqnn-simulator[qiskit]'`."
            )

    @property
    def supports_exact_gradients(self) -> bool:
        return False

    def forward(
        self,
        model: BackendCompatibleQCNN,
        state: ComplexArray,
        parameters: np.ndarray,
    ):
        parameter_array = np.asarray(parameters, dtype=np.float64)
        final_density, final_num_qubits = self._forward_density(model, state, parameter_array)
        forward_pass = model.finalize_forward_pass(
            final_density,
            final_num_qubits,
            parameter_array[model.readout_slice],
        )
        return self._postprocess_forward_pass(forward_pass)

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
        del model, states, labels, parameters, loss_name, finite_difference_eps
        raise NotImplementedError("Qiskit pure-state gradients currently use trainer-side finite differences")

    def _forward_density(
        self,
        model: BackendCompatibleQCNN,
        state: ComplexArray,
        parameters: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        current_density = self._coerce_density_matrix(state)
        current_num_qubits = int(model.block_num_qubits[0])

        for block_index, convolution in enumerate(model.convolutions):
            convolution_parameters = parameters[model.convolution_slices[block_index]]
            current_density = self._apply_convolution(
                convolution,
                current_density,
                convolution_parameters,
            )
            current_density = self._apply_post_convolution_noise(
                convolution,
                current_density,
                current_num_qubits,
            )

            if block_index < len(model.poolings):
                pooling = model.poolings[block_index]
                pooling_parameters = parameters[model.pooling_slices[block_index]]
                current_density = np.asarray(
                    pooling.apply(current_density, parameters=pooling_parameters),
                    dtype=np.complex128,
                )
                current_num_qubits = int(pooling.output_num_qubits)

        return current_density, current_num_qubits

    def _apply_convolution(
        self,
        convolution: object,
        density_matrix: np.ndarray,
        parameters: np.ndarray,
    ) -> np.ndarray:
        num_qubits = int(convolution.config.num_qubits)
        circuit = build_convolution_circuit(convolution, parameters)
        qiskit_density = DensityMatrix(repo_density_to_qiskit(density_matrix, num_qubits))
        evolved = qiskit_density.evolve(circuit)
        return np.asarray(qiskit_density_to_repo(np.asarray(evolved.data), num_qubits), dtype=np.complex128)

    def _apply_post_convolution_noise(
        self,
        convolution: object,
        density_matrix: np.ndarray,
        num_qubits: int,
    ) -> np.ndarray:
        del convolution, num_qubits
        return np.asarray(density_matrix, dtype=np.complex128)

    def _coerce_density_matrix(self, state: ComplexArray) -> np.ndarray:
        state_array = np.asarray(state, dtype=np.complex128)
        if state_array.ndim == 1:
            return np.outer(state_array, np.conjugate(state_array))
        if state_array.ndim == 2 and state_array.shape[0] == state_array.shape[1]:
            return np.asarray(state_array, dtype=np.complex128)
        raise ValueError("State must be either a statevector or a square density matrix")

    def _postprocess_forward_pass(self, forward_pass: object) -> object:
        return forward_pass
