"""Torch-native pure-state backend for QCNN simulations."""

from __future__ import annotations

from typing import Any

import numpy as np

from eqnn.backends.base import BackendCompatibleQCNN
from eqnn.backends.torch_ops import (
    TORCH_AVAILABLE,
    as_density_matrix as as_torch_density_matrix,
    expectation_value as torch_expectation_value,
    kron_all as torch_kron_all,
    partial_trace_density_matrix as torch_partial_trace_density_matrix,
    statevectors_to_density_matrices,
)
from eqnn.layers import (
    AnisotropicConvolution,
    HEAConvolution,
    PartialTracePooling,
    SU2SwapConvolution,
)
from eqnn.types import ComplexArray

try:  # pragma: no cover - exercised indirectly when torch is installed
    import torch
except ImportError:  # pragma: no cover - local test environments may not have torch
    torch = None  # type: ignore[assignment]


class TorchPureStateBackend:
    """Pure-state QCNN backend implemented with Torch tensors and autograd."""

    def __init__(
        self,
        *,
        device: Any = "cpu",
        complex_dtype: Any | None = None,
    ) -> None:
        if torch is None:  # pragma: no cover - guarded by TORCH_AVAILABLE in tests
            raise ImportError(
                "TorchPureStateBackend requires the optional 'torch' dependency. "
                "Install it with `pip install 'eqnn-simulator[torch]'` or `pip install torch`."
            )
        self.device = torch.device(device)
        self.complex_dtype = torch.complex128 if complex_dtype is None else complex_dtype
        self.real_dtype = torch.float64 if self.complex_dtype == torch.complex128 else torch.float32
        self._operator_cache: dict[tuple[str, int], torch.Tensor] = {}

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
        parameter_tensor = self._parameter_tensor(parameter_array, requires_grad=False)
        state_tensor = self._complex_tensor(state)
        final_density, final_num_qubits = self._forward_density(
            model,
            state_tensor,
            parameter_tensor,
            exact_gradients_required=False,
            batched_statevectors=False,
        )
        return model.finalize_forward_pass(
            self._to_numpy(final_density),
            final_num_qubits,
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

        with torch.no_grad():
            parameter_tensor = self._parameter_tensor(parameters, requires_grad=False)
            state_tensor = self._complex_tensor(states_array)
            probabilities = self.predict_batch_tensor(model, state_tensor, parameter_tensor)
        return np.asarray(probabilities.detach().cpu().numpy(), dtype=np.float64)

    def predict_batch_tensor(
        self,
        model: BackendCompatibleQCNN,
        state_tensor: "torch.Tensor",
        parameter_tensor: "torch.Tensor",
    ) -> "torch.Tensor":
        return self._probabilities_from_states(
            model,
            state_tensor,
            parameter_tensor,
            exact_gradients_required=False,
            batched_statevectors=True,
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
        states_array = np.asarray(states, dtype=np.complex128)
        labels_array = np.asarray(labels, dtype=np.float64)
        if states_array.ndim != 2:
            raise ValueError("states must have shape (num_examples, hilbert_dimension)")
        if labels_array.shape != (states_array.shape[0],):
            raise ValueError("labels must align with states")

        with torch.no_grad():
            parameter_tensor = self._parameter_tensor(parameters, requires_grad=False)
            state_tensor = self._complex_tensor(states_array)
            label_tensor = torch.as_tensor(labels_array, dtype=self.real_dtype, device=self.device)
            probabilities = self.predict_batch_tensor(model, state_tensor, parameter_tensor)
            if loss_name == "mse":
                loss_tensor = torch.mean((probabilities - label_tensor) ** 2)
            elif loss_name == "bce":
                clipped = torch.clamp(probabilities, 1e-12, 1.0 - 1e-12)
                loss_tensor = -torch.mean(
                    label_tensor * torch.log(clipped) + (1.0 - label_tensor) * torch.log(1.0 - clipped)
                )
            else:
                raise ValueError("loss_name must be 'bce' or 'mse'")

            prediction_tensor = (probabilities >= float(threshold)).to(dtype=torch.int64)
            accuracy_tensor = torch.mean(
                (prediction_tensor == label_tensor.to(dtype=torch.int64)).to(dtype=self.real_dtype)
            )

        return {
            "probabilities": np.asarray(probabilities.detach().cpu().numpy(), dtype=np.float64),
            "predictions": np.asarray(prediction_tensor.detach().cpu().numpy(), dtype=np.int64),
            "loss": float(loss_tensor.detach().cpu().item()),
            "accuracy": float(accuracy_tensor.detach().cpu().item()),
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
        del finite_difference_eps
        if loss_name not in {"bce", "mse"}:
            raise ValueError("loss_name must be 'bce' or 'mse'")
        if any(not isinstance(pooling, PartialTracePooling) for pooling in model.poolings):
            raise NotImplementedError(
                "TorchPureStateBackend exact gradients currently support partial-trace pooling only"
            )

        parameter_tensor = self._parameter_tensor(parameters, requires_grad=True)
        states_array = np.asarray(states, dtype=np.complex128)
        labels_array = np.asarray(labels, dtype=np.float64)

        if states_array.ndim != 2:
            raise ValueError("states must have shape (num_examples, hilbert_dimension)")
        if labels_array.shape != (states_array.shape[0],):
            raise ValueError("labels must align with states")

        state_tensor = self._complex_tensor(states_array)
        probability_tensor = self._probabilities_from_states(
            model,
            state_tensor,
            parameter_tensor,
            exact_gradients_required=True,
            batched_statevectors=True,
        )
        label_tensor = torch.as_tensor(labels_array, dtype=self.real_dtype, device=self.device)

        if loss_name == "mse":
            loss = torch.mean((probability_tensor - label_tensor) ** 2)
        else:
            clipped = torch.clamp(probability_tensor, 1e-8, 1.0 - 1e-8)
            loss = -torch.mean(
                label_tensor * torch.log(clipped) + (1.0 - label_tensor) * torch.log(1.0 - clipped)
            )

        loss.backward()
        assert parameter_tensor.grad is not None
        return np.asarray(parameter_tensor.grad.detach().cpu().numpy(), dtype=np.float64)

    def _probabilities_from_states(
        self,
        model: BackendCompatibleQCNN,
        states: "torch.Tensor",
        parameters: "torch.Tensor",
        *,
        exact_gradients_required: bool,
        batched_statevectors: bool,
    ) -> "torch.Tensor":
        final_density, final_num_qubits = self._forward_density(
            model,
            states,
            parameters,
            exact_gradients_required=exact_gradients_required,
            batched_statevectors=batched_statevectors,
        )
        return self._readout_probability(
            model,
            final_density,
            final_num_qubits,
            parameters[model.readout_slice],
        )

    def _forward_density(
        self,
        model: BackendCompatibleQCNN,
        state: "torch.Tensor",
        parameters: "torch.Tensor",
        *,
        exact_gradients_required: bool,
        batched_statevectors: bool,
    ) -> tuple["torch.Tensor", int]:
        current_density = self._states_to_density_matrices(state, batched_statevectors=batched_statevectors)
        current_num_qubits = int(model.block_num_qubits[0])

        for block_index, convolution in enumerate(model.convolutions):
            convolution_parameters = parameters[model.convolution_slices[block_index]]
            unitary = self._convolution_unitary(convolution, convolution_parameters)
            current_density = self._apply_unitary(unitary, current_density)

            if block_index < len(model.poolings):
                pooling = model.poolings[block_index]
                pooling_parameters = parameters[model.pooling_slices[block_index]]
                current_density, current_num_qubits = self._apply_pooling(
                    pooling,
                    current_density,
                    pooling_parameters,
                    exact_gradients_required=exact_gradients_required,
                )

        return current_density, current_num_qubits

    def _apply_pooling(
        self,
        pooling: object,
        density_matrix: "torch.Tensor",
        parameters: "torch.Tensor",
        *,
        exact_gradients_required: bool,
    ) -> tuple["torch.Tensor", int]:
        if isinstance(pooling, PartialTracePooling):
            reduced = torch_partial_trace_density_matrix(
                density_matrix,
                pooling.config.num_qubits,
                pooling.trace_out_sites(),
            )
            return reduced, int(pooling.output_num_qubits)

        if exact_gradients_required:
            raise NotImplementedError(
                "TorchPureStateBackend exact gradients currently support partial-trace pooling only"
            )

        density_numpy = self._to_numpy(density_matrix)
        parameter_numpy = np.asarray(parameters.detach().cpu().numpy(), dtype=np.float64)
        if density_numpy.ndim == 3:
            reduced_numpy = np.asarray(
                [pooling.apply(sample, parameters=parameter_numpy) for sample in density_numpy],
                dtype=np.complex128,
            )
        else:
            reduced_numpy = pooling.apply(density_numpy, parameters=parameter_numpy)
        return self._complex_tensor(reduced_numpy), int(pooling.output_num_qubits)

    def _readout_probability(
        self,
        model: BackendCompatibleQCNN,
        density_matrix: "torch.Tensor",
        num_qubits: int,
        readout_parameters: "torch.Tensor",
    ) -> "torch.Tensor":
        if model.config.readout_mode == "swap":
            if num_qubits != 2:
                raise ValueError("swap readout expects the QCNN to terminate on exactly 2 qubits")
            swap_value = torch_expectation_value(density_matrix, self._swap_operator())
            return torch.clamp(0.5 * (swap_value + 1.0), 0.0, 1.0)

        feature_operator = self._dimerization_operator(model, num_qubits)
        feature_value = torch_expectation_value(density_matrix, feature_operator)
        readout_weight = readout_parameters[0]
        readout_bias = readout_parameters[1]
        return torch.sigmoid(readout_weight * feature_value + readout_bias)

    def _convolution_unitary(
        self,
        convolution: object,
        parameters: "torch.Tensor",
    ) -> "torch.Tensor":
        if isinstance(convolution, SU2SwapConvolution):
            return self._brickwork_unitary(
                num_qubits=convolution.config.num_qubits,
                active_parities=convolution.active_parities(),
                shared_parameter=convolution.config.shared_parameter,
                block_size=1,
                parameters=parameters,
                pairs_for_parity=convolution.pairs_for_parity,
                gate_builder=lambda gate_parameters: self._su2_gate(gate_parameters[0]),
            )

        if isinstance(convolution, AnisotropicConvolution):
            return self._brickwork_unitary(
                num_qubits=convolution.config.num_qubits,
                active_parities=convolution.active_parities(),
                shared_parameter=convolution.config.shared_parameter,
                block_size=3,
                parameters=parameters,
                pairs_for_parity=convolution.pairs_for_parity,
                gate_builder=self._anisotropic_gate,
            )

        if isinstance(convolution, HEAConvolution):
            return self._brickwork_unitary(
                num_qubits=convolution.config.num_qubits,
                active_parities=convolution.active_parities(),
                shared_parameter=convolution.config.shared_parameter,
                block_size=8,
                parameters=parameters,
                pairs_for_parity=convolution.pairs_for_parity,
                gate_builder=self._hea_gate,
            )

        raise NotImplementedError(
            f"TorchPureStateBackend does not know how to build {type(convolution).__name__} unitaries"
        )

    def _brickwork_unitary(
        self,
        *,
        num_qubits: int,
        active_parities: tuple[str, ...],
        shared_parameter: bool,
        block_size: int,
        parameters: "torch.Tensor",
        pairs_for_parity: Any,
        gate_builder: Any,
    ) -> "torch.Tensor":
        total = self._identity(1 << num_qubits)
        parameter_offset = 0
        for parity in active_parities:
            pairs = pairs_for_parity(parity)
            pair_count = len(pairs)
            if shared_parameter:
                parity_parameters = parameters[parameter_offset : parameter_offset + block_size]
            else:
                parity_parameters = parameters[parameter_offset : parameter_offset + block_size * pair_count]
            sublayer = self._sublayer_unitary(
                num_qubits=num_qubits,
                pair_starts={left_site for left_site, _ in pairs},
                shared_parameter=shared_parameter,
                block_size=block_size,
                parity_parameters=parity_parameters,
                gate_builder=gate_builder,
            )
            total = sublayer @ total
            parameter_offset += int(parity_parameters.numel())
        return total

    def _sublayer_unitary(
        self,
        *,
        num_qubits: int,
        pair_starts: set[int],
        shared_parameter: bool,
        block_size: int,
        parity_parameters: "torch.Tensor",
        gate_builder: Any,
    ) -> "torch.Tensor":
        local_factors = []
        pair_index = 0
        site = 0
        while site < num_qubits:
            if site in pair_starts:
                if shared_parameter:
                    gate_parameters = parity_parameters[:block_size]
                else:
                    start = block_size * pair_index
                    gate_parameters = parity_parameters[start : start + block_size]
                local_factors.append(gate_builder(gate_parameters))
                pair_index += 1
                site += 2
            else:
                local_factors.append(self._identity(2))
                site += 1
        return torch_kron_all(local_factors)

    def _su2_gate(self, theta: "torch.Tensor") -> "torch.Tensor":
        identity = self._identity(4)
        swap = self._swap_operator()
        return torch.cos(theta) * identity - 1.0j * torch.sin(theta) * swap

    def _anisotropic_gate(self, parameters: "torch.Tensor") -> "torch.Tensor":
        theta_x, theta_y, theta_z = parameters.unbind()
        identity = self._identity(4)
        xx = self._xx_operator()
        yy = self._yy_operator()
        zz = self._zz_operator()
        x_gate = torch.cos(theta_x) * identity - 1.0j * torch.sin(theta_x) * xx
        y_gate = torch.cos(theta_y) * identity - 1.0j * torch.sin(theta_y) * yy
        z_gate = torch.cos(theta_z) * identity - 1.0j * torch.sin(theta_z) * zz
        return z_gate @ y_gate @ x_gate

    def _hea_gate(self, parameters: "torch.Tensor") -> "torch.Tensor":
        (
            pre_q0_y,
            pre_q0_z,
            pre_q1_y,
            pre_q1_z,
            post_q0_y,
            post_q0_z,
            post_q1_y,
            post_q1_z,
        ) = parameters.unbind()

        pre_q0 = self._rz(pre_q0_z) @ self._ry(pre_q0_y)
        pre_q1 = self._rz(pre_q1_z) @ self._ry(pre_q1_y)
        post_q0 = self._rz(post_q0_z) @ self._ry(post_q0_y)
        post_q1 = self._rz(post_q1_z) @ self._ry(post_q1_y)

        pre_layer = torch_kron_all([pre_q0, pre_q1])
        post_layer = torch_kron_all([post_q0, post_q1])
        return post_layer @ self._cz_operator() @ pre_layer

    def _ry(self, theta: "torch.Tensor") -> "torch.Tensor":
        half = 0.5 * theta
        cosine = torch.cos(half).to(dtype=self.complex_dtype)
        sine = torch.sin(half).to(dtype=self.complex_dtype)
        return torch.stack(
            (
                torch.stack((cosine, -sine)),
                torch.stack((sine, cosine)),
            )
        )

    def _rz(self, theta: "torch.Tensor") -> "torch.Tensor":
        half = 0.5 * theta
        phase_minus = torch.exp((-1.0j) * half)
        phase_plus = torch.exp(1.0j * half)
        zero = torch.zeros((), dtype=self.complex_dtype, device=self.device)
        return torch.stack(
            (
                torch.stack((phase_minus, zero)),
                torch.stack((zero, phase_plus)),
            )
        )

    def _states_to_density_matrices(
        self,
        state: "torch.Tensor",
        *,
        batched_statevectors: bool,
    ) -> "torch.Tensor":
        if batched_statevectors:
            if state.ndim == 2:
                return statevectors_to_density_matrices(state)
            if state.ndim == 3 and state.shape[-2] == state.shape[-1]:
                return state
            raise ValueError("Batched inputs must be statevectors with shape (batch, dim) or density matrices")

        if state.ndim == 1:
            return as_torch_density_matrix(state)
        if state.ndim == 2 and state.shape[-2] == state.shape[-1]:
            return state
        raise ValueError("Single-state inputs must be a statevector or a square density matrix")

    def _apply_unitary(
        self,
        unitary: "torch.Tensor",
        density_matrix: "torch.Tensor",
    ) -> "torch.Tensor":
        unitary_dagger = torch.conj(unitary).transpose(-2, -1)
        if density_matrix.ndim == 2:
            return unitary @ density_matrix @ unitary_dagger
        return torch.matmul(torch.matmul(unitary.unsqueeze(0), density_matrix), unitary_dagger.unsqueeze(0))

    def _identity(self, dimension: int) -> "torch.Tensor":
        cache_key = ("identity", int(dimension))
        if cache_key not in self._operator_cache:
            self._operator_cache[cache_key] = torch.eye(dimension, dtype=self.complex_dtype, device=self.device)
        return self._operator_cache[cache_key]

    def _swap_operator(self) -> "torch.Tensor":
        cache_key = ("swap", 4)
        if cache_key not in self._operator_cache:
            self._operator_cache[cache_key] = self._complex_tensor(
                np.asarray(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.complex128,
                )
            )
        return self._operator_cache[cache_key]

    def _cz_operator(self) -> "torch.Tensor":
        cache_key = ("cz", 4)
        if cache_key not in self._operator_cache:
            self._operator_cache[cache_key] = self._complex_tensor(
                np.diag((1.0, 1.0, 1.0, -1.0)).astype(np.complex128)
            )
        return self._operator_cache[cache_key]

    def _xx_operator(self) -> "torch.Tensor":
        cache_key = ("xx", 4)
        if cache_key not in self._operator_cache:
            self._operator_cache[cache_key] = self._complex_tensor(
                np.asarray(
                    (
                        (0.0, 0.0, 0.0, 1.0),
                        (0.0, 0.0, 1.0, 0.0),
                        (0.0, 1.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0, 0.0),
                    ),
                    dtype=np.complex128,
                )
            )
        return self._operator_cache[cache_key]

    def _yy_operator(self) -> "torch.Tensor":
        cache_key = ("yy", 4)
        if cache_key not in self._operator_cache:
            self._operator_cache[cache_key] = self._complex_tensor(
                np.asarray(
                    (
                        (0.0, 0.0, 0.0, -1.0),
                        (0.0, 0.0, 1.0, 0.0),
                        (0.0, 1.0, 0.0, 0.0),
                        (-1.0, 0.0, 0.0, 0.0),
                    ),
                    dtype=np.complex128,
                )
            )
        return self._operator_cache[cache_key]

    def _zz_operator(self) -> "torch.Tensor":
        cache_key = ("zz", 4)
        if cache_key not in self._operator_cache:
            self._operator_cache[cache_key] = self._complex_tensor(
                np.diag((1.0, -1.0, -1.0, 1.0)).astype(np.complex128)
            )
        return self._operator_cache[cache_key]

    def _dimerization_operator(
        self,
        model: BackendCompatibleQCNN,
        num_qubits: int,
    ) -> "torch.Tensor":
        cache_key = (f"dimerization_{model.config.boundary}", int(num_qubits))
        if cache_key not in self._operator_cache:
            self._operator_cache[cache_key] = self._complex_tensor(model.dimerization_operator(num_qubits))
        return self._operator_cache[cache_key]

    def _complex_tensor(self, array: ComplexArray | np.ndarray | "torch.Tensor") -> "torch.Tensor":
        return torch.as_tensor(array, dtype=self.complex_dtype, device=self.device)

    def _parameter_tensor(
        self,
        parameters: np.ndarray | list[float],
        *,
        requires_grad: bool,
    ) -> "torch.Tensor":
        parameter_tensor = torch.tensor(
            np.asarray(parameters, dtype=np.float64),
            dtype=self.real_dtype,
            device=self.device,
        )
        if requires_grad:
            parameter_tensor.requires_grad_(True)
        return parameter_tensor

    @staticmethod
    def _to_numpy(tensor: "torch.Tensor") -> np.ndarray:
        return np.asarray(tensor.detach().cpu().numpy())


__all__ = ["TORCH_AVAILABLE", "TorchPureStateBackend"]
