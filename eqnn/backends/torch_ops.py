"""Torch-native linear-algebra helpers for pure-state QCNN backends."""

from __future__ import annotations

from functools import reduce
from typing import Iterable

try:  # pragma: no cover - exercised indirectly when torch is installed
    import torch
except ImportError:  # pragma: no cover - local test environments may not have torch
    torch = None  # type: ignore[assignment]


TORCH_AVAILABLE = torch is not None


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - guarded by TORCH_AVAILABLE in tests
        raise ImportError(
            "TorchPureStateBackend requires the optional 'torch' dependency. "
            "Install it with `pip install 'eqnn-simulator[torch]'` or `pip install torch`."
        )


def kron_all(operators: Iterable["torch.Tensor"]) -> "torch.Tensor":
    _require_torch()
    operator_list = list(operators)
    if not operator_list:
        raise ValueError("kron_all requires at least one operator")
    return reduce(torch.kron, operator_list)


def statevector_to_density_matrix(state: "torch.Tensor") -> "torch.Tensor":
    _require_torch()
    if state.ndim != 1:
        raise ValueError("statevector_to_density_matrix expects a one-dimensional statevector")
    return torch.outer(state, torch.conj(state))


def statevectors_to_density_matrices(states: "torch.Tensor") -> "torch.Tensor":
    _require_torch()
    if states.ndim != 2:
        raise ValueError("statevectors_to_density_matrices expects shape (batch, hilbert_dimension)")
    return states.unsqueeze(-1) * torch.conj(states).unsqueeze(-2)


def as_density_matrix(state: "torch.Tensor") -> "torch.Tensor":
    _require_torch()
    if state.ndim == 1:
        return statevector_to_density_matrix(state)
    if state.ndim == 2 and state.shape[0] == state.shape[1]:
        return state
    raise ValueError("State must be either a statevector or a square density matrix")


def partial_trace_density_matrix(
    density_matrix: "torch.Tensor",
    num_qubits: int,
    traced_out_sites: Iterable[int],
) -> "torch.Tensor":
    _require_torch()
    traced_out = sorted(set(int(site) for site in traced_out_sites))
    expected_dimension = 1 << num_qubits
    if density_matrix.shape != (expected_dimension, expected_dimension):
        raise ValueError(
            "density_matrix shape does not match the provided num_qubits: "
            f"expected {(expected_dimension, expected_dimension)}, got {tuple(density_matrix.shape)}"
        )

    batch_shape = tuple(density_matrix.shape[:-2])
    batch_ndim = len(batch_shape)
    tensor = density_matrix.reshape(batch_shape + (2,) * num_qubits * 2)
    current_num_qubits = num_qubits
    for site in reversed(traced_out):
        tensor = torch.diagonal(
            tensor,
            offset=0,
            dim1=batch_ndim + site,
            dim2=batch_ndim + site + current_num_qubits,
        ).sum(dim=-1)
        current_num_qubits -= 1

    reduced_dimension = 1 << current_num_qubits
    return tensor.reshape(batch_shape + (reduced_dimension, reduced_dimension))


def expectation_value(
    density_matrix: "torch.Tensor",
    operator: "torch.Tensor",
) -> "torch.Tensor":
    _require_torch()
    return torch.real(torch.einsum("...ij,ji->...", density_matrix, operator))
