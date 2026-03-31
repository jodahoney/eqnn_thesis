"""Base protocols for QCNN simulation backends."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from eqnn.models.base import QCNNForwardPass
from eqnn.types import ComplexArray


class BackendCompatibleQCNN(Protocol):
    """Minimal QCNN execution surface consumed by numerical backends."""

    config: object
    block_num_qubits: Sequence[int]
    convolutions: Sequence[object]
    poolings: Sequence[object]

    @property
    def convolution_slices(self) -> Sequence[slice]:
        ...

    @property
    def pooling_slices(self) -> Sequence[slice]:
        ...

    @property
    def readout_slice(self) -> slice:
        ...

    def finalize_forward_pass(
        self,
        density_matrix: ComplexArray,
        num_qubits: int,
        readout_parameters: np.ndarray,
    ) -> QCNNForwardPass:
        ...

    def readout_loss_gradient(
        self,
        density_matrix: ComplexArray,
        num_qubits: int,
        readout_parameters: np.ndarray,
        label: float,
        *,
        loss_name: str,
    ) -> tuple[ComplexArray, np.ndarray]:
        ...

    def apply_pooling_adjoint(
        self,
        pooling: object,
        observable: ComplexArray,
        pooling_parameters: np.ndarray,
    ) -> ComplexArray:
        ...

    def loss(
        self,
        states: ComplexArray,
        labels: np.ndarray,
        parameters: np.ndarray | None = None,
        *,
        loss_name: str = "bce",
    ) -> float:
        ...


class QCNNBackend(Protocol):
    """Backend interface for QCNN forward simulation and exact gradients."""

    @property
    def supports_exact_gradients(self) -> bool:
        ...

    def forward(
        self,
        model: BackendCompatibleQCNN,
        state: ComplexArray,
        parameters: np.ndarray,
    ) -> QCNNForwardPass:
        ...

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
        ...
