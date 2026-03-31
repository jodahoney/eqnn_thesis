"""Base interfaces for trainable quantum models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from eqnn.types import ComplexArray


@dataclass(frozen=True)
class QCNNForwardPass:
    """Interpretable outputs from a QCNN forward pass."""

    final_density_matrix: ComplexArray
    final_num_qubits: int
    readout_mode: str
    primary_singlet_mean: float
    secondary_singlet_mean: float
    dimerization_feature: float
    swap_expectation: float | None
    logit: float | None
    probability: float


@runtime_checkable
class QuantumModel(Protocol):
    """Minimal prediction interface for trainable quantum models."""

    def predict(self, state: ComplexArray, parameters: np.ndarray | None = None) -> float:
        ...

    def predict_batch(
        self,
        states: ComplexArray,
        parameters: np.ndarray | None = None,
    ) -> np.ndarray:
        ...


@runtime_checkable
class ThresholdedModel(Protocol):
    """Optional threshold interface used by the trainer and reproduction path."""

    def get_classification_threshold(self) -> float:
        ...

    def set_classification_threshold(self, threshold: float) -> None:
        ...

    def predict_labels_batch(
        self,
        states: ComplexArray,
        parameters: np.ndarray | None = None,
        *,
        threshold: float | None = None,
    ) -> np.ndarray:
        ...


@runtime_checkable
class TrainableModel(QuantumModel, Protocol):
    """High-level model contract expected by the trainer and experiments."""

    @property
    def parameter_count(self) -> int:
        ...

    def get_parameters(self) -> np.ndarray:
        ...

    def set_parameters(self, parameters: np.ndarray) -> None:
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

    def loss_gradient(
        self,
        states: ComplexArray,
        labels: np.ndarray,
        parameters: np.ndarray | None = None,
        *,
        finite_difference_eps: float = 1e-3,
        loss_name: str = "bce",
    ) -> np.ndarray:
        ...
