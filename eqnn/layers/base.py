"""Common interfaces for quantum layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from eqnn.types import ComplexArray


class QuantumLayer(Protocol):
    """Callable interface shared by simulator layers."""

    def __call__(self, state: ComplexArray) -> ComplexArray:
        ...


@dataclass(frozen=True)
class LayerContext:
    """Lightweight metadata for upcoming layer implementations."""

    num_qubits: int
    symmetry_group: str = "SU(2)"
