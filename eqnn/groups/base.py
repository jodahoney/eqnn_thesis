"""Base interfaces for symmetry groups and representations."""

from __future__ import annotations

from typing import Protocol

from eqnn.types import ComplexArray


class SymmetryGroup(Protocol):
    """Protocol for future group-aware layer implementations."""

    name: str

    def representation(self, num_qubits: int, *parameters: float) -> ComplexArray:
        ...
