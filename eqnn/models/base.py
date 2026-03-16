"""Base interfaces for trainable quantum models."""

from __future__ import annotations

from typing import Protocol

from eqnn.types import ComplexArray


class QuantumModel(Protocol):
    """Minimal prediction interface for future training code."""

    def predict(self, state: ComplexArray) -> float:
        ...
