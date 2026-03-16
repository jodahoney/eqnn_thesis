"""Pooling layers based on pairwise partial traces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eqnn.physics.quantum import as_density_matrix, partial_trace_density_matrix
from eqnn.types import ComplexArray


@dataclass(frozen=True)
class PartialTracePoolingConfig:
    num_qubits: int
    keep: str = "left"

    def __post_init__(self) -> None:
        if self.num_qubits < 2:
            raise ValueError("num_qubits must be at least 2")
        if self.keep not in {"left", "right"}:
            raise ValueError("keep must be either 'left' or 'right'")


class PartialTracePooling:
    """Pool each adjacent pair by tracing out one qubit."""

    def __init__(self, config: PartialTracePoolingConfig) -> None:
        self.config = config

    def __call__(self, state: ComplexArray) -> ComplexArray:
        density_matrix = as_density_matrix(state)
        return np.asarray(
            partial_trace_density_matrix(
                density_matrix,
                self.config.num_qubits,
                self.trace_out_sites(),
            ),
            dtype=np.complex128,
        )

    @property
    def output_num_qubits(self) -> int:
        return self.config.num_qubits - len(self.trace_out_sites())

    def trace_out_sites(self) -> tuple[int, ...]:
        """Return the sites removed by this pooling layer."""

        if self.config.keep == "left":
            return tuple(range(1, self.config.num_qubits, 2))
        if self.config.num_qubits % 2 == 0:
            return tuple(range(0, self.config.num_qubits, 2))
        return tuple(range(0, self.config.num_qubits - 1, 2))
