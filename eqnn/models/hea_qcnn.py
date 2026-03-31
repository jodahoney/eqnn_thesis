"""HEA-inspired QCNN baseline model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from eqnn.backends import QCNNBackend
from eqnn.layers.hea import HEAConvolution, HEAConvolutionConfig
from eqnn.models.base import QCNNForwardPass
from eqnn.models.qcnn import BaseQCNNModel, QCNNConfig


@dataclass(frozen=True)
class HEAQCNNConfig(QCNNConfig):
    symmetry_group: str = "none"
    entangler: str = "cz"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.entangler != "cz":
            raise ValueError("Only entangler='cz' is currently supported")


class HEAQCNN(BaseQCNNModel):
    """QCNN baseline with HEA-style local convolution blocks."""

    def __init__(
        self,
        config: HEAQCNNConfig,
        parameters: Iterable[float] | None = None,
        backend: QCNNBackend | None = None,
    ) -> None:
        block_num_qubits = self.build_block_num_qubits(config)
        convolutions = [
            HEAConvolution(
                HEAConvolutionConfig(
                    num_qubits=num_qubits,
                    parity_sequence=config.parity_sequence,
                    shared_parameter=config.shared_convolution_parameter,
                    entangler=config.entangler,
                )
            )
            for num_qubits in block_num_qubits
        ]
        super().__init__(
            config,
            block_num_qubits=block_num_qubits,
            convolutions=convolutions,
            backend=backend,
            parameters=parameters,
        )


__all__ = ["HEAQCNN", "HEAQCNNConfig", "QCNNForwardPass"]
