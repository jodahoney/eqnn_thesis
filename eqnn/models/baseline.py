"""Symmetry-agnostic QCNN baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from eqnn.backends import QCNNBackend
from eqnn.layers.baseline import AnisotropicConvolution, AnisotropicConvolutionConfig
from eqnn.models.base import QCNNForwardPass
from eqnn.models.qcnn import BaseQCNNModel, QCNNConfig


@dataclass(frozen=True)
class BaselineQCNNConfig:
    num_qubits: int
    min_readout_qubits: int | None = None
    boundary: str = "open"
    parity_sequence: tuple[str, ...] = ("even", "odd")
    shared_convolution_parameter: bool = True
    pooling_mode: str = "partial_trace"
    pooling_keep: str = "left"
    readout_mode: str = "swap"
    symmetry_group: str = "none"

    def __post_init__(self) -> None:
        if self.readout_mode not in {"swap", "dimerization"}:
            raise ValueError("readout_mode must be 'swap' or 'dimerization'")
        min_readout_qubits = (
            (2 if self.readout_mode == "swap" else min(4, self.num_qubits))
            if self.min_readout_qubits is None
            else self.min_readout_qubits
        )
        object.__setattr__(self, "min_readout_qubits", min_readout_qubits)

        if self.num_qubits < 2:
            raise ValueError("num_qubits must be at least 2")
        if min_readout_qubits < 2 or min_readout_qubits > self.num_qubits:
            raise ValueError("min_readout_qubits must lie in [2, num_qubits]")
        if self.readout_mode == "swap" and min_readout_qubits != 2:
            raise ValueError("swap readout requires min_readout_qubits=2")
        if self.boundary not in {"open", "periodic"}:
            raise ValueError("boundary must be 'open' or 'periodic'")
        if self.pooling_mode not in {"partial_trace", "equivariant"}:
            raise ValueError("pooling_mode must be 'partial_trace' or 'equivariant'")


class BaselineQCNN(BaseQCNNModel):
    """QCNN baseline that uses anisotropic, non-equivariant convolutions."""

    def __init__(
        self,
        config: BaselineQCNNConfig,
        parameters: Iterable[float] | None = None,
        backend: QCNNBackend | None = None,
    ) -> None:
        qcnn_config = QCNNConfig(
            num_qubits=config.num_qubits,
            min_readout_qubits=config.min_readout_qubits,
            boundary=config.boundary,
            parity_sequence=config.parity_sequence,
            shared_convolution_parameter=config.shared_convolution_parameter,
            pooling_mode=config.pooling_mode,
            pooling_keep=config.pooling_keep,
            readout_mode=config.readout_mode,
            symmetry_group=config.symmetry_group,
        )
        block_num_qubits = self.build_block_num_qubits(qcnn_config)
        convolutions = [
            AnisotropicConvolution(
                AnisotropicConvolutionConfig(
                    num_qubits=num_qubits,
                    parity_sequence=config.parity_sequence,
                    shared_parameter=config.shared_convolution_parameter,
                )
            )
            for num_qubits in block_num_qubits
        ]
        super().__init__(
            qcnn_config,
            block_num_qubits=block_num_qubits,
            convolutions=convolutions,
            backend=backend,
            parameters=parameters,
        )


__all__ = ["BaselineQCNN", "BaselineQCNNConfig", "QCNNForwardPass"]
