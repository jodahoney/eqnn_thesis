"""Symmetry-agnostic QCNN baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from eqnn.layers.baseline import AnisotropicConvolution, AnisotropicConvolutionConfig
from eqnn.models.qcnn import QCNNForwardPass, SU2QCNN


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


class BaselineQCNN(SU2QCNN):
    """QCNN baseline that uses anisotropic, non-equivariant convolutions."""

    def __init__(
        self,
        config: BaselineQCNNConfig,
        parameters: Iterable[float] | None = None,
    ) -> None:
        self.config = config
        self.block_num_qubits = self._build_block_num_qubits()
        self.convolutions = [
            AnisotropicConvolution(
                AnisotropicConvolutionConfig(
                    num_qubits=num_qubits,
                    parity_sequence=self.config.parity_sequence,
                    shared_parameter=self.config.shared_convolution_parameter,
                )
            )
            for num_qubits in self.block_num_qubits
        ]
        self.poolings = [
            self._build_pooling(num_qubits)
            for num_qubits in self.block_num_qubits[:-1]
        ]
        self._convolution_slices = self._build_convolution_slices()
        self._pooling_slices = self._build_pooling_slices()
        self._readout_slice = self._build_readout_slice()
        self.parameters = self._initialize_parameters(parameters)
        self.classification_threshold = 0.5


__all__ = ["BaselineQCNN", "BaselineQCNNConfig", "QCNNForwardPass"]
