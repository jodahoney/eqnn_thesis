"""Layer interfaces and upcoming implementations."""

from eqnn.layers.convolution import SU2SwapConvolution, SU2SwapConvolutionConfig
from eqnn.layers.pooling import (
    PartialTracePooling,
    PartialTracePoolingConfig,
    SU2EquivariantPooling,
    SU2EquivariantPoolingConfig,
)

__all__ = [
    "PartialTracePooling",
    "PartialTracePoolingConfig",
    "SU2EquivariantPooling",
    "SU2EquivariantPoolingConfig",
    "SU2SwapConvolution",
    "SU2SwapConvolutionConfig",
]
