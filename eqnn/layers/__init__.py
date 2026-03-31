"""Layer interfaces and upcoming implementations."""

from eqnn.layers.baseline import AnisotropicConvolution, AnisotropicConvolutionConfig
from eqnn.layers.convolution import SU2SwapConvolution, SU2SwapConvolutionConfig
from eqnn.layers.hea import HEAConvolution, HEAConvolutionConfig
from eqnn.layers.pooling import (
    PartialTracePooling,
    PartialTracePoolingConfig,
    SU2EquivariantPooling,
    SU2EquivariantPoolingConfig,
)

__all__ = [
    "AnisotropicConvolution",
    "AnisotropicConvolutionConfig",
    "HEAConvolution",
    "HEAConvolutionConfig",
    "PartialTracePooling",
    "PartialTracePoolingConfig",
    "SU2EquivariantPooling",
    "SU2EquivariantPoolingConfig",
    "SU2SwapConvolution",
    "SU2SwapConvolutionConfig",
]
