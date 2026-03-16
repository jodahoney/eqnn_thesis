"""Layer interfaces and upcoming implementations."""

from eqnn.layers.convolution import SU2SwapConvolution, SU2SwapConvolutionConfig
from eqnn.layers.pooling import PartialTracePooling, PartialTracePoolingConfig

__all__ = [
    "PartialTracePooling",
    "PartialTracePoolingConfig",
    "SU2SwapConvolution",
    "SU2SwapConvolutionConfig",
]
