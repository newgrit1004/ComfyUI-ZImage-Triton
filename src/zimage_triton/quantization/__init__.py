"""INT8 dynamic quantization for ZImage GEMM acceleration."""

from zimage_triton.quantization.int8_linear import (
    apply_int8_quantization,
    restore_int8_quantization,
)

__all__ = ["apply_int8_quantization", "restore_int8_quantization"]
