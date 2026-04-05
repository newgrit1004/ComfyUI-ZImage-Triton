"""Fused RMSNorm Triton kernel (forward-only, inference optimized).

Optimized for ZImage S3-DiT hidden_size=3840 on SM120 (RTX 5090).
"""

import torch
import triton
import triton.language as tl

from zimage_triton.kernels.utils import calculate_settings

try:
    from triton.language.extra.libdevice import rsqrt
except ModuleNotFoundError:
    from triton.language.extra.cuda.libdevice import rsqrt


@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Triton JIT kernel for RMSNorm forward pass.

    Each program handles one row of the input tensor.

    Args:
        Y_ptr: Pointer to output tensor.
        Y_row_stride: Row stride of output tensor.
        X_ptr: Pointer to input tensor.
        X_row_stride: Row stride of input tensor.
        W_ptr: Pointer to weight (scale) tensor.
        n_cols: Number of columns (hidden_size).
        eps: Small constant for numerical stability.
        BLOCK_SIZE: Compile-time block size (power of 2 >= n_cols).
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    X_row = tl.load(X_ptr + row_idx * X_row_stride + col_offsets, mask=mask, other=0.0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)

    X_row_dtype = X_row.dtype
    X_row_fp32 = X_row.to(tl.float32)

    mean_square = tl.sum(X_row_fp32 * X_row_fp32, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)

    X_norm = (X_row_fp32 * rstd).to(X_row_dtype)
    Y_row = X_norm * W_row

    tl.store(Y_ptr + row_idx * Y_row_stride + col_offsets, Y_row, mask=mask)


def triton_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply RMSNorm using a fused Triton kernel.

    Normalises the last dimension of x using root-mean-square statistics,
    then scales by weight.  Computation is performed in float32 internally
    for numerical stability; the output is cast back to the input dtype.

    Args:
        x: Input tensor of any shape (..., hidden_size).
        weight: Scale parameter of shape (hidden_size,).
        eps: Numerical stability constant added before taking sqrt.

    Returns:
        Normalised tensor with the same shape and dtype as x.
    """
    shape = x.shape
    n_cols = shape[-1]
    x = x.contiguous().view(-1, n_cols)
    weight = weight.contiguous()
    n_rows = x.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    y = torch.empty_like(x)

    _rms_norm_forward_kernel[(n_rows,)](
        y,
        y.stride(0),
        x,
        x.stride(0),
        weight,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return y.view(*shape)


class TritonRMSNorm(torch.nn.Module):
    """Drop-in RMSNorm module backed by a fused Triton kernel.

    Replaces the standard PyTorch RMSNorm with a single-pass Triton
    kernel that avoids multiple DRAM round-trips.

    Args:
        hidden_size: Size of the last dimension to normalise.
        eps: Numerical stability constant (default 1e-5).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            RMSNorm-normalised tensor with the same shape and dtype as x.
        """
        return triton_rms_norm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"
