"""Fused RMSNorm + gate-multiply + residual-add Triton kernel (forward-only).

Implements the ZImage S3-DiT post-attention/FFN pattern in a single kernel:

    output = residual + gate * RMSNorm(x, weight, eps)

Three separate DRAM round-trips (norm, gate multiply, residual add) are
collapsed into one.  Optimised for hidden_size=3840 on SM120 (RTX 5090).
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
def _fused_norm_gate_residual_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,
    R_row_stride,
    G_ptr,
    G_row_stride,
    W_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Fused RMSNorm(X) * G + R kernel.

    Each program handles one row:
        1. norm = RMSNorm(X, W, eps)
        2. Y = R + G * norm

    Args:
        Y_ptr: Output tensor pointer.
        X_ptr: Input tensor (attention/FFN output) pointer.
        R_ptr: Residual stream tensor pointer.
        G_ptr: Gate tensor pointer (already flattened to 2-D).
        W_ptr: RMSNorm weight vector pointer (hidden_size,).
        n_cols: Hidden size (columns per row).
        eps: Numerical stability constant for RMSNorm.
        BLOCK_SIZE: Compile-time block size (power of 2 >= n_cols).
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    X_row = tl.load(X_ptr + row_idx * X_row_stride + col_offsets, mask=mask, other=0.0)
    R_row = tl.load(R_ptr + row_idx * R_row_stride + col_offsets, mask=mask, other=0.0)
    G_row = tl.load(G_ptr + row_idx * G_row_stride + col_offsets, mask=mask, other=0.0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)

    out_dtype = X_row.dtype

    # RMSNorm in fp32
    X_fp32 = X_row.to(tl.float32)
    mean_square = tl.sum(X_fp32 * X_fp32, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)
    X_norm = (X_fp32 * rstd).to(out_dtype)
    X_scaled = X_norm * W_row  # apply weight

    # Gate multiply + residual add
    Y_row = R_row + G_row * X_scaled
    tl.store(Y_ptr + row_idx * Y_row_stride + col_offsets, Y_row, mask=mask)


def triton_fused_norm_gate_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused RMSNorm + gate + residual-add in one Triton kernel pass.

    Computes: ``output = residual + gate * RMSNorm(x, weight, eps)``

    Args:
        x: Attention/FFN output of shape (B, seq_len, hidden_size).
        residual: Residual stream of same shape as x.
        gate: Gate tensor, broadcastable to x shape.
        weight: RMSNorm weight of shape (hidden_size,).
        eps: RMSNorm epsilon (default 1e-5).

    Returns:
        Updated residual tensor, same shape as x.
    """
    shape = x.shape
    n_cols = shape[-1]
    n_rows = x.numel() // n_cols

    x_2d = x.contiguous().view(n_rows, n_cols)
    r_2d = residual.contiguous().view(n_rows, n_cols)
    g_2d = gate.expand(shape).contiguous().view(n_rows, n_cols)
    w_1d = weight.contiguous()

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    y = torch.empty_like(x_2d)

    _fused_norm_gate_residual_kernel[(n_rows,)](
        y,
        y.stride(0),
        x_2d,
        x_2d.stride(0),
        r_2d,
        r_2d.stride(0),
        g_2d,
        g_2d.stride(0),
        w_1d,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return y.view(*shape)
