"""3D RoPE Triton kernel (forward-only, inference optimized).

Replaces ZImage's ``apply_rotary_emb`` which uses complex64 multiplication
with real arithmetic:

    complex multiply (a + bi)(c + di) = (ac - bd) + (ad + bc)i

For interleaved pairs (x[2k], x[2k+1]) with (cos[k], sin[k]):
    new_x[2k]   = x[2k]   * cos[k] - x[2k+1] * sin[k]
    new_x[2k+1] = x[2k+1] * cos[k] + x[2k]   * sin[k]

Optimized for ZImage S3-DiT head_dim=128, heads=30 on SM120 (RTX 5090).
"""

import torch
import triton
import triton.language as tl

from zimage_triton.kernels.utils import calculate_settings


@triton.jit
def _rope_3d_forward_kernel(
    X_ptr,
    COS_ptr,
    SIN_ptr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Triton JIT kernel for 3D RoPE forward pass (in-place).

    Each program handles one row (one token × one head).  Real arithmetic
    avoids the overhead of complex dtype conversion.

    Args:
        X_ptr: Pointer to input/output tensor, shape (n_rows, head_dim).
               Modified in-place.
        COS_ptr: Pointer to cos values, shape (n_rows, half_dim).
        SIN_ptr: Pointer to sin values, shape (n_rows, half_dim).
        head_dim: Full head dimension (compile-time constant).
        half_dim: head_dim // 2 (compile-time constant).
        BLOCK_SIZE: Next power-of-2 >= half_dim (compile-time constant).
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_mask = col_offsets < half_dim

    # Even/odd element offsets within the row
    even_offsets = col_offsets * 2
    odd_offsets = even_offsets + 1
    even_mask = even_offsets < head_dim
    odd_mask = odd_offsets < head_dim

    # Load interleaved pairs
    x_even = tl.load(
        X_ptr + row_idx * head_dim + even_offsets, mask=even_mask, other=0.0
    )
    x_odd = tl.load(X_ptr + row_idx * head_dim + odd_offsets, mask=odd_mask, other=0.0)

    # Load cos/sin values
    cos_val = tl.load(
        COS_ptr + row_idx * half_dim + col_offsets, mask=half_mask, other=0.0
    )
    sin_val = tl.load(
        SIN_ptr + row_idx * half_dim + col_offsets, mask=half_mask, other=0.0
    )

    # Promote to fp32 for numerical precision
    x_even_f = x_even.to(tl.float32)
    x_odd_f = x_odd.to(tl.float32)
    cos_f = cos_val.to(tl.float32)
    sin_f = sin_val.to(tl.float32)

    # Complex multiply as real ops:  (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    new_even = x_even_f * cos_f - x_odd_f * sin_f
    new_odd = x_odd_f * cos_f + x_even_f * sin_f

    # Store back in original dtype
    tl.store(
        X_ptr + row_idx * head_dim + even_offsets,
        new_even.to(x_even.dtype),
        mask=even_mask,
    )
    tl.store(
        X_ptr + row_idx * head_dim + odd_offsets,
        new_odd.to(x_odd.dtype),
        mask=odd_mask,
    )


def triton_rope_3d(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply 3D RoPE using a fused Triton kernel.

    Replaces ZImage's ``apply_rotary_emb`` which uses complex64 multiplication.
    Uses real arithmetic to avoid dtype overhead:
    ``(a + bi)(c + di) = (ac - bd) + (ad + bc)i``

    The operation is performed in float32 internally for numerical stability;
    the output is cast back to the input dtype.

    Args:
        x: Input tensor of shape ``(batch, seq, heads, head_dim)``.
           The tensor is modified and returned; a contiguous copy is made
           internally when needed.
        freqs_cis: Precomputed complex frequencies of shape
                   ``(batch, seq, head_dim // 2)``, dtype ``torch.complex64``.

    Returns:
        Rotary-embedded tensor with the same shape and dtype as ``x``.

    Raises:
        ValueError: If ``head_dim`` is odd or ``freqs_cis`` half-dim does not
                    match ``head_dim // 2``.
    """
    batch, seq, heads, head_dim = x.shape
    half_dim = head_dim // 2

    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    if freqs_cis.shape[-1] != half_dim:
        raise ValueError(
            f"freqs_cis last dim {freqs_cis.shape[-1]} != head_dim//2 {half_dim}"
        )

    # Handle batch broadcast: freqs_cis may have batch=1 when x has batch>1
    # (e.g., CFG>1 batches positive+negative with shared RoPE frequencies)
    if freqs_cis.shape[0] != batch:
        freqs_cis = freqs_cis.expand(batch, -1, -1).contiguous()

    # Extract real/imag from complex freqs_cis: each (batch, seq, half_dim)
    cos = freqs_cis.real.contiguous()
    sin = freqs_cis.imag.contiguous()

    # Reshape x to (n_rows, head_dim) for row-parallel kernel launch
    x = x.contiguous()
    x_flat = x.view(-1, head_dim)  # (batch*seq*heads, head_dim)

    # Expand cos/sin to match heads: (batch, seq, 1, half_dim) -> (n_rows, half_dim)
    cos_expanded = (
        cos.unsqueeze(2).expand(-1, -1, heads, -1).contiguous().view(-1, half_dim)
    )
    sin_expanded = (
        sin.unsqueeze(2).expand(-1, -1, heads, -1).contiguous().view(-1, half_dim)
    )

    n_rows = x_flat.shape[0]
    BLOCK_SIZE, num_warps = calculate_settings(half_dim)

    _rope_3d_forward_kernel[(n_rows,)](
        x_flat,
        cos_expanded,
        sin_expanded,
        head_dim=head_dim,
        half_dim=half_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return x_flat.view(batch, seq, heads, head_dim)
