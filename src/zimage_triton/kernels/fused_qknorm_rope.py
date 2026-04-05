"""Fused QK-Norm + 3D RoPE Triton kernel (forward-only, inference optimized).

Fuses RMSNorm on Q/K and 3D RoPE application into a single kernel per tensor,
reducing the ZImage attention processor from 4 kernel launches to 2:

    Before (4 launches):
        norm_q(query)                    -> RMSNorm
        apply_rotary_emb(query, freqs)   -> RoPE
        norm_k(key)                      -> RMSNorm
        apply_rotary_emb(key, freqs)     -> RoPE

    After (2 launches):
        triton_fused_qknorm_rope(query, w_q, freqs)
        triton_fused_qknorm_rope(key,   w_k, freqs)

Each kernel pass performs in registers (head_dim=128 fits easily):
    1. Load all head_dim elements + weight
    2. RMSNorm in fp32: x_norm = (x / RMS(x)) * w
    3. Extract even/odd pairs from normalized values
    4. Apply RoPE: (ac-bd) + (ad+bc)i via real arithmetic
    5. Store result

Optimized for ZImage S3-DiT head_dim=128, heads=30 on SM120 (RTX 5090).
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
def _fused_qknorm_rope_kernel(
    X_ptr,
    W_ptr,
    COS_ptr,
    SIN_ptr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
) -> None:
    """Triton JIT kernel for fused RMSNorm + 3D RoPE (in-place).

    Each program handles one row (one token × one head).

    Pass 1 – RMSNorm in fp32 over all head_dim elements:
        mean_sq = sum(x^2) / head_dim
        x_norm  = x * rsqrt(mean_sq + eps) * w

    Pass 2 – 3D RoPE via real arithmetic on normalized values:
        new_x[2k]   = x_norm[2k]   * cos[k] - x_norm[2k+1] * sin[k]
        new_x[2k+1] = x_norm[2k+1] * cos[k] + x_norm[2k]   * sin[k]

    RMSNorm result is stored in-place, then even/odd pairs are reloaded for
    the RoPE pass.  A ``tl.debug_barrier()`` between the store and reload
    ensures the write is visible before the subsequent reads (GPU memory
    ordering within a single warp).

    Args:
        X_ptr: Pointer to input/output tensor (n_rows, head_dim), in-place.
        W_ptr: Pointer to RMSNorm weight vector (head_dim,).
        COS_ptr: Pointer to cos values (n_rows, half_dim).
        SIN_ptr: Pointer to sin values (n_rows, half_dim).
        head_dim: Full head dimension (compile-time constant).
        half_dim: head_dim // 2 (compile-time constant).
        eps: RMSNorm stability epsilon (compile-time constant).
        BLOCK_SIZE: Next power-of-2 >= head_dim (compile-time constant).
        HALF_BLOCK: Next power-of-2 >= half_dim (compile-time constant).
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < head_dim

    # ------------------------------------------------------------------
    # Step 1: RMSNorm
    # ------------------------------------------------------------------
    x_row = tl.load(X_ptr + row_idx * head_dim + col_offsets, mask=mask, other=0.0)
    w_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)

    x_fp32 = x_row.to(tl.float32)
    mean_sq = tl.sum(x_fp32 * x_fp32, axis=0) / head_dim
    rstd = rsqrt(mean_sq + eps)

    # Apply norm + weight; keep normalized values in the original dtype so
    # the RoPE step uses the same type as the final output.
    x_norm = (x_fp32 * rstd).to(x_row.dtype) * w_row  # (BLOCK_SIZE,)

    # ------------------------------------------------------------------
    # Step 2: 3D RoPE on the normalized values
    # ------------------------------------------------------------------
    # Store the RMSNorm result first so we can reload even/odd pairs cleanly.
    tl.store(X_ptr + row_idx * head_dim + col_offsets, x_norm, mask=mask)

    # Memory barrier: ensure the store above is visible to subsequent loads
    # from the same pointer within this kernel instance.
    tl.debug_barrier()

    half_offsets = tl.arange(0, HALF_BLOCK)
    half_mask = half_offsets < half_dim

    # Even/odd positions in x_norm
    even_offsets = half_offsets * 2  # 0, 2, 4, ...
    odd_offsets = even_offsets + 1  # 1, 3, 5, ...
    even_mask = even_offsets < head_dim
    odd_mask = odd_offsets < head_dim

    # Reload even/odd pairs from the freshly stored normed values.
    x_norm_even = tl.load(
        X_ptr + row_idx * head_dim + even_offsets, mask=even_mask, other=0.0
    )
    x_norm_odd = tl.load(
        X_ptr + row_idx * head_dim + odd_offsets, mask=odd_mask, other=0.0
    )

    # Load cos/sin
    cos_val = tl.load(
        COS_ptr + row_idx * half_dim + half_offsets, mask=half_mask, other=0.0
    )
    sin_val = tl.load(
        SIN_ptr + row_idx * half_dim + half_offsets, mask=half_mask, other=0.0
    )

    # Promote to fp32 for precision
    xe_f = x_norm_even.to(tl.float32)
    xo_f = x_norm_odd.to(tl.float32)
    cos_f = cos_val.to(tl.float32)
    sin_f = sin_val.to(tl.float32)

    # Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    new_even = xe_f * cos_f - xo_f * sin_f
    new_odd = xo_f * cos_f + xe_f * sin_f

    # Store RoPE result in original dtype
    tl.store(
        X_ptr + row_idx * head_dim + even_offsets,
        new_even.to(x_row.dtype),
        mask=even_mask,
    )
    tl.store(
        X_ptr + row_idx * head_dim + odd_offsets,
        new_odd.to(x_row.dtype),
        mask=odd_mask,
    )


def triton_fused_qknorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply fused RMSNorm + 3D RoPE using a single Triton kernel.

    Replaces two sequential operations in the ZImage attention processor
    (``norm_q``/``norm_k`` followed by ``apply_rotary_emb``) with a single
    kernel that performs both passes in registers, avoiding intermediate
    DRAM round-trips.

    The RMSNorm is computed in float32 internally for numerical stability;
    the output is cast back to the input dtype before RoPE is applied.

    Args:
        x: Input tensor of shape ``(batch, seq, heads, head_dim)``.
           A contiguous copy is made internally when needed.
        weight: RMSNorm scale parameter of shape ``(head_dim,)``.
        freqs_cis: Precomputed complex frequencies of shape
                   ``(batch, seq, head_dim // 2)``, dtype ``torch.complex64``.
        eps: Numerical stability constant for RMSNorm (default ``1e-5``).

    Returns:
        RMSNorm-then-RoPE-applied tensor with the same shape and dtype as
        ``x``.

    Raises:
        ValueError: If ``head_dim`` is odd or the freqs_cis half-dim does not
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

    # Extract real/imag from complex freqs_cis
    cos = freqs_cis.real.contiguous()  # (batch, seq, half_dim)
    sin = freqs_cis.imag.contiguous()

    # Clone x to avoid modifying the caller's tensor (kernel operates in-place)
    out = x.contiguous().clone()
    out_flat = out.view(-1, head_dim)  # (batch*seq*heads, head_dim)
    weight = weight.contiguous()

    # Expand cos/sin: (batch, seq, half_dim) -> (batch*seq*heads, half_dim)
    cos_expanded = (
        cos.unsqueeze(2).expand(-1, -1, heads, -1).contiguous().view(-1, half_dim)
    )
    sin_expanded = (
        sin.unsqueeze(2).expand(-1, -1, heads, -1).contiguous().view(-1, half_dim)
    )

    n_rows = out_flat.shape[0]
    BLOCK_SIZE, num_warps = calculate_settings(head_dim)
    HALF_BLOCK, _ = calculate_settings(half_dim)

    _fused_qknorm_rope_kernel[(n_rows,)](
        out_flat,
        weight,
        cos_expanded,
        sin_expanded,
        head_dim=head_dim,
        half_dim=half_dim,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        HALF_BLOCK=HALF_BLOCK,
        num_warps=num_warps,
    )

    return out_flat.view(batch, seq, heads, head_dim)
