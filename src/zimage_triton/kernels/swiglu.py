"""Fused SwiGLU Triton kernel (forward-only, inference optimized).

Fuses the ``silu(gate) * up`` step of the ZImage FFN:

    FeedForward.forward(x) = w2(silu(w1(x)) * w3(x))

Here we fuse ``silu(gate) * up`` into a single Triton kernel, replacing
the two separate elementwise operations that would otherwise hit DRAM twice.

Optimized for ZImage S3-DiT FFN hidden_dim=10240 on SM120 (RTX 5090).
"""

import torch
import triton
import triton.language as tl

from zimage_triton.kernels.utils import calculate_settings


@triton.jit
def _swiglu_forward_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Triton JIT kernel for fused SwiGLU forward pass.

    Each program processes one row of the input tensors.  The SiLU activation
    is computed in float32 for numerical stability, then cast back to the
    input dtype before the elementwise multiply with ``up``.

    Args:
        gate_ptr: Pointer to gate tensor (2-D, row-major).
        up_ptr: Pointer to up tensor (2-D, row-major).
        out_ptr: Pointer to output tensor (2-D, row-major).
        stride: Row stride shared by all three tensors (they are congruent).
        n_cols: Number of columns (FFN hidden_dim).
        BLOCK_SIZE: Compile-time block size (next power of 2 >= n_cols).
    """
    program_id = tl.program_id(0).to(tl.int64)
    gate_ptr += program_id * stride
    up_ptr += program_id * stride
    out_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    gate = tl.load(gate_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(up_ptr + col_offsets, mask=mask, other=0)

    # SiLU(gate) = gate * sigmoid(gate), computed in fp32 for stability
    silu_gate = gate * tl.sigmoid(gate)

    # Cast back to the original dtype and multiply with up
    out = silu_gate.cast(up.dtype) * up
    tl.store(out_ptr + col_offsets, out, mask=mask)


def triton_swiglu_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute ``silu(gate) * up`` using a fused Triton kernel.

    Replaces the two-step ``F.silu(gate) * up`` with a single kernel pass
    that reads each tensor once and writes the result once, halving DRAM
    traffic compared to the naive PyTorch implementation.

    Args:
        gate: Gate projection tensor of shape ``(..., hidden_dim)``.
            Typically the output of ``w1(x)`` in the ZImage FFN.
        up: Up projection tensor of the same shape as ``gate``.
            Typically the output of ``w3(x)`` in the ZImage FFN.

    Returns:
        Output tensor ``silu(gate) * up`` with the same shape and dtype
        as the inputs.

    Raises:
        ValueError: If ``gate`` and ``up`` have different shapes.
    """
    if gate.shape != up.shape:
        raise ValueError(
            f"gate and up must have the same shape, "
            f"got gate={gate.shape} and up={up.shape}"
        )

    gate = gate.contiguous()
    up = up.contiguous()
    ori_shape = gate.shape
    n_cols = ori_shape[-1]

    gate_2d = gate.view(-1, n_cols)
    up_2d = up.view(-1, n_cols)
    out_2d = torch.empty_like(gate_2d)
    n_rows = gate_2d.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_kernel[(n_rows,)](
        gate_2d,
        up_2d,
        out_2d,
        out_2d.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return out_2d.view(*ori_shape)


class TritonSwiGLU(torch.nn.Module):
    """Drop-in SwiGLU activation backed by a fused Triton kernel.

    Computes ``silu(gate) * up`` in a single kernel launch, avoiding
    the intermediate DRAM writes of the sequential PyTorch version.

    Example::

        swiglu = TritonSwiGLU()
        gate = w1(x)   # (B, T, hidden_dim)
        up   = w3(x)   # (B, T, hidden_dim)
        out  = swiglu(gate, up)  # feeds into w2
    """

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Apply fused SwiGLU.

        Args:
            gate: Gate projection tensor ``(..., hidden_dim)``.
            up: Up projection tensor ``(..., hidden_dim)``.

        Returns:
            ``silu(gate) * up`` with the same shape and dtype as inputs.
        """
        return triton_swiglu_forward(gate, up)
