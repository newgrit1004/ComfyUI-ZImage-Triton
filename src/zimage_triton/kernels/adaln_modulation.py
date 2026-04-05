"""Fused AdaLN modulation Triton kernel (forward-only, inference optimized).

Fuses the post-linear AdaLN modulation ops in ZImageTransformerBlock:

    raw = adaLN_modulation(adaln_input).unsqueeze(1)   # (B, 1, 4*dim)
    scale_msa, gate_msa, scale_mlp, gate_mlp = raw.chunk(4, dim=2)
    gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
    scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

The kernel fuses chunk(4) + tanh(gates) + (1+scales) into a single pass,
eliminating three extra DRAM round-trips compared to the sequential version.

Optimized for ZImage S3-DiT dim=3840 (4*dim=15360) on SM120 (RTX 5090).
"""

import torch
import triton
import triton.language as tl

from zimage_triton.kernels.utils import calculate_settings

try:
    from triton.language.extra.libdevice import tanh as libdevice_tanh
except ModuleNotFoundError:
    from triton.language.extra.cuda.libdevice import tanh as libdevice_tanh


@triton.jit
def _adaln_modulation_kernel(
    scale_msa_ptr,
    gate_msa_ptr,
    scale_mlp_ptr,
    gate_mlp_ptr,
    raw_ptr,
    raw_row_stride,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Triton JIT kernel for fused AdaLN modulation.

    Each program processes one row of the flattened raw tensor (B*1 rows).
    The row contains 4*dim elements split into four consecutive chunks of
    size dim: [scale_msa | gate_msa | scale_mlp | gate_mlp].

    Operations applied per chunk:
    - scale_msa -> scale_msa + 1.0
    - gate_msa  -> tanh(gate_msa)   (computed in fp32 for stability)
    - scale_mlp -> scale_mlp + 1.0
    - gate_mlp  -> tanh(gate_mlp)   (computed in fp32 for stability)

    Args:
        scale_msa_ptr: Output pointer for scale_msa (n_rows * dim elements).
        gate_msa_ptr:  Output pointer for gate_msa  (n_rows * dim elements).
        scale_mlp_ptr: Output pointer for scale_mlp (n_rows * dim elements).
        gate_mlp_ptr:  Output pointer for gate_mlp  (n_rows * dim elements).
        raw_ptr: Pointer to the raw input tensor (2-D, row-major, n_rows x 4*dim).
        raw_row_stride: Row stride of the raw tensor (should be 4*dim).
        dim: Size of each chunk (ZImage: 3840). Compile-time constant.
        BLOCK_SIZE: Compile-time block size (next power of 2 >= dim).
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < dim

    base = raw_ptr + row_idx * raw_row_stride

    # Load 4 consecutive chunks from the raw row
    scale_msa = tl.load(base + col_offsets, mask=mask, other=0.0)
    gate_msa = tl.load(base + dim + col_offsets, mask=mask, other=0.0)
    scale_mlp = tl.load(base + 2 * dim + col_offsets, mask=mask, other=0.0)
    gate_mlp = tl.load(base + 3 * dim + col_offsets, mask=mask, other=0.0)

    dtype = scale_msa.dtype

    # tanh on gates in fp32 for numerical stability, then cast back
    gate_msa = libdevice_tanh(gate_msa.to(tl.float32)).to(dtype)
    gate_mlp = libdevice_tanh(gate_mlp.to(tl.float32)).to(dtype)

    # 1 + scale
    scale_msa = scale_msa + 1.0
    scale_mlp = scale_mlp + 1.0

    # Each output tensor is (n_rows, dim) contiguous, so stride == dim
    out_stride = dim
    tl.store(scale_msa_ptr + row_idx * out_stride + col_offsets, scale_msa, mask=mask)
    tl.store(gate_msa_ptr + row_idx * out_stride + col_offsets, gate_msa, mask=mask)
    tl.store(scale_mlp_ptr + row_idx * out_stride + col_offsets, scale_mlp, mask=mask)
    tl.store(gate_mlp_ptr + row_idx * out_stride + col_offsets, gate_mlp, mask=mask)


def triton_adaln_modulation(
    raw: torch.Tensor,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused AdaLN modulation: chunk(4) + tanh(gates) + (1+scales).

    Takes the raw output from ``adaLN_modulation`` (a Linear layer with
    out_features=4*dim) and produces the four modulation tensors used in
    ZImageTransformerBlock in a single Triton kernel pass.

    Args:
        raw: Linear output tensor of shape ``(B, 1, 4*dim)`` or ``(B, 4*dim)``.
            The last dimension must equal ``4 * dim``.
        dim: Size of each output chunk (ZImage S3-DiT: 3840).

    Returns:
        Tuple ``(scale_msa, gate_msa, scale_mlp, gate_mlp)`` where each
        element has the same leading shape as ``raw`` with the last dimension
        replaced by ``dim``.  Concretely for input ``(B, 1, 4*dim)`` the
        outputs each have shape ``(B, 1, dim)``.

    Raises:
        ValueError: If the last dimension of ``raw`` is not ``4 * dim``.
    """
    expected_last = 4 * dim
    if raw.shape[-1] != expected_last:
        raise ValueError(
            f"raw.shape[-1] must equal 4 * dim = {expected_last}, "
            f"got raw.shape={raw.shape}"
        )

    raw = raw.contiguous()
    leading_shape = raw.shape[:-1]  # e.g. (B, 1) or (B,)
    n_rows = raw.numel() // (4 * dim)

    raw_2d = raw.view(n_rows, 4 * dim)

    BLOCK_SIZE, num_warps = calculate_settings(dim)

    # Allocate four output buffers of shape (n_rows, dim)
    scale_msa = torch.empty(n_rows, dim, dtype=raw.dtype, device=raw.device)
    gate_msa = torch.empty(n_rows, dim, dtype=raw.dtype, device=raw.device)
    scale_mlp = torch.empty(n_rows, dim, dtype=raw.dtype, device=raw.device)
    gate_mlp = torch.empty(n_rows, dim, dtype=raw.dtype, device=raw.device)

    _adaln_modulation_kernel[(n_rows,)](
        scale_msa,
        gate_msa,
        scale_mlp,
        gate_mlp,
        raw_2d,
        raw_2d.stride(0),
        dim=dim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    # Restore leading dimensions so output shapes match input leading shape
    out_shape = (*leading_shape, dim)
    return (
        scale_msa.view(out_shape),
        gate_msa.view(out_shape),
        scale_mlp.view(out_shape),
        gate_mlp.view(out_shape),
    )


class TritonAdaLNModulation(torch.nn.Module):
    """Drop-in AdaLN modulation module backed by a fused Triton kernel.

    Wraps ``triton_adaln_modulation`` as an ``nn.Module`` for easy integration
    into ZImageTransformerBlock.

    Args:
        dim: Per-chunk dimension size (ZImage S3-DiT: 3840).

    Example::

        modulate = TritonAdaLNModulation(dim=3840)
        raw = adaLN_modulation(adaln_input).unsqueeze(1)  # (B, 1, 15360)
        scale_msa, gate_msa, scale_mlp, gate_mlp = modulate(raw)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(
        self,
        raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply fused AdaLN modulation.

        Args:
            raw: Linear output of shape ``(B, 1, 4*dim)`` or ``(B, 4*dim)``.

        Returns:
            Tuple ``(scale_msa, gate_msa, scale_mlp, gate_mlp)``, each with
            last dimension ``dim``.
        """
        return triton_adaln_modulation(raw, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
