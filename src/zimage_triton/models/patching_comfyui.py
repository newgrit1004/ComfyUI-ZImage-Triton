"""Patch ZImage model (ComfyUI NextDiT architecture) with Triton kernels.

Adapts the patching logic from patching.py for ComfyUI's model structure:
- ComfyUI uses NextDiT (from comfy.ldm.lumina.model), not diffusers'
  ZImageTransformer2DModel
- RMSNorm is comfy.ops.RMSNorm (detected by class name, not isinstance)
- Attention uses JointAttention with fused QKV, not ZSingleStreamAttnProcessor
- RoPE format is 2x2 rotation matrix (b, n, d//2, 2, 2), not complex64
- FFN uses FeedForward with w1/w2/w3 (identical structure to diffusers)

All replacements share original weight parameters (zero additional VRAM).

Phase 1 additions: AdaLN modulation fusion and FusedNormGateResidual.
"""

import logging
import types
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RMSNorm replacement
# ---------------------------------------------------------------------------


def _replace_all_rms_norm_comfyui(model: nn.Module, stats: dict) -> None:
    """Walk model tree and replace every RMSNorm with TritonRMSNorm.

    ComfyUI's RMSNorm comes from comfy.ops and is detected by class name
    rather than isinstance check (avoids importing comfy internals).

    Targets: attention_norm1/2, ffn_norm1/2, q_norm, k_norm, cap_embedder[0].
    """
    from zimage_triton.kernels.rms_norm import TritonRMSNorm

    for name, module in list(model.named_modules()):
        if isinstance(module, TritonRMSNorm):
            continue

        cls_name = type(module).__name__
        if cls_name != "RMSNorm":
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue

        # Navigate to parent module
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1]

        old = getattr(parent, attr)
        hidden_size = old.weight.shape[0]
        eps = getattr(old, "eps", 1e-5)
        if eps is None:
            eps = 1e-5

        new_norm = TritonRMSNorm(hidden_size, eps=eps)
        new_norm.weight = old.weight  # share parameter, zero VRAM
        new_norm._original_module = old  # store for restore
        setattr(parent, attr, new_norm)
        stats["rms_norm"] += 1


# ---------------------------------------------------------------------------
# FFN SwiGLU patching
# ---------------------------------------------------------------------------


def _make_ffn_forward_comfyui(ffn: nn.Module):
    """Return new _forward_silu_gating for FeedForward using Triton SwiGLU.

    ComfyUI's FeedForward._forward_silu_gating does:
        clamp_fp16(F.silu(x1) * x3)

    We replace with triton_swiglu_forward(x1, x3) which fuses silu+multiply.
    clamp_fp16 is only needed for fp16; RTX 5090 uses bf16 by default.
    """
    from zimage_triton.kernels.swiglu import triton_swiglu_forward

    def _forward_silu_gating(self, x1: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        return triton_swiglu_forward(x1, x3)

    return types.MethodType(_forward_silu_gating, ffn)


def _patch_all_ffn_comfyui(model: nn.Module, stats: dict) -> None:
    """Patch every FeedForward._forward_silu_gating to use Triton SwiGLU."""
    for module in model.modules():
        cls_name = type(module).__name__
        if (
            cls_name == "FeedForward"
            and hasattr(module, "w1")
            and hasattr(module, "w2")
            and hasattr(module, "w3")
        ):
            # Store original for restoration (prevents double-patching)
            if not hasattr(module, "_original_silu_gating"):
                module._original_silu_gating = module._forward_silu_gating
            module._forward_silu_gating = _make_ffn_forward_comfyui(module)
            stats["swiglu"] += 1


# ---------------------------------------------------------------------------
# Attention patching — Fused QKNorm + RoPE
# ---------------------------------------------------------------------------


def _convert_rope_matrix_to_complex(freqs_cis: torch.Tensor) -> torch.Tensor:
    """Convert ComfyUI's 2x2 rotation matrix RoPE to complex64 format.

    ComfyUI rope format: (b, n, [1,] d//2, 2, 2) where the 2x2 matrix is:
        [[cos, -sin],
         [sin,  cos]]

    The optional singleton dim at position 2 is a broadcast dimension for
    attention heads, inserted by EmbedND.unsqueeze(1) + movedim(1, 2).

    Existing Triton kernel expects: complex64 (batch, seq, d//2)
    where real=cos, imag=sin.
    """
    cos = freqs_cis[..., 0, 0]  # (b, n, [1,] d//2)
    sin = freqs_cis[..., 1, 0]  # (b, n, [1,] d//2)
    # Squeeze broadcast head dimension if present (6D → 5D input)
    if cos.ndim == 4 and cos.shape[2] == 1:
        cos = cos.squeeze(2)
        sin = sin.squeeze(2)
    return torch.complex(cos.float(), sin.float())


def _make_fused_attention_forward(attn: nn.Module):
    """Replace JointAttention.forward with fused QKNorm+RoPE version.

    Replaces the sequence:
        q_norm(xq) → k_norm(xk) → apply_rope(xq, xk, freqs_cis)
    with:
        triton_fused_qknorm_rope(xq, q_norm.weight, freqs_complex)
        triton_fused_qknorm_rope(xk, k_norm.weight, freqs_complex)

    Reduces 4 kernel launches to 2.
    """
    from comfy.ldm.modules.attention import optimized_attention_masked

    from zimage_triton.kernels.fused_qknorm_rope import triton_fused_qknorm_rope

    # Cache attention config from the module
    n_local_heads = attn.n_local_heads
    n_local_kv_heads = attn.n_local_kv_heads
    head_dim = attn.head_dim
    n_rep = n_local_heads // n_local_kv_heads

    # Get norm weights (may be TritonRMSNorm or original RMSNorm)
    q_norm_weight = attn.q_norm.weight
    k_norm_weight = attn.k_norm.weight
    q_norm_eps = getattr(attn.q_norm, "eps", 1e-5) or 1e-5
    k_norm_eps = getattr(attn.k_norm, "eps", 1e-5) or 1e-5

    # Split sizes for fused QKV
    split_sizes = [
        n_local_heads * head_dim,
        n_local_kv_heads * head_dim,
        n_local_kv_heads * head_dim,
    ]

    def _forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        transformer_options: dict = {},
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq, xk, xv = torch.split(self.qkv(x), split_sizes, dim=-1)
        xq = xq.view(bsz, seqlen, n_local_heads, head_dim)
        xk = xk.view(bsz, seqlen, n_local_kv_heads, head_dim)
        xv = xv.view(bsz, seqlen, n_local_kv_heads, head_dim)

        # Convert RoPE format: 2x2 matrix → complex64
        freqs_complex = _convert_rope_matrix_to_complex(freqs_cis)

        # Fused QKNorm + RoPE (2 kernel launches instead of 4)
        xq = triton_fused_qknorm_rope(xq, q_norm_weight, freqs_complex, eps=q_norm_eps)
        xk = triton_fused_qknorm_rope(xk, k_norm_weight, freqs_complex, eps=k_norm_eps)

        # GQA key/value expansion
        if n_rep > 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        output = optimized_attention_masked(
            xq.movedim(1, 2),
            xk.movedim(1, 2),
            xv.movedim(1, 2),
            n_local_heads,
            x_mask,
            skip_reshape=True,
            transformer_options=transformer_options,
        )

        return self.out(output)

    return types.MethodType(_forward, attn)


def _make_plain_rope_attention_forward(attn: nn.Module):
    """Replace JointAttention.forward with plain Triton RoPE version.

    Used when enable_fused_qknorm_rope=False. Keeps norm_q/norm_k as-is
    (already replaced with TritonRMSNorm if enable_fused_norm=True)
    and only replaces apply_rope with triton_rope_3d.
    """
    from comfy.ldm.modules.attention import optimized_attention_masked

    from zimage_triton.kernels.rope_3d import triton_rope_3d

    n_local_heads = attn.n_local_heads
    n_local_kv_heads = attn.n_local_kv_heads
    head_dim = attn.head_dim
    n_rep = n_local_heads // n_local_kv_heads

    split_sizes = [
        n_local_heads * head_dim,
        n_local_kv_heads * head_dim,
        n_local_kv_heads * head_dim,
    ]

    def _forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        transformer_options: dict = {},
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq, xk, xv = torch.split(self.qkv(x), split_sizes, dim=-1)
        xq = xq.view(bsz, seqlen, n_local_heads, head_dim)
        xk = xk.view(bsz, seqlen, n_local_kv_heads, head_dim)
        xv = xv.view(bsz, seqlen, n_local_kv_heads, head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # Convert RoPE format and apply
        freqs_complex = _convert_rope_matrix_to_complex(freqs_cis)
        xq = triton_rope_3d(xq, freqs_complex)
        xk = triton_rope_3d(xk, freqs_complex)

        if n_rep > 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        output = optimized_attention_masked(
            xq.movedim(1, 2),
            xk.movedim(1, 2),
            xv.movedim(1, 2),
            n_local_heads,
            x_mask,
            skip_reshape=True,
            transformer_options=transformer_options,
        )

        return self.out(output)

    return types.MethodType(_forward, attn)


def _patch_attention_comfyui(
    model: nn.Module,
    stats: dict,
    fused: bool = True,
) -> None:
    """Patch every JointAttention.forward in the model.

    Args:
        model: Root NextDiT module.
        stats: Mutable replacement-count dict.
        fused: If True, use fused QKNorm+RoPE kernel. If False, use plain
               Triton RoPE with separate norm.
    """
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name != "JointAttention":
            continue
        if not hasattr(module, "qkv"):
            continue
        # Skip if q_norm/k_norm are Identity (no QK-Norm to fuse)
        if not hasattr(module, "q_norm") or not hasattr(module.q_norm, "weight"):
            continue

        # Store original forward for restoration (prevents double-patching)
        if not hasattr(module, "_original_forward"):
            module._original_forward = module.forward

        if fused:
            module.forward = _make_fused_attention_forward(module)
            stats["fused_qknorm_rope"] += 1
        else:
            module.forward = _make_plain_rope_attention_forward(module)
            stats["rope_3d"] += 1


# ---------------------------------------------------------------------------
# Block-level forward patching (AdaLN + FusedNormGateResidual)
# ---------------------------------------------------------------------------


def _make_block_forward(
    block: nn.Module,
    dim: int,
    use_adaln: bool,
    use_fused_nr: bool,
) -> types.MethodType:
    """Build a patched JointTransformerBlock.forward.

    Fuses AdaLN modulation (chunk+tanh+scale) and/or post-attention/FFN
    norm+gate+residual into single Triton kernel passes. Falls back to
    the original forward for omni mode (timestep_zero_index is not None).
    """
    if use_adaln:
        from zimage_triton.kernels.adaln_modulation import triton_adaln_modulation
    if use_fused_nr:
        from zimage_triton.kernels.fused_norm_residual import (
            triton_fused_norm_gate_residual,
        )

    # Cache norm weights/eps for the fused kernel
    an2_w = block.attention_norm2.weight
    an2_eps = getattr(block.attention_norm2, "eps", 1e-5) or 1e-5
    fn2_w = block.ffn_norm2.weight
    fn2_eps = getattr(block.ffn_norm2, "eps", 1e-5) or 1e-5

    def _forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        timestep_zero_index=None,
        transformer_options={},
    ) -> torch.Tensor:
        # Omni mode: fall back to original (slice-based modulate/apply_gate)
        if timestep_zero_index is not None:
            return self._original_forward(
                x,
                x_mask,
                freqs_cis,
                adaln_input,
                timestep_zero_index,
                transformer_options,
            )

        assert adaln_input is not None

        # --- AdaLN modulation ---
        if use_adaln:
            raw = self.adaLN_modulation(adaln_input)
            # scale already has 1+ applied, gate already has tanh applied
            s_msa, g_msa, s_mlp, g_mlp = triton_adaln_modulation(raw, dim)
        else:
            s_msa, g_msa, s_mlp, g_mlp = self.adaLN_modulation(adaln_input).chunk(
                4, dim=1
            )
            g_msa = g_msa.tanh()
            g_mlp = g_mlp.tanh()

        # --- Attention branch ---
        if use_adaln:
            attn_in = self.attention_norm1(x) * s_msa.unsqueeze(1)
        else:
            attn_in = self.attention_norm1(x) * (1 + s_msa.unsqueeze(1))
        attn_out = self.attention(
            attn_in,
            x_mask,
            freqs_cis,
            transformer_options=transformer_options,
        )

        if use_fused_nr:
            x = triton_fused_norm_gate_residual(
                attn_out,
                x,
                g_msa.unsqueeze(1),
                an2_w,
                an2_eps,
            )
        else:
            x = x + g_msa.unsqueeze(1) * self.attention_norm2(attn_out)

        # --- FFN branch ---
        if use_adaln:
            ffn_in = self.ffn_norm1(x) * s_mlp.unsqueeze(1)
        else:
            ffn_in = self.ffn_norm1(x) * (1 + s_mlp.unsqueeze(1))
        ffn_out = self.feed_forward(ffn_in)

        if use_fused_nr:
            x = triton_fused_norm_gate_residual(
                ffn_out,
                x,
                g_mlp.unsqueeze(1),
                fn2_w,
                fn2_eps,
            )
        else:
            x = x + g_mlp.unsqueeze(1) * self.ffn_norm2(ffn_out)

        return x

    return types.MethodType(_forward, block)


def _patch_block_forward_comfyui(
    model: nn.Module,
    stats: dict,
    enable_adaln: bool = True,
    enable_fused_norm_residual: bool = True,
) -> None:
    """Patch JointTransformerBlock.forward with fused block-level ops."""
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name != "JointTransformerBlock":
            continue
        if not getattr(module, "modulation", False):
            continue
        if not hasattr(module, "dim"):
            continue

        # Store original forward for omni-mode fallback
        if not hasattr(module, "_original_forward"):
            module._original_forward = module.forward

        module.forward = _make_block_forward(
            module,
            module.dim,
            enable_adaln,
            enable_fused_norm_residual,
        )
        if enable_adaln:
            stats["adaln_modulation"] += 1
        if enable_fused_norm_residual:
            stats["fused_norm_gate_residual"] += 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_triton_kernels_comfyui(
    model: nn.Module,
    enable_fused_norm: bool = True,
    enable_swiglu: bool = True,
    enable_fused_qknorm_rope: bool = True,
    enable_fused_adaln: bool = True,
    enable_fused_norm_residual: bool = True,
) -> dict[str, int]:
    """Apply Triton kernel replacements to a ComfyUI NextDiT model.

    Walks the model graph and performs replacements in order:

    1. Replace all RMSNorm modules with TritonRMSNorm (weight shared).
    2. Patch FeedForward._forward_silu_gating to use triton_swiglu_forward.
    3. Patch JointAttention.forward to use fused QKNorm+RoPE kernel
       (or plain triton_rope_3d as fallback).
    4. Patch JointTransformerBlock.forward with fused AdaLN modulation
       and/or fused norm+gate+residual.

    All replacements share original weight parameters — zero additional
    GPU memory allocated.

    Args:
        model: A NextDiT model (from ComfyUI's comfy.ldm.lumina.model).
        enable_fused_norm: Replace RMSNorm modules with TritonRMSNorm.
        enable_swiglu: Fuse SiLU+gate in FFN with Triton kernel.
        enable_fused_qknorm_rope: Use fused QKNorm+RoPE kernel in attention.
            If False, uses plain triton_rope_3d with separate norm.
        enable_fused_adaln: Fuse AdaLN chunk+tanh+(1+scale) in one kernel.
        enable_fused_norm_residual: Fuse post-attn/FFN norm+gate+residual.

    Returns:
        Dict mapping replacement type to count.
    """
    stats: dict[str, int] = {
        "rms_norm": 0,
        "swiglu": 0,
        "rope_3d": 0,
        "fused_qknorm_rope": 0,
        "adaln_modulation": 0,
        "fused_norm_gate_residual": 0,
    }

    if enable_fused_norm:
        _replace_all_rms_norm_comfyui(model, stats)

    if enable_swiglu:
        _patch_all_ffn_comfyui(model, stats)

    _patch_attention_comfyui(model, stats, fused=enable_fused_qknorm_rope)

    if enable_fused_adaln or enable_fused_norm_residual:
        _patch_block_forward_comfyui(
            model,
            stats,
            enable_fused_adaln,
            enable_fused_norm_residual,
        )

    logger.info("ComfyUI Triton patching complete: %s", stats)
    return stats
