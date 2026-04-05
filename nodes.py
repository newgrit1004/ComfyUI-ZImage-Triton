"""ComfyUI node definitions for ZImage Triton kernel acceleration."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _restore_original_forwards(transformer) -> None:  # type: ignore[no-untyped-def]
    """Restore all monkey-patched forward methods to their originals.

    Called before re-patching with a different config so that stale
    patches from a previous run do not accumulate or conflict.

    Restores three categories of patches:
    1. Block/Attention forward methods (stored as _original_forward)
    2. FFN _forward_silu_gating methods (stored as _original_silu_gating)
    3. RMSNorm module replacements (stored as _original_module)
    """
    import torch.nn as nn

    # Restore INT8-quantized Linear modules first (weight, buffers, forward)
    if getattr(transformer, "_int8_quantized", False):
        from zimage_triton.quantization import restore_int8_quantization

        n_restored = restore_int8_quantization(transformer)
        logger.info(
            "[ZImage-Triton] Restored %d INT8-quantized modules.", n_restored
        )

    # Restore patched forward methods (Attention, Block)
    for module in transformer.modules():
        if hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            delattr(module, "_original_forward")

        # Restore FFN _forward_silu_gating (patched by SwiGLU kernel)
        if hasattr(module, "_original_silu_gating"):
            module._forward_silu_gating = module._original_silu_gating
            delattr(module, "_original_silu_gating")

    # Restore RMSNorm modules replaced with TritonRMSNorm.
    # Each TritonRMSNorm stores the original module in _original_module.
    for name, child in list(transformer.named_modules()):
        original = getattr(child, "_original_module", None)
        if original is not None:
            parts = name.split(".")
            parent: nn.Module = transformer
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], original)

    transformer._triton_patched = False
    transformer._triton_config = None
    logger.info("[ZImage-Triton] Restored original forwards.")


def _log_patch_state(transformer, label: str) -> None:
    """Log module-level patching state for debugging base model issues."""
    n_original_fwd = 0
    n_original_silu = 0
    n_triton_norm = 0
    n_layers = 0

    for module in transformer.modules():
        if hasattr(module, "_original_forward"):
            n_original_fwd += 1
        if hasattr(module, "_original_silu_gating"):
            n_original_silu += 1
        if type(module).__name__ == "TritonRMSNorm":
            n_triton_norm += 1
        if type(module).__name__ == "JointTransformerBlock":
            n_layers += 1

    patched = getattr(transformer, "_triton_patched", False)
    config = getattr(transformer, "_triton_config", None)
    logger.debug(
        "[ZImage-Triton] %s: layers=%d, _triton_patched=%s, "
        "config=%s, _original_forward=%d, _original_silu=%d, "
        "TritonRMSNorm=%d",
        label,
        n_layers,
        patched,
        config,
        n_original_fwd,
        n_original_silu,
        n_triton_norm,
    )


class ZImageTritonApply:
    """Apply Triton kernel optimizations to a ZImage S3-DiT model.

    Replaces standard PyTorch operations in the ZImage transformer with
    custom Triton kernels for faster inference. All replacements share
    original weight parameters (zero additional VRAM).

    Compatible with LoRA, ControlNet, and dynamic input shapes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "enable_int8": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enable W8A8+Hadamard INT8 quantization. "
                        "ON (default): 1.29x speedup, VRAM -2.7GB. "
                        "OFF: Triton-only, best quality (1.08x speedup).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_triton"
    CATEGORY = "ZImage-Triton"
    DESCRIPTION = (
        "Accelerate ZImage S3-DiT with Triton kernels + INT8 quantization. "
        "Default: 1.29x speedup with Hadamard rotation for quality. "
        "Uncheck INT8 for best quality (1.08x speedup)."
    )

    def apply_triton(
        self,
        model,
        enable_int8: bool = True,
    ):
        from zimage_triton.models.patching_comfyui import apply_triton_kernels_comfyui

        transformer = model.model.diffusion_model

        # Validate model type
        if not hasattr(transformer, "layers"):
            logger.warning(
                "[ZImage-Triton] Model does not appear to be ZImage/Lumina2 "
                "(no .layers attribute). Skipping Triton patching."
            )
            return (model,)

        # Debug: log pre-patch module state
        _log_patch_state(transformer, "pre-patch")

        # Internal kernel defaults (not exposed to UI)
        enable_fused_norm = True
        enable_swiglu = True
        enable_fused_qknorm_rope = True
        enable_fused_adaln = False
        enable_fused_norm_residual = True

        # Build current config tuple for change detection
        current_config = (
            enable_fused_norm,
            enable_swiglu,
            enable_fused_qknorm_rope,
            enable_fused_adaln,
            enable_fused_norm_residual,
            enable_int8,
        )

        # Idempotent: skip if already patched with identical config
        prev_config = getattr(transformer, "_triton_config", None)
        if getattr(transformer, "_triton_patched", False):
            if prev_config == current_config:
                logger.info(
                    "[ZImage-Triton] Already patched with same config, " "skipping."
                )
                return (model,)
            # Config changed: restore original forwards before re-patching
            logger.info(
                "[ZImage-Triton] Config changed (%s -> %s), restoring "
                "originals before re-patching.",
                prev_config,
                current_config,
            )
            _restore_original_forwards(transformer)

        stats = apply_triton_kernels_comfyui(
            transformer,
            enable_fused_norm=enable_fused_norm,
            enable_swiglu=enable_swiglu,
            enable_fused_qknorm_rope=enable_fused_qknorm_rope,
            enable_fused_adaln=enable_fused_adaln,
            enable_fused_norm_residual=enable_fused_norm_residual,
        )

        transformer._triton_patched = True
        transformer._triton_config = current_config
        transformer._triton_stats = stats

        # Debug: log post-patch module state
        _log_patch_state(transformer, "post-patch")

        logger.info(
            "[ZImage-Triton] Applied Triton kernels: %s",
            stats,
        )

        # Apply INT8 quantization (W8A8+Hadamard) if enabled
        if enable_int8 and not getattr(transformer, "_int8_quantized", False):
            from zimage_triton.quantization import apply_int8_quantization

            int8_stats = apply_int8_quantization(
                transformer,
                mode="w8a8_hadamard",
            )
            transformer._int8_quantized = True
            transformer._int8_stats = int8_stats

            # --- ComfyUI VRAM state correction ---
            # state_dict hook (_int8_state_dict_hook) excludes CPU weights
            # so module_size() reports correct GPU-only size (~7GB).
            # _int8_apply pins _w_int8/_w_scale to GPU during partial offload
            # so torch._int_mm never falls back to CPU.

            # Invalidate model size cache so ComfyUI recalculates using
            # the corrected state_dict (CPU weights excluded by hook).
            model.size = 0

            # Upgrade to custom ModelPatcher that keeps INT8 weights on
            # CPU during LoRA patching, preventing ~12 GB VRAM waste.
            from zimage_triton.model_patcher import ZImageTritonModelPatcher

            if not isinstance(model, ZImageTritonModelPatcher):
                model.__class__ = ZImageTritonModelPatcher
                logger.info("[ZImage-Triton] ModelPatcher upgraded.")

            logger.info(
                "[ZImage-Triton] INT8 W8A8+Hadamard: %s",
                int8_stats,
            )

        return (model,)


