"""Monkey-patch ZImage model to use Triton kernels.

Replaces standard PyTorch modules in a ZImageTransformer2DModel with
Triton-backed equivalents. All replacements share the original weight
parameters (no extra VRAM consumed).

Supported replacements
----------------------
- RMSNorm  -> TritonRMSNorm  (attention_norm1/2, ffn_norm1/2, cap_embedder[0],
                               per-head norm_q / norm_k inside Attention modules)
- FeedForward.forward  -> fused SwiGLU via triton_swiglu_forward
- ZSingleStreamAttnProcessor.__call__  -> fused QKNorm+RoPE via
  triton_fused_qknorm_rope  (when enable_fused_qknorm_rope=True)
  OR plain triton_rope_3d  (when enable_fused_qknorm_rope=False)
"""

import logging
import types

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _replace_rms_norm(parent: nn.Module, attr_name: str, stats: dict) -> None:
    """Replace a single RMSNorm attribute on *parent* with TritonRMSNorm.

    The weight parameter is shared (no copy) so no additional VRAM is used.

    Args:
        parent: Module that owns the attribute.
        attr_name: Name of the attribute holding the RMSNorm.
        stats: Mutable dict used for counting replacements.
    """
    from zimage_triton.kernels.rms_norm import TritonRMSNorm

    old = getattr(parent, attr_name, None)
    if old is None:
        return

    # Accept both diffusers RMSNorm and any nn.Module with a .weight parameter
    # whose name contains "RMSNorm" or that already is a TritonRMSNorm.
    if isinstance(old, TritonRMSNorm):
        return  # already replaced

    hidden_size = old.weight.shape[0]
    eps = getattr(old, "eps", 1e-5)

    new_norm = TritonRMSNorm(hidden_size, eps=eps)
    new_norm.weight = old.weight  # share parameter
    setattr(parent, attr_name, new_norm)
    stats["rms_norm"] += 1


def _replace_all_rms_norm(model: nn.Module, stats: dict) -> None:
    """Walk the full model tree and replace every RMSNorm with TritonRMSNorm.

    Handles:
    - ZImageTransformerBlock: attention_norm1, attention_norm2, ffn_norm1,
      ffn_norm2
    - cap_embedder: Sequential whose first child is an RMSNorm
    - Attention modules: norm_q, norm_k  (per-head RMSNorm)

    Args:
        model: Root module (ZImageTransformer2DModel).
        stats: Mutable replacement-count dict.
    """
    from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm

    from zimage_triton.kernels.rms_norm import TritonRMSNorm

    # Walk the entire module tree and replace any RMSNorm that has a weight
    # (i.e. elementwise-affine=True).
    for name, module in list(model.named_modules()):
        # Skip modules that are already replaced.
        if isinstance(module, TritonRMSNorm):
            continue

        # Only care about DiffusersRMSNorm instances with a weight parameter.
        if not isinstance(module, DiffusersRMSNorm):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue

        # Navigate to the *parent* so we can setattr on it.
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        attr = parts[-1]
        _replace_rms_norm(parent, attr, stats)


# ---------------------------------------------------------------------------
# FFN patching
# ---------------------------------------------------------------------------


def _make_ffn_forward(ffn: nn.Module):
    """Return a new forward function for a FeedForward module using SwiGLU.

    The original forward is::

        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    The replacement fuses ``silu(w1(x)) * w3(x)`` into one Triton kernel.

    Args:
        ffn: The FeedForward module (holds w1, w2, w3 as Linear layers).

    Returns:
        A bound method suitable for assignment to ``ffn.forward``.
    """
    from zimage_triton.kernels.swiglu import triton_swiglu_forward

    def _forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        gate = self.w1(x)
        up = self.w3(x)
        return self.w2(triton_swiglu_forward(gate, up))

    return types.MethodType(_forward, ffn)


def _patch_all_ffn(model: nn.Module, stats: dict) -> None:
    """Patch every FeedForward.forward in the model to use triton_swiglu_forward.

    Args:
        model: Root module.
        stats: Mutable replacement-count dict.
    """
    # Import the FeedForward class from the same source the model uses.
    # We use the class name string as a fallback to avoid a hard import of the
    # model file.
    for module in model.modules():
        cls_name = type(module).__name__
        if (
            cls_name == "FeedForward"
            and hasattr(module, "w1")
            and hasattr(module, "w2")
            and hasattr(module, "w3")
        ):
            module.forward = _make_ffn_forward(module)
            stats["swiglu"] += 1


# ---------------------------------------------------------------------------
# Attention processor patching – fused QKNorm + RoPE
# ---------------------------------------------------------------------------


def _make_fused_attn_call(processor: object):
    """Return a new __call__ for ZSingleStreamAttnProcessor using fused QKNorm+RoPE.

    Replaces the inner ``norm_q -> apply_rotary_emb`` and
    ``norm_k -> apply_rotary_emb`` sequence with two calls to
    ``triton_fused_qknorm_rope``.

    Args:
        processor: The ZSingleStreamAttnProcessor instance to patch.

    Returns:
        A bound method suitable for ``processor.__call__ = ...``.
    """
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    from zimage_triton.kernels.fused_qknorm_rope import triton_fused_qknorm_rope

    def _call(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        freqs_cis=None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if freqs_cis is not None:
            # Fused QKNorm + RoPE (requires both norm and freqs_cis)
            if attn.norm_q is not None:
                query = triton_fused_qknorm_rope(
                    query,
                    attn.norm_q.weight,
                    freqs_cis,
                    eps=getattr(attn.norm_q, "eps", 1e-5),
                )
            if attn.norm_k is not None:
                key = triton_fused_qknorm_rope(
                    key,
                    attn.norm_k.weight,
                    freqs_cis,
                    eps=getattr(attn.norm_k, "eps", 1e-5),
                )
        else:
            # No freqs_cis: fall back to plain norm only
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)

        return output

    return types.MethodType(_call, processor)


def _make_plain_rope_attn_call(processor: object):
    """Return a new __call__ for ZSingleStreamAttnProcessor using triton_rope_3d.

    Used when enable_fused_qknorm_rope=False. Keeps norm_q / norm_k as-is
    (already replaced with TritonRMSNorm by _replace_all_rms_norm) and
    replaces only the ``apply_rotary_emb`` with triton_rope_3d.

    Args:
        processor: The ZSingleStreamAttnProcessor instance to patch.

    Returns:
        A bound method suitable for ``processor.__call__ = ...``.
    """
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    from zimage_triton.kernels.rope_3d import triton_rope_3d

    def _call(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        freqs_cis=None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if freqs_cis is not None:
            query = triton_rope_3d(query, freqs_cis)
            key = triton_rope_3d(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)

        return output

    return types.MethodType(_call, processor)


def _patch_attention_fused(model: nn.Module, stats: dict) -> None:
    """Patch every ZSingleStreamAttnProcessor with fused QKNorm+RoPE.

    ZSingleStreamAttnProcessor is a plain Python object (not nn.Module),
    so it doesn't appear in ``model.modules()``.  Instead we find Attention
    modules and access their ``.processor`` attribute.

    Args:
        model: Root module.
        stats: Mutable replacement-count dict.
    """
    for module in model.modules():
        processor = getattr(module, "processor", None)
        if processor is not None:
            if type(processor).__name__ == "ZSingleStreamAttnProcessor":
                module.processor.__call__ = _make_fused_attn_call(processor)
                stats["fused_qknorm_rope"] += 1


def _patch_attention_rope(model: nn.Module, stats: dict) -> None:
    """Patch every ZSingleStreamAttnProcessor with plain triton_rope_3d.

    See :func:`_patch_attention_fused` for why we access ``.processor``.

    Args:
        model: Root module.
        stats: Mutable replacement-count dict.
    """
    for module in model.modules():
        processor = getattr(module, "processor", None)
        if processor is not None:
            if type(processor).__name__ == "ZSingleStreamAttnProcessor":
                module.processor.__call__ = _make_plain_rope_attn_call(processor)
                stats["rope_3d"] += 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_triton_kernels(
    model: nn.Module,
    enable_fused_norm: bool = True,
    enable_fused_qknorm_rope: bool = True,
    enable_adaln: bool = True,
) -> dict[str, int]:
    """Apply Triton kernel replacements to a ZImageTransformer2DModel.

    Walks the model graph and performs the following replacements in order:

    1. Replace all ``RMSNorm`` modules with ``TritonRMSNorm`` (weight shared).
    2. Patch ``FeedForward.forward`` to use ``triton_swiglu_forward``.
    3. Patch ``ZSingleStreamAttnProcessor.__call__`` to use either the fused
       QKNorm+RoPE kernel (``triton_fused_qknorm_rope``, default) or the plain
       ``triton_rope_3d`` fallback.

    All replacements share the original weight parameters so no additional GPU
    memory is allocated.

    Args:
        model: A ``ZImageTransformer2DModel`` (or any compatible module tree).
        enable_fused_norm: If ``True``, replace RMSNorm modules and patch FFN
            forward (default ``True``).
        enable_fused_qknorm_rope: If ``True``, use the fused QKNorm+RoPE kernel
            in the attention processor; otherwise use plain ``triton_rope_3d``
            (default ``True``).
        enable_adaln: Reserved for future AdaLN fusion (currently unused).

    Returns:
        A dict mapping replacement type to count, e.g.::

            {"rms_norm": 136, "swiglu": 34, "fused_qknorm_rope": 34, ...}
    """
    stats: dict[str, int] = {
        "rms_norm": 0,
        "swiglu": 0,
        "rope_3d": 0,
        "adaln": 0,
        "fused_norm_res": 0,
        "fused_qknorm_rope": 0,
    }

    if enable_fused_norm:
        _replace_all_rms_norm(model, stats)
        _patch_all_ffn(model, stats)

    if enable_fused_qknorm_rope:
        _patch_attention_fused(model, stats)
    else:
        _patch_attention_rope(model, stats)

    logger.info("Triton patching complete: %s", stats)
    return stats


def find_patchable_model(model: object) -> nn.Module:
    """Find the underlying nn.Module from a model wrapper.

    ZImagePipeline wraps a transformer model internally.
    This searches for the nn.Module that contains patchable layers.

    Args:
        model: A model object (may or may not be an nn.Module).

    Returns:
        The underlying nn.Module suitable for Triton patching.

    Raises:
        RuntimeError: If no nn.Module can be found inside the wrapper.
    """
    if isinstance(model, nn.Module):
        return model

    candidates = [
        "transformer",
        "model",
        "unet",
        "_model",
    ]
    for attr in candidates:
        inner = getattr(model, attr, None)
        if isinstance(inner, nn.Module):
            logger.info("Found patchable model at .%s", attr)
            return inner

    for attr in dir(model):
        if attr.startswith("_"):
            continue
        val = getattr(model, attr, None)
        if isinstance(val, nn.Module):
            logger.info("Found patchable model at .%s", attr)
            return val

    raise RuntimeError("Cannot find nn.Module inside the model wrapper.")
