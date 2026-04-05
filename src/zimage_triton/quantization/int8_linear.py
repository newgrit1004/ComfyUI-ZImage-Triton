"""INT8 quantization for Linear layers using torch._int_mm.

Supports three modes:

- **W8A8** (``mode="w8a8"``): Both weights and activations quantized to INT8.
  Fastest (INT8 tensor cores), but activation quantization can degrade quality.
- **W8A8+Hadamard** (``mode="w8a8_hadamard"``): W8A8 with group-wise Hadamard
  rotation before activation quantization. Spreads outliers across channels for
  better quantization quality with minimal runtime overhead.
- **W8A16** (``mode="w8a16"``): Only weights quantized to INT8, activations
  stay in original dtype. Near-lossless quality with VRAM savings.

Sensitive layers (embedders, AdaLN, final_layer, first/last transformer blocks)
are automatically skipped to preserve quality.
"""

import logging
import re
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_MIN_DIM = 1024
_HADAMARD_GROUP_SIZE = 128  # matches Z-Image head_dim; 3840/128=30, 10240/128=80

_SENSITIVE_PATTERNS: list[str] = [
    "cap_embedder",
    "t_embedder",
    "x_embedder",
    "adaLN_modulation",
    "context_refiner",
    "noise_refiner",
    "final_layer",
]

_FIRST_BLOCK_RE = re.compile(r"layers\.0\.")
_LAST_BLOCKS_RE: re.Pattern[str] | None = None


def _is_sensitive_layer(name: str, n_blocks: int) -> bool:
    for pattern in _SENSITIVE_PATTERNS:
        if pattern in name:
            return True
    if _FIRST_BLOCK_RE.search(name):
        return True
    global _LAST_BLOCKS_RE  # noqa: PLW0603
    if _LAST_BLOCKS_RE is None and n_blocks > 2:
        last_indices = "|".join(str(i) for i in range(n_blocks - 2, n_blocks))
        _LAST_BLOCKS_RE = re.compile(rf"layers\.({last_indices})\.")
    if _LAST_BLOCKS_RE is not None and _LAST_BLOCKS_RE.search(name):
        return True
    return False


def _count_blocks(model: nn.Module) -> int:
    n = 0
    for module in model.modules():
        if type(module).__name__ == "JointTransformerBlock":
            n += 1
    return n


def _quantize_weight_per_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight to INT8 per output channel (row)."""
    scale = weight.abs().amax(dim=1).clamp(min=1e-5) / 127.0
    weight_int8 = (weight / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return weight_int8, scale


# --- Forward builders ---


def _make_w8a8_forward(module):
    """W8A8: naive per-token activation quantization."""
    has_bias = module.bias is not None
    bias = module.bias

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        orig_dtype = x.dtype
        x_2d = x.reshape(-1, orig_shape[-1])

        x_scale = x_2d.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
        x_int8 = (x_2d / x_scale).round().clamp(-128, 127).to(torch.int8)

        y_int32 = torch._int_mm(x_int8, self._w_int8.t())
        y = y_int32.to(orig_dtype) * (x_scale * self._w_scale.unsqueeze(0))

        if has_bias:
            y = y + bias
        return y.view(*orig_shape[:-1], y.shape[-1])

    return types.MethodType(_forward, module)


def _make_w8a8_hadamard_forward(module, H):
    """W8A8 + Hadamard: rotate activation before quantization.

    Group-wise Hadamard rotation spreads outliers across channels,
    making per-token amax a better scale estimate.
    Weight is pre-rotated offline (W_rot = W @ H_block^T).
    """
    has_bias = module.bias is not None
    bias = module.bias
    group_size = H.shape[0]
    _H_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        orig_dtype = x.dtype
        x_2d = x.reshape(-1, orig_shape[-1])

        # Online group-wise Hadamard rotation (cached after first call)
        cache_key = (x_2d.device, x_2d.dtype)
        if cache_key not in _H_cache:
            _H_cache[cache_key] = H.to(dtype=x_2d.dtype, device=x_2d.device)
        H_dev = _H_cache[cache_key]
        n_groups = x_2d.shape[-1] // group_size
        x_grouped = x_2d.view(x_2d.shape[0], n_groups, group_size)
        x_rot = torch.matmul(x_grouped, H_dev).reshape_as(x_2d)

        # Per-token quantization on rotated (outlier-spread) activation
        x_scale = x_rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
        x_int8 = (x_rot / x_scale).round().clamp(-128, 127).to(torch.int8)

        # INT8 GEMM with pre-rotated weight
        y_int32 = torch._int_mm(x_int8, self._w_int8.t())
        y = y_int32.to(orig_dtype) * (x_scale * self._w_scale.unsqueeze(0))

        if has_bias:
            y = y + bias
        return y.view(*orig_shape[:-1], y.shape[-1])

    return types.MethodType(_forward, module)


def _make_w8a16_forward(module):
    """W8A16: weight-only INT8, activation in original dtype."""
    bias = module.bias

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        w_deq = self._w_int8.to(x.dtype) * self._w_scale.unsqueeze(1).to(x.dtype)
        return F.linear(x, w_deq, bias)

    return types.MethodType(_forward, module)


def _int8_state_dict_hook(
    module: nn.Module,
    state_dict: dict,
    prefix: str,
    local_metadata: dict,
) -> None:
    """Exclude CPU weight from state_dict so ComfyUI's module_size() is accurate.

    Without this, module_size() counts both the CPU BF16 weight and the GPU
    INT8 buffers, causing VRAM overreporting. When ControlNet is also loaded,
    ComfyUI partially offloads it due to perceived VRAM pressure.
    """
    weight_key = prefix + "weight"
    if weight_key in state_dict:
        del state_dict[weight_key]


def _int8_apply(self, fn, recurse=True):
    """Custom _apply for INT8 modules: skip weight, pin INT8 buffers to GPU.

    - ``weight`` stays on CPU (forward uses ``_w_int8`` instead).
    - ``_w_int8``/``_w_scale`` are only moved TO GPU, never back to CPU.
      This prevents ComfyUI's partial offload from moving them off GPU,
      which would cause torch._int_mm to run on CPU (extremely slow).
      The buffers are explicitly cleaned up by ``restore_int8_quantization``.

    Follows the pattern in comfy/ops.py:934-946 for quantized layers.
    """
    if recurse:
        for module in self.children():
            module._apply(fn)
    for key, param in self._parameters.items():
        if param is None:
            continue
        if key == "weight":
            continue  # stays on CPU; forward uses _w_int8 buffer instead
        self.register_parameter(
            key, torch.nn.Parameter(fn(param), requires_grad=False)
        )
    for key, buf in self._buffers.items():
        if buf is not None:
            # INT8 buffers: allow CPU→GPU but block GPU→CPU.
            # Partial offload would make forward() use CPU _int_mm.
            if key in ("_w_int8", "_w_scale") and buf.device.type == "cuda":
                continue
            self._buffers[key] = fn(buf)
    return self


# --- Main entry point ---


_VALID_MODES = ("w8a8", "w8a8_hadamard", "w8a16")


def apply_int8_quantization(
    model: nn.Module,
    min_dim: int = _MIN_DIM,
    mode: str = "w8a8",
) -> dict[str, int]:
    """Apply INT8 quantization to qualifying Linear layers.

    Args:
        model: The transformer model.
        min_dim: Minimum dimension for both in/out features.
        mode: ``"w8a8"`` | ``"w8a8_hadamard"`` | ``"w8a16"``.

    Returns:
        Dict with counts: {"quantized", "skipped", "sensitive", "had_skipped"}
    """
    if not hasattr(torch, "_int_mm"):
        logger.warning("torch._int_mm not available, skipping INT8")
        return {"quantized": 0, "skipped": 0, "sensitive": 0, "had_skipped": 0}

    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown mode: {mode!r}. Use one of {_VALID_MODES}")

    use_hadamard = mode == "w8a8_hadamard"
    H: torch.Tensor | None = None
    if use_hadamard:
        from zimage_triton.quantization.hadamard import build_hadamard

        H = build_hadamard(_HADAMARD_GROUP_SIZE)

    n_blocks = _count_blocks(model)
    stats = {"quantized": 0, "skipped": 0, "sensitive": 0, "had_skipped": 0}

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        in_f = module.in_features
        out_f = module.out_features

        if in_f < min_dim or out_f < min_dim:
            stats["skipped"] += 1
            continue

        if _is_sensitive_layer(name, n_blocks):
            stats["sensitive"] += 1
            continue

        weight_fp32 = module.weight.data.float()

        # For Hadamard mode: rotate weight offline (W_rot = W @ H_block^T)
        if use_hadamard and H is not None and in_f % _HADAMARD_GROUP_SIZE == 0:
            from zimage_triton.quantization.hadamard import rotate_weight

            weight_fp32 = rotate_weight(weight_fp32, H, _HADAMARD_GROUP_SIZE)
        elif use_hadamard and in_f % _HADAMARD_GROUP_SIZE != 0:
            # Can't apply Hadamard if dim not divisible — fall back to naive
            stats["had_skipped"] += 1
            logger.debug(
                "Hadamard skip %s: in=%d not divisible by %d",
                name,
                in_f,
                _HADAMARD_GROUP_SIZE,
            )

        weight_int8, weight_scale = _quantize_weight_per_channel(weight_fp32)
        weight_int8 = weight_int8.to(module.weight.device)
        weight_scale = weight_scale.to(
            device=module.weight.device,
            dtype=module.weight.dtype,
        )

        module.register_buffer("_w_int8", weight_int8)
        module.register_buffer("_w_scale", weight_scale)
        module.weight.data = module.weight.data.to(device="cpu")

        # Choose forward
        if mode == "w8a16":
            fwd = _make_w8a16_forward(module)
        elif use_hadamard and H is not None and in_f % _HADAMARD_GROUP_SIZE == 0:
            fwd = _make_w8a8_hadamard_forward(module, H)
        else:
            fwd = _make_w8a8_forward(module)

        module.forward = fwd
        module._int8_quantized = True
        module._int8_use_hadamard = (
            use_hadamard and H is not None and in_f % _HADAMARD_GROUP_SIZE == 0
        )
        module._int8_layer_name = name

        # Keep weight on CPU to avoid VRAM waste; only _w_int8/_w_scale
        # buffers live on GPU. LoRA patches applied by ComfyUI to the CPU
        # weight are intentionally ignored — forward reads _w_int8 directly.
        module._apply = types.MethodType(_int8_apply, module)

        # Hide CPU weight from state_dict so ComfyUI's module_size()
        # reports actual GPU footprint (prevents ControlNet partial offload).
        module._int8_sd_hook = module._register_state_dict_hook(
            _int8_state_dict_hook,
        )

        stats["quantized"] += 1
        logger.debug("INT8 [%s] %s (%d->%d)", mode, name, in_f, out_f)

    logger.info(
        "INT8 [%s]: %d quantized, %d skipped, %d sensitive, %d had_skip",
        mode,
        stats["quantized"],
        stats["skipped"],
        stats["sensitive"],
        stats["had_skipped"],
    )
    return stats


def restore_int8_quantization(model: nn.Module) -> int:
    """Undo INT8 quantization, restoring original BF16 weights.

    Reverses all changes made by apply_int8_quantization:
    - Restores weight from CPU to GPU
    - Removes _w_int8/_w_scale buffers
    - Removes _apply override and restores nn.Linear.forward
    - Cleans up metadata flags

    Returns:
        Number of modules restored.
    """
    restored = 0

    for name, module in list(model.named_modules()):
        if not getattr(module, "_int8_quantized", False):
            continue
        if not isinstance(module, nn.Linear):
            continue

        # Restore weight from CPU to GPU
        target_device = (
            module._w_int8.device
            if hasattr(module, "_w_int8")
            else torch.device("cpu")
        )
        module.weight = nn.Parameter(
            module.weight.data.to(device=target_device),
            requires_grad=False,
        )

        # Remove INT8 buffers
        for buf_name in ("_w_int8", "_w_scale"):
            if buf_name in module._buffers:
                del module._buffers[buf_name]

        # Remove state_dict hook
        hook = getattr(module, "_int8_sd_hook", None)
        if hook is not None:
            hook.remove()

        # Remove instance-level overrides (falls back to class methods)
        for attr in (
            "forward",
            "_apply",
            "_int8_quantized",
            "_int8_use_hadamard",
            "_int8_layer_name",
            "_int8_sd_hook",
        ):
            module.__dict__.pop(attr, None)

        restored += 1

    # Clean up model-level state
    for attr in ("_int8_quantized", "_int8_stats"):
        if hasattr(model, attr):
            delattr(model, attr)

    logger.info("INT8 restore: %d modules restored.", restored)
    return restored
