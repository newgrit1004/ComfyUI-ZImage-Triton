"""Custom ModelPatcher for INT8-quantized Z-Image models.

Prevents double-GPU memory usage when LoRA is applied to INT8-quantized
modules. Without this, ``patch_weight_to_device()`` copies the BF16 weight
to GPU for LoRA computation, even though INT8 forward only reads
``_w_int8`` buffers. The extra ~12 GB BF16 weight on GPU causes VRAM
pressure and triggers partial offload when ControlNet is also loaded.
"""

from __future__ import annotations

import logging

import comfy.utils
from comfy.model_patcher import ModelPatcher

logger = logging.getLogger(__name__)


class ZImageTritonModelPatcher(ModelPatcher):
    """ModelPatcher that keeps INT8 module weights on CPU.

    Overrides two methods:
    - ``patch_weight_to_device``: passes ``device_to=None`` for INT8 keys
      so LoRA patches are applied on CPU and the weight never moves to GPU.
    - ``partially_unload``: skips INT8 modules whose GPU buffers cannot be
      offloaded (pinned by ``_int8_apply``), preventing memory accounting
      errors.
    """

    def _is_int8_key(self, key: str) -> bool:
        """Return True if *key* belongs to an INT8-quantized module."""
        parts = key.rsplit(".", 1)
        if len(parts) < 2:
            return False
        try:
            module = comfy.utils.get_attr(self.model, parts[0])
            return getattr(module, "_int8_quantized", False)
        except (AttributeError, KeyError):
            return False

    # ------------------------------------------------------------------
    # patch_weight_to_device override
    # ------------------------------------------------------------------

    def patch_weight_to_device(
        self,
        key: str,
        device_to=None,
        inplace_update: bool = False,
        return_weight: bool = False,
    ):
        """Keep INT8 weights on CPU to prevent double-GPU memory.

        INT8 forward reads ``_w_int8`` (GPU buffer), not ``weight``.
        Copying ``weight`` to GPU wastes ~12 GB and triggers partial
        offload when ControlNet coexists.  Passing ``device_to=None``
        makes the base class apply LoRA on CPU and store the result on CPU.
        """
        if self._is_int8_key(key):
            return super().patch_weight_to_device(
                key,
                device_to=None,
                inplace_update=inplace_update,
                return_weight=return_weight,
            )
        return super().patch_weight_to_device(
            key,
            device_to=device_to,
            inplace_update=inplace_update,
            return_weight=return_weight,
        )

    # ------------------------------------------------------------------
    # partially_unload override
    # ------------------------------------------------------------------

    def partially_unload(
        self,
        device_to,
        memory_to_free: int = 0,
        force_patch_weights: bool = False,
    ) -> int:
        """Skip INT8 modules during partial offload.

        ``_int8_apply`` pins ``_w_int8`` / ``_w_scale`` on GPU, so
        ``m.to(cpu)`` cannot actually free them.  If the base class
        counts them as freed, ``model_loaded_weight_memory`` underflows
        and subsequent VRAM budgeting breaks.

        Temporarily setting ``comfy_patched_weights = False`` on INT8
        modules makes the base loop skip them entirely.
        """
        int8_saved: list[tuple] = []
        for _name, module in self.model.named_modules():
            if not getattr(module, "_int8_quantized", False):
                continue
            if getattr(module, "comfy_patched_weights", False):
                module.comfy_patched_weights = False
                int8_saved.append((module, True))

        try:
            result = super().partially_unload(
                device_to, memory_to_free, force_patch_weights,
            )
        finally:
            for module, val in int8_saved:
                module.comfy_patched_weights = val

        return result
