"""Triton-optimized ZImage inference runner."""

import logging

from zimage_triton.models.base_runner import BaseRunner

logger = logging.getLogger(__name__)


class TritonRunner(BaseRunner):
    """ZImage inference with Triton kernel optimisations.

    Inherits the full pipeline setup from :class:`BaseRunner` and then
    applies Triton kernel patches to the transformer using
    :func:`~zimage_triton.models.patching.apply_triton_kernels`.

    Patching is deferred until :meth:`load_model` is called (or the first
    ``generate()`` call) because the base class uses deferred loading —
    ``self.pipe`` is ``None`` until that point.

    Args:
        model_id: HuggingFace model identifier.
        enable_fused_norm: Replace RMSNorm modules and patch FFN forward.
        enable_fused_qknorm_rope: Use the fused QKNorm+RoPE attention kernel.
        enable_adaln: Reserved for future AdaLN fusion (currently unused).
        **kwargs: Additional keyword arguments forwarded to :class:`BaseRunner`
            (``device``, ``dtype``).
    """

    def __init__(
        self,
        model_id: str = "Tongyi-MAI/Z-Image-Turbo",
        enable_fused_norm: bool = True,
        enable_fused_qknorm_rope: bool = True,
        enable_adaln: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(model_id, **kwargs)
        self._enable_fused_norm = enable_fused_norm
        self._enable_fused_qknorm_rope = enable_fused_qknorm_rope
        self._enable_adaln = enable_adaln
        self.patch_stats: dict[str, int] = {}

    def load_model(self) -> None:
        """Load the pipeline then apply Triton kernel patches."""
        if self.pipe is not None:
            return
        super().load_model()

        from zimage_triton.models.patching import apply_triton_kernels

        self.patch_stats = apply_triton_kernels(
            self.pipe.transformer,
            enable_fused_norm=self._enable_fused_norm,
            enable_fused_qknorm_rope=self._enable_fused_qknorm_rope,
            enable_adaln=self._enable_adaln,
        )
        logger.info("Applied Triton kernels: %s", self.patch_stats)

    def get_runner_name(self) -> str:
        """Return a human-readable identifier for this runner."""
        return "Triton"
