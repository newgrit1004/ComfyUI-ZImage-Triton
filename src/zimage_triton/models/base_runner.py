"""Base ZImage inference runner (vanilla PyTorch)."""

import logging

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class BaseRunner:
    """Vanilla PyTorch ZImage inference.

    Loads the ZImage pipeline from HuggingFace and exposes a simple
    ``generate`` method.  No Triton optimisations are applied here; this
    class is the reference baseline used for correctness and latency
    comparisons.

    Model loading is deferred: the pipeline is not loaded in ``__init__``
    but on the first call to ``generate()`` (or by calling ``load_model()``
    explicitly).  Call ``unload_model()`` to free GPU VRAM.

    Args:
        model_id: HuggingFace model identifier.
        device: Target device string (e.g. ``"cuda"``).
        dtype: Torch dtype for model weights (default ``torch.bfloat16``).
    """

    def __init__(
        self,
        model_id: str = "Tongyi-MAI/Z-Image-Turbo",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.pipe = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Return True if the pipeline is currently loaded."""
        return self.pipe is not None

    def load_model(self) -> None:
        """Load the ZImage pipeline into memory.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self.pipe is not None:
            return
        from diffusers import ZImagePipeline

        self.pipe = ZImagePipeline.from_pretrained(
            self.model_id, torch_dtype=self.dtype
        ).to(self.device)
        logger.info("Loaded %s on %s (%s)", self.model_id, self.device, self.dtype)

    def unload_model(self) -> None:
        """Delete the pipeline and free GPU VRAM.

        Safe to call when the model is not loaded.
        """
        if self.pipe is None:
            return
        del self.pipe
        self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded %s", self.model_id)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 8,
        guidance_scale: float = 3.5,
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
    ) -> Image.Image:
        """Run text-to-image generation.

        Loads the model automatically on first call if not already loaded.

        Args:
            prompt: Text prompt describing the desired image.
            num_inference_steps: Number of diffusion denoising steps.
            guidance_scale: Classifier-free guidance scale.
            width: Output image width in pixels.
            height: Output image height in pixels.
            seed: Optional random seed for reproducibility.

        Returns:
            Generated PIL image.
        """
        if not self.is_loaded:
            self.load_model()

        generator = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
        return result.images[0]

    def get_runner_name(self) -> str:
        """Return a human-readable identifier for this runner."""
        return "Base"
