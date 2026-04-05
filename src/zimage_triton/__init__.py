"""ZImage Triton kernel optimization for S3-DiT image generation.

Provides fused Triton kernels targeting RTX 5090 (SM120) for the
ZImage 6.15B parameter diffusion transformer.
"""

__version__ = "0.1.0"


def _check_torch() -> None:
    """Verify that PyTorch with CUDA is available."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "zimage-triton requires PyTorch with CUDA support. "
            "Install it with: pip install torch --index-url "
            "https://download.pytorch.org/whl/cu128"
        ) from exc


_check_torch()

# Kernels
from zimage_triton.kernels import (  # noqa: E402
    TritonAdaLNModulation,
    TritonRMSNorm,
    TritonSwiGLU,
    triton_adaln_modulation,
    triton_fused_norm_gate_residual,
    triton_fused_qknorm_rope,
    triton_rms_norm,
    triton_rope_3d,
    triton_swiglu_forward,
)

# Models
from zimage_triton.models import (  # noqa: E402
    BaseRunner,
    TritonRunner,
    apply_triton_kernels,
)

__all__ = [
    # Kernels
    "triton_rms_norm",
    "TritonRMSNorm",
    "triton_swiglu_forward",
    "TritonSwiGLU",
    "triton_rope_3d",
    "triton_adaln_modulation",
    "TritonAdaLNModulation",
    "triton_fused_norm_gate_residual",
    "triton_fused_qknorm_rope",
    # Models
    "BaseRunner",
    "TritonRunner",
    "apply_triton_kernels",
]
