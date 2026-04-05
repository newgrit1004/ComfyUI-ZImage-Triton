"""ZImage inference runners with Triton kernel patching."""

from zimage_triton.models.base_runner import BaseRunner
from zimage_triton.models.patching import apply_triton_kernels
from zimage_triton.models.triton_runner import TritonRunner

_RUNNER_MAP: dict[str, type] = {
    "base": BaseRunner,
    "triton": TritonRunner,
}

ALL_RUNNER_NAMES = list(_RUNNER_MAP.keys())


def get_runner_class(name: str) -> type:
    """Get runner class by name."""
    if name not in _RUNNER_MAP:
        raise ValueError(f"Unknown runner '{name}'. Available: {ALL_RUNNER_NAMES}")
    return _RUNNER_MAP[name]


def create_runner(name: str = "base", **kwargs):
    """Create a runner instance by name."""
    cls = get_runner_class(name)
    return cls(**kwargs)


__all__ = [
    "BaseRunner",
    "TritonRunner",
    "apply_triton_kernels",
    "get_runner_class",
    "create_runner",
    "ALL_RUNNER_NAMES",
]
