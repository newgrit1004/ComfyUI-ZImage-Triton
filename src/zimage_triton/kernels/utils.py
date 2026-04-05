"""Kernel utility functions for Triton kernels on SM120 (RTX 5090)."""

import logging

import torch
import triton

logger = logging.getLogger(__name__)

# ZImage model dimensions
ZIMAGE_DIM = 3840
ZIMAGE_FFN_DIM = 10240
ZIMAGE_HEAD_DIM = 128
ZIMAGE_N_HEADS = 30


def calculate_settings(n: int) -> tuple[int, int]:
    """Calculate BLOCK_SIZE and num_warps for a given dimension size.

    Based on Liger Kernel / Unsloth heuristics with SM120 tuning.

    Args:
        n: The dimension size (e.g. hidden_size).

    Returns:
        Tuple of (BLOCK_SIZE, num_warps).

    Raises:
        RuntimeError: If n exceeds the maximum fused size.
    """
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} "
            f"exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


def detect_sm120() -> bool:
    """Detect if the current GPU is SM120 (RTX 5090 Blackwell consumer).

    Returns:
        True if GPU compute capability is (12, 0).
    """
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] == 12 and cap[1] == 0


def get_device_info() -> dict[str, str | int | float | bool]:
    """Get GPU device information for benchmarking and logging.

    Returns:
        Dictionary with device name, memory, compute capability, etc.
    """
    if not torch.cuda.is_available():
        return {"available": False}

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    cap = torch.cuda.get_device_capability(device)

    return {
        "available": True,
        "name": props.name,
        "compute_capability": f"{cap[0]}.{cap[1]}",
        "is_sm120": detect_sm120(),
        "total_memory_gb": round(props.total_memory / (1024**3), 2),
        "multi_processor_count": props.multi_processor_count,
        "cuda_version": torch.version.cuda or "unknown",
        "triton_version": triton.__version__,
        "torch_version": torch.__version__,
    }
