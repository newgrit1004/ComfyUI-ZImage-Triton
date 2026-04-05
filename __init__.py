"""ComfyUI-ZImage-Triton: Triton kernel acceleration for ZImage S3-DiT."""

import os
import sys

# Add src/ to sys.path so zimage_triton package can be imported
_src = os.path.join(os.path.dirname(__file__), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

try:
    from .nodes import ZImageTritonApply
except ImportError:
    ZImageTritonApply = None  # type: ignore[assignment, misc]

if ZImageTritonApply is not None:
    NODE_CLASS_MAPPINGS = {
        "ZImageTritonApply": ZImageTritonApply,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "ZImageTritonApply": "ZImage Triton Accelerate",
    }
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
