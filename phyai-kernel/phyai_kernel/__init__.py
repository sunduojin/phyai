"""phyai-kernel — JIT-compiled CPU/CUDA kernels for phyai via tvm-ffi."""

__version__ = "0.1.0"

from . import jit_utils
from .jit_utils import jit

__all__ = [
    "__version__",
    "jit",
    "jit_utils",
]
