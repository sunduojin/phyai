"""Concrete LinearKernel implementations.

Each ``_<vendor>.py`` module owns one kernel; the parent
:mod:`phyai.layers.linear` package decides registration order in its
:func:`init` function.
"""

from __future__ import annotations

from phyai.layers.linear.backends._flashinfer import FlashInferKernel
from phyai.layers.linear.backends._torch import TorchKernel

__all__ = ["TorchKernel", "FlashInferKernel"]
