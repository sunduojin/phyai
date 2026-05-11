"""Exception types raised by phyai.vgpu.

Each error class is identifiable by callers without parsing message strings;
the message text carries diagnostic detail.
"""

from __future__ import annotations


class VGPUError(Exception):
    """Base for all phyai.vgpu errors."""


class VGPUNotApplicableError(VGPUError):
    """vGPU cannot be used in the current setup.

    Raised by ``init`` when ``world_size > 1``, the device is not CUDA, the
    compute capability is below 8.0, or the named backend is not registered.
    """


class VGPURuntimeError(VGPUError):
    """Capacity or configuration error at split / allocation time.

    Examples: requested SM total exceeds the device's SM count, an
    unsupported compute capability, or non-positive SM counts.
    """


class VGPUDriverError(VGPUError):
    """A CUDA driver call surfaced an error that vgpu chose not to swallow."""


class BackendCapabilityError(VGPUError):
    """The active backend does not support the requested operation.

    Typical case: the torch backend asked to ``split_by_count`` /
    ``split_by_sm_counts``. Switch to ``backend='flashinfer'`` or use a
    single-shard API like ``vGPU(sm_count=...)``.
    """
