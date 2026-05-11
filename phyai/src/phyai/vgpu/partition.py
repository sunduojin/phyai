"""User-facing partition entry points.

:func:`split_device` mirrors :func:`flashinfer.green_ctx.split_device_green_ctx`
and :func:`split_device_by_sm_count` mirrors
:func:`flashinfer.green_ctx.split_device_green_ctx_by_sm_count`. Both
return a list of :class:`Shard` instances, with the trailing element being
the remainder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from phyai.utils.cuda import device_capability
from phyai.vgpu._spec import ShardSpec
from phyai.vgpu.backend import get_backend, resolve
from phyai.vgpu.topology import round_up_sm_count, validate_total


@dataclass(frozen=True, slots=True)
class Shard:
    """One green-ctx slice of a CUDA device.

    Attributes:
        name: Debug-friendly identifier such as ``"shard_0"`` or
            ``"shard_2_rem"``.
        device: The CUDA device this shard lives on.
        sm_count: Actual SM count after round-up.
        requested_sm_count: The user's original SM ask. For remainder
            shards this equals ``sm_count``.
        is_remainder: True when this shard captures the SMs not allocated
            to any explicit group.
        stream: Non-blocking CUDA stream bound to the green ctx.
        backend: Name of the backend that produced this shard.
        _spec: Backend-internal handle used for cleanup. Not part of the
            stable user API.
    """

    name: str
    device: torch.device
    sm_count: int
    requested_sm_count: int
    is_remainder: bool
    stream: torch.cuda.Stream
    backend: str
    _spec: ShardSpec


def to_shard(
    spec: ShardSpec,
    *,
    name: str,
    requested: int,
    device: torch.device,
) -> Shard:
    """Wrap a :class:`ShardSpec` into a :class:`Shard`."""
    return Shard(
        name=name,
        device=device,
        sm_count=spec.sm_count,
        requested_sm_count=requested if not spec.is_remainder else spec.sm_count,
        is_remainder=spec.is_remainder,
        stream=spec.stream,
        backend=spec._backend_name,
        _spec=spec,
    )


def _select_backend(backend: str | None):
    return resolve(backend) if backend is not None else get_backend()


def split_device(
    device: "str | torch.device",
    *,
    num_groups: int,
    min_count: int,
    backend: str | None = None,
) -> list[Shard]:
    """Split a device into ``num_groups`` shards plus a remainder.

    Mirrors :func:`flashinfer.green_ctx.split_device_green_ctx`. Returns
    ``num_groups + 1`` :class:`Shard` objects; the last one is the
    remainder.

    Args:
        device: ``"cuda:0"``, a ``torch.device``, or any value the latter
            accepts.
        num_groups: Number of equal-sized groups. Must be positive.
        min_count: Minimum SMs per group; rounded up to the CC's alignment.
        backend: Optional backend override (``"flashinfer"`` / ``"torch"``).
            Defaults to the process-level backend.

    Raises:
        VGPURuntimeError: when the post-round-up total exceeds the device's
            SM count, or when ``min_count`` is non-positive.
        BackendCapabilityError: when the active backend does not implement
            multi-shard splitting (e.g. the torch backend).
    """
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}")
    dev = torch.device(device)
    cc = device_capability(dev)
    rounded = round_up_sm_count(min_count, cc)
    total_sms = torch.cuda.get_device_properties(dev).multi_processor_count
    validate_total([rounded] * num_groups, total_sms)
    b = _select_backend(backend)
    specs = b.split_by_count(dev, num_groups, rounded)
    out: list[Shard] = []
    for i, spec in enumerate(specs):
        suffix = "_rem" if spec.is_remainder else ""
        out.append(
            to_shard(
                spec,
                name=f"shard_{i}{suffix}",
                requested=min_count,
                device=dev,
            ),
        )
    return out


def split_device_by_sm_count(
    device: "str | torch.device",
    *,
    sm_counts: list[int],
    backend: str | None = None,
) -> list[Shard]:
    """Split a device into shards with explicit per-shard SM counts.

    Mirrors :func:`flashinfer.green_ctx.split_device_green_ctx_by_sm_count`.
    Returns ``len(sm_counts) + 1`` :class:`Shard` objects; the last one is
    the remainder.

    Args:
        device: target CUDA device.
        sm_counts: list of requested SM counts, one per non-remainder
            shard. Each count is rounded up per the CC's alignment.
        backend: optional backend override.

    Raises:
        VGPURuntimeError: when the post-round-up total exceeds the device's
            SM count.
        BackendCapabilityError: when the active backend does not implement
            multi-shard splitting.
    """
    if not sm_counts:
        raise ValueError("sm_counts must not be empty")
    dev = torch.device(device)
    cc = device_capability(dev)
    rounded = [round_up_sm_count(c, cc) for c in sm_counts]
    total_sms = torch.cuda.get_device_properties(dev).multi_processor_count
    validate_total(rounded, total_sms)
    b = _select_backend(backend)
    specs = b.split_by_sm_counts(dev, rounded)
    out: list[Shard] = []
    for i, spec in enumerate(specs):
        suffix = "_rem" if spec.is_remainder else ""
        if i < len(sm_counts):
            requested = sm_counts[i]
        else:
            requested = spec.sm_count
        out.append(
            to_shard(
                spec,
                name=f"shard_{i}{suffix}",
                requested=requested,
                device=dev,
            ),
        )
    return out
