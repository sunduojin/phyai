"""ShardSpec — minimal value object passed between backend and phyai layer.

Backends populate a ``ShardSpec`` per green-ctx slice they create. The
phyai partition layer wraps it into a user-facing :class:`Shard`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class ShardSpec:
    """Backend-internal representation of one green-ctx shard.

    Attributes:
        stream: A non-blocking ``torch.cuda.Stream`` bound to the green ctx.
        sm_count: Actual SM count after round-up.
        is_remainder: True for the trailing remainder shard returned by
            ``split_by_count`` / ``split_by_sm_counts``.
        _backend_handle: Backend-specific opaque handle (flashinfer:
            ``CUdevResource``; torch: ``GreenContext``). Backends use this
            in ``destroy``.
        _backend_name: Name of the backend that produced this spec.
    """

    stream: torch.cuda.Stream
    sm_count: int
    is_remainder: bool
    _backend_handle: Any
    _backend_name: str
