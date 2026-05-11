"""Dispatcher with hashmap cache.

The hot path is a tuple-keyed dict lookup; only on cache miss do we walk
the registry to score candidates. ``CallCtx`` is intentionally not built
on the hot path — its construction would dominate the lookup cost.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from phyai.parallel.backend import Backend, Op
from phyai.parallel.exceptions import NoBackendError
from phyai.parallel.mesh import Mesh
from phyai.parallel.registry import DefaultPolicy, ForcedPolicy, Policy, Registry
from phyai.parallel.state import current_mode


def _size_bucket(nbytes: int) -> int:
    """Logarithmic bucket so similar-sized tensors share a cache entry."""
    return nbytes.bit_length()


class Dispatcher:
    """Process-level dispatcher.

    The hot path is a tuple-keyed dict lookup; only on miss do we ask the
    Registry / Policy to pick a backend.
    """

    def __init__(
        self,
        registry: Registry,
        policy: Policy | None = None,
    ) -> None:
        self.registry = registry
        forced = os.environ.get("PHYAI_FORCE_BACKEND")
        if policy is not None:
            self.policy = policy
        elif forced:
            self.policy = ForcedPolicy(forced)
        else:
            self.policy = DefaultPolicy()
        self._cache: dict[tuple[Any, ...], Backend] = {}

    def select(
        self,
        *,
        op: Op,
        mesh: Mesh,
        axis: str,
        tensor: torch.Tensor | None,
        extra_key: tuple[Any, ...] = (),
        **extra_kwargs: Any,
    ) -> Backend:
        nbytes = tensor.numel() * tensor.element_size() if tensor is not None else 0
        dtype = tensor.dtype if tensor is not None else torch.uint8
        mode = current_mode()
        ws = mesh.axis_size(axis)
        key = (op, axis, mesh.name, dtype, _size_bucket(nbytes), mode, ws, extra_key)
        b = self._cache.get(key)
        if b is None:
            pg = mesh.axis_group(axis)
            cands = self.registry.candidates(
                op=op,
                mode=mode,
                nbytes=nbytes,
                dtype=dtype,
                world_size=ws,
                topology=mesh.topology(),
                pg=pg,
                **extra_kwargs,
            )
            if not cands:
                raise NoBackendError(
                    f"no backend for op={op.value} mode={mode.value} "
                    f"axis={axis} mesh={mesh.name} ws={ws} "
                    f"nbytes={nbytes} dtype={dtype}"
                )
            b = self.policy.select(cands)
            self._cache[key] = b
        return b

    def has_for(
        self,
        *,
        op: Op,
        mesh: Mesh,
        axis: str,
        tensor: torch.Tensor | None = None,
    ) -> bool:
        try:
            self.select(op=op, mesh=mesh, axis=axis, tensor=tensor)
            return True
        except NoBackendError:
            return False

    def clear_cache(self) -> None:
        """Test helper / used after re-registering backends."""
        self._cache.clear()


# Process-level singleton; populated in `phyai.parallel.init(...)`.
_dispatcher: Dispatcher | None = None


def get_dispatcher() -> Dispatcher:
    if _dispatcher is None:
        raise RuntimeError("phyai.parallel.init(...) not called yet.")
    return _dispatcher


def set_dispatcher(d: Dispatcher) -> None:
    global _dispatcher
    _dispatcher = d
