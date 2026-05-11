"""Registry of backends + selection Policy.

Each backend is registered once with optional ``prefer_for`` hints
(per-op override). The default policy uses ``prefer_for`` first, then
registration order, then a capture-safety filter. There are no numeric
scores — backends are ordered by registration intent rather than tuned
priority numbers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from phyai.parallel.backend import Backend, Op, Topology
from phyai.parallel.exceptions import NoBackendError
from phyai.parallel.state import Mode


class Registry:
    """Process-level registry. Built once at init time."""

    def __init__(self) -> None:
        self._backends: list[Backend] = []
        self._prefer: dict[Op, list[str]] = {}

    def register(
        self,
        backend: Backend,
        *,
        prefer_for: set[Op] | None = None,
    ) -> None:
        self._backends.append(backend)
        if prefer_for:
            for op in prefer_for:
                self._prefer.setdefault(op, []).append(backend.name)

    def candidates(
        self,
        *,
        op: Op,
        mode: Mode,
        nbytes: int,
        dtype: torch.dtype,
        world_size: int,
        topology: Topology,
        **extra: object,
    ) -> list[Backend]:
        out: list[Backend] = []
        prefer_names = self._prefer.get(op, [])

        # 1. Preferred backends in declaration order
        for name in prefer_names:
            for b in self._backends:
                if b.name != name:
                    continue
                if mode == Mode.GRAPH_CAPTURING and not b.supports_capture():
                    continue
                if b.can_handle(
                    op=op,
                    mode=mode,
                    nbytes=nbytes,
                    dtype=dtype,
                    world_size=world_size,
                    topology=topology,
                    **extra,
                ):
                    out.append(b)
                break

        # 2. Remaining backends in registration order
        for b in self._backends:
            if b.name in prefer_names:
                continue
            if mode == Mode.GRAPH_CAPTURING and not b.supports_capture():
                continue
            if b.can_handle(
                op=op,
                mode=mode,
                nbytes=nbytes,
                dtype=dtype,
                world_size=world_size,
                topology=topology,
                **extra,
            ):
                out.append(b)
        return out

    def has(
        self,
        *,
        op: Op,
        mode: Mode,
        nbytes: int,
        dtype: torch.dtype,
        world_size: int,
        topology: Topology,
        **extra: object,
    ) -> bool:
        return bool(
            self.candidates(
                op=op,
                mode=mode,
                nbytes=nbytes,
                dtype=dtype,
                world_size=world_size,
                topology=topology,
                **extra,
            )
        )

    def all(self) -> list[Backend]:
        return list(self._backends)

    def validate(self) -> None:
        """Sanity-check: each (op, mode) standard ctx has at least one
        backend, every preferred name actually got registered.

        Probes with ``pg=None`` so backends fall through their permissive
        probe-time path. ``Mode.GRAPH_CAPTURING`` is only checked when at
        least one registered backend supports capture (gloo-only setups
        legitimately have no capture coverage).
        """
        # Check preferred names exist
        names = {b.name for b in self._backends}
        for op, prefs in self._prefer.items():
            for n in prefs:
                if n not in names:
                    raise NoBackendError(
                        f"prefer_for[{op.value}]={n!r} but no backend with "
                        f"that name is registered (have: {sorted(names)})"
                    )

        has_capture_backend = any(b.supports_capture() for b in self._backends)
        modes: list[Mode] = [Mode.EAGER]
        if has_capture_backend:
            modes.append(Mode.GRAPH_CAPTURING)

        # Check eager + graph fallback for the common ops with a probe ctx
        probe_topology = Topology(True, True, 1, 8)
        for op in (
            Op.ALL_REDUCE,
            Op.ALL_GATHER,
            Op.REDUCE_SCATTER,
            Op.BROADCAST,
            Op.ALL_TO_ALL,
            Op.SEND,
            Op.RECV,
        ):
            for mode in modes:
                if not self.has(
                    op=op,
                    mode=mode,
                    nbytes=1024,
                    dtype=torch.bfloat16,
                    world_size=2,
                    topology=probe_topology,
                ):
                    raise NoBackendError(
                        f"no backend handles op={op.value} mode={mode.value} "
                        f"(at least one fallback is required)"
                    )


@runtime_checkable
class Policy(Protocol):
    def select(self, candidates: list[Backend]) -> Backend: ...


class DefaultPolicy:
    """Pick the first candidate. Registry has already ordered them by
    (preferred-for-this-op, registration-order)."""

    def select(self, candidates: list[Backend]) -> Backend:
        if not candidates:
            raise NoBackendError("no candidate backend")
        return candidates[0]


class ForcedPolicy:
    """Honor PHYAI_FORCE_BACKEND=<name> if set; otherwise fall back."""

    def __init__(self, name: str, fallback: Policy = DefaultPolicy()) -> None:
        self.name = name
        self.fallback = fallback

    def select(self, candidates: list[Backend]) -> Backend:
        for b in candidates:
            if b.name == self.name:
                return b
        return self.fallback.select(candidates)
