"""Mock-based tests for Registry, Policy, and ContextVar mode."""

from __future__ import annotations

import pytest
import torch

from phyai.parallel.backend import Op, Topology
from phyai.parallel.exceptions import NoBackendError
from phyai.parallel.registry import (
    DefaultPolicy,
    ForcedPolicy,
    Registry,
)
from phyai.parallel.state import Mode


def _topo() -> Topology:
    return Topology(
        is_full_nvlink=True,
        is_single_node=True,
        n_nodes=1,
        n_gpus_per_node=8,
    )


class FakeBackend:
    """Configurable mock backend."""

    def __init__(
        self,
        name: str,
        *,
        ops: set[Op],
        modes: set[Mode],
        capture: bool = True,
        max_bytes: int | None = None,
    ):
        self.name = name
        self._ops = ops
        self._modes = modes
        self._capture = capture
        self._max_bytes = max_bytes
        self.executed = 0
        self.last_kwargs: dict | None = None

    def can_handle(self, *, op, mode, nbytes, dtype, world_size, topology, **extra):
        if op not in self._ops:
            return False
        if mode not in self._modes:
            return False
        if self._max_bytes is not None and nbytes > self._max_bytes:
            return False
        return True

    def supports_capture(self):
        return self._capture

    def execute(self, *, op, pg, **kwargs):
        self.executed += 1
        self.last_kwargs = kwargs
        return None


# ---------------------------------------------------------------------------
# Registry: ordering, prefer_for, capture filter
# ---------------------------------------------------------------------------


def test_registry_returns_in_registration_order():
    r = Registry()
    a = FakeBackend("a", ops={Op.ALL_REDUCE}, modes={Mode.EAGER})
    b = FakeBackend("b", ops={Op.ALL_REDUCE}, modes={Mode.EAGER})
    r.register(a)
    r.register(b)

    cands = r.candidates(
        op=Op.ALL_REDUCE,
        mode=Mode.EAGER,
        nbytes=1024,
        dtype=torch.bfloat16,
        world_size=2,
        topology=_topo(),
    )
    assert [c.name for c in cands] == ["a", "b"]


def test_registry_prefer_for_takes_precedence():
    r = Registry()
    a = FakeBackend("a", ops={Op.ALL_REDUCE}, modes={Mode.EAGER})
    b = FakeBackend("b", ops={Op.ALL_REDUCE}, modes={Mode.EAGER})
    r.register(a)
    r.register(b, prefer_for={Op.ALL_REDUCE})

    cands = r.candidates(
        op=Op.ALL_REDUCE,
        mode=Mode.EAGER,
        nbytes=1024,
        dtype=torch.bfloat16,
        world_size=2,
        topology=_topo(),
    )
    assert [c.name for c in cands] == ["b", "a"]


def test_registry_filters_capture_unsafe():
    r = Registry()
    a = FakeBackend(
        "a", ops={Op.ALL_REDUCE}, modes={Mode.GRAPH_CAPTURING}, capture=False
    )
    b = FakeBackend(
        "b", ops={Op.ALL_REDUCE}, modes={Mode.GRAPH_CAPTURING}, capture=True
    )
    r.register(a)
    r.register(b)

    cands = r.candidates(
        op=Op.ALL_REDUCE,
        mode=Mode.GRAPH_CAPTURING,
        nbytes=1024,
        dtype=torch.bfloat16,
        world_size=2,
        topology=_topo(),
    )
    assert [c.name for c in cands] == ["b"]


def test_registry_filters_can_handle_false():
    r = Registry()
    a = FakeBackend("a", ops={Op.ALL_REDUCE}, modes={Mode.EAGER}, max_bytes=512)
    b = FakeBackend("b", ops={Op.ALL_REDUCE}, modes={Mode.EAGER})
    r.register(a)
    r.register(b)

    cands = r.candidates(
        op=Op.ALL_REDUCE,
        mode=Mode.EAGER,
        nbytes=2048,
        dtype=torch.bfloat16,
        world_size=2,
        topology=_topo(),
    )
    assert [c.name for c in cands] == ["b"]


def test_registry_validate_raises_when_no_fallback():
    r = Registry()
    r.register(FakeBackend("a", ops={Op.ALL_REDUCE}, modes={Mode.EAGER}))
    with pytest.raises(NoBackendError):
        r.validate()


def test_registry_validate_raises_for_unknown_prefer_name():
    r = Registry()
    r.register(FakeBackend("a", ops=set(Op), modes={Mode.EAGER, Mode.GRAPH_CAPTURING}))
    # prefer name doesn't match any registered backend
    r._prefer[Op.ALL_REDUCE] = ["nonexistent"]
    with pytest.raises(NoBackendError):
        r.validate()


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_default_policy_picks_first():
    a = FakeBackend("a", ops={Op.ALL_REDUCE}, modes={Mode.EAGER})
    b = FakeBackend("b", ops={Op.ALL_REDUCE}, modes={Mode.EAGER})
    assert DefaultPolicy().select([a, b]).name == "a"


def test_default_policy_raises_on_empty():
    with pytest.raises(NoBackendError):
        DefaultPolicy().select([])


def test_forced_policy_finds_named_backend():
    a = FakeBackend("a", ops=set(), modes=set())
    b = FakeBackend("b", ops=set(), modes=set())
    assert ForcedPolicy("b").select([a, b]).name == "b"


def test_forced_policy_falls_back_when_name_missing():
    a = FakeBackend("a", ops=set(), modes=set())
    b = FakeBackend("b", ops=set(), modes=set())
    assert ForcedPolicy("nonexistent").select([a, b]).name == "a"
