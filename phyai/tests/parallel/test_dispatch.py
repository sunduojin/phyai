"""Mock-based tests for the Dispatcher + cache."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from phyai.parallel.backend import Op
from phyai.parallel.dispatch import Dispatcher
from phyai.parallel.mesh import Mesh
from phyai.parallel.registry import Registry
from phyai.parallel.state import Mode

from .test_registry import FakeBackend, _topo


def _fake_mesh(name: str = "model", *, ws: int = 2) -> Mesh:
    m = MagicMock()
    m.mesh_dim_names = ("tp",)
    m.size.side_effect = lambda axis: ws
    m.get_local_rank.side_effect = lambda axis: 0
    m.get_group.side_effect = lambda axis: MagicMock()
    return Mesh(m, name=name)


def _make(registry, *, eager_backends, graph_backends):
    for b in eager_backends + graph_backends:
        registry.register(b)


def test_dispatcher_picks_first_candidate():
    a = FakeBackend("a", ops={Op.ALL_REDUCE}, modes={Mode.EAGER, Mode.GRAPH_CAPTURING})
    b = FakeBackend("b", ops={Op.ALL_REDUCE}, modes={Mode.EAGER, Mode.GRAPH_CAPTURING})
    r = Registry()
    r.register(a)
    r.register(b)

    d = Dispatcher(r)
    chosen = d.select(
        op=Op.ALL_REDUCE,
        mesh=_fake_mesh(),
        axis="tp",
        tensor=torch.zeros(8),
    )
    assert chosen.name == "a"


def test_dispatcher_caches_decision():
    a = FakeBackend("a", ops={Op.ALL_REDUCE}, modes={Mode.EAGER, Mode.GRAPH_CAPTURING})
    r = Registry()
    r.register(a)
    d = Dispatcher(r)

    spy = patch.object(r, "candidates", wraps=r.candidates).start()
    try:
        x = torch.zeros(64, dtype=torch.bfloat16)
        for _ in range(5):
            d.select(op=Op.ALL_REDUCE, mesh=_fake_mesh(), axis="tp", tensor=x)
        # Same key, registry should be queried at most once.
        assert spy.call_count == 1
    finally:
        patch.stopall()


def test_dispatcher_cache_keyed_by_dtype_and_mode():
    a = FakeBackend("a", ops={Op.ALL_REDUCE}, modes={Mode.EAGER, Mode.GRAPH_CAPTURING})
    r = Registry()
    r.register(a)
    d = Dispatcher(r)

    x_bf16 = torch.zeros(64, dtype=torch.bfloat16)
    x_fp32 = torch.zeros(64, dtype=torch.float32)

    spy = patch.object(r, "candidates", wraps=r.candidates).start()
    try:
        d.select(op=Op.ALL_REDUCE, mesh=_fake_mesh(), axis="tp", tensor=x_bf16)
        d.select(op=Op.ALL_REDUCE, mesh=_fake_mesh(), axis="tp", tensor=x_fp32)
        # Different dtypes → different cache keys → 2 lookups.
        assert spy.call_count == 2
    finally:
        patch.stopall()


def test_dispatcher_clear_cache():
    a = FakeBackend("a", ops={Op.ALL_REDUCE}, modes={Mode.EAGER, Mode.GRAPH_CAPTURING})
    r = Registry()
    r.register(a)
    d = Dispatcher(r)

    x = torch.zeros(64)
    d.select(op=Op.ALL_REDUCE, mesh=_fake_mesh(), axis="tp", tensor=x)
    assert d._cache  # populated

    spy = patch.object(r, "candidates", wraps=r.candidates).start()
    try:
        d.clear_cache()
        assert not d._cache
        d.select(op=Op.ALL_REDUCE, mesh=_fake_mesh(), axis="tp", tensor=x)
        # After clear, registry queried again.
        assert spy.call_count == 1
    finally:
        patch.stopall()


def test_dispatcher_raises_when_no_backend():
    a = FakeBackend("a", ops={Op.ALL_GATHER}, modes={Mode.EAGER})  # only AG
    r = Registry()
    r.register(a)
    d = Dispatcher(r)

    from phyai.parallel.exceptions import NoBackendError

    with pytest.raises(NoBackendError):
        d.select(op=Op.ALL_REDUCE, mesh=_fake_mesh(), axis="tp", tensor=torch.zeros(8))


def test_dispatcher_uses_force_env(monkeypatch):
    monkeypatch.setenv("PHYAI_FORCE_BACKEND", "b")
    a = FakeBackend("a", ops={Op.ALL_REDUCE}, modes={Mode.EAGER, Mode.GRAPH_CAPTURING})
    b = FakeBackend("b", ops={Op.ALL_REDUCE}, modes={Mode.EAGER, Mode.GRAPH_CAPTURING})
    r = Registry()
    r.register(a)
    r.register(b)
    d = Dispatcher(r)
    chosen = d.select(
        op=Op.ALL_REDUCE, mesh=_fake_mesh(), axis="tp", tensor=torch.zeros(8)
    )
    assert chosen.name == "b"
