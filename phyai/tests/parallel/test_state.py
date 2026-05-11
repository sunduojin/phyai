"""Tests for ContextVar mode + use_mesh."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from phyai.parallel.mesh import Mesh
from phyai.parallel.state import (
    Mode,
    _meshes,
    current_mode,
    default_mesh,
    graph_capture,
    register_mesh,
    use_mesh,
)


@contextmanager
def _isolated_meshes():
    """Snapshot+restore the global mesh registry so tests stay independent."""
    saved = dict(_meshes)
    _meshes.clear()
    try:
        yield
    finally:
        _meshes.clear()
        _meshes.update(saved)


def _fake_mesh(name: str = "model") -> Mesh:
    m = MagicMock()
    m.size.side_effect = lambda axis: 4
    m.get_local_rank.side_effect = lambda axis: 0
    m.get_group.side_effect = lambda axis: MagicMock()
    return Mesh(m, name=name)


def test_default_mode_is_eager():
    assert current_mode() is Mode.EAGER


def test_graph_capture_switches_mode():
    assert current_mode() is Mode.EAGER
    with graph_capture():
        assert current_mode() is Mode.GRAPH_CAPTURING
    assert current_mode() is Mode.EAGER


def test_graph_capture_restores_on_exception():
    try:
        with graph_capture():
            assert current_mode() is Mode.GRAPH_CAPTURING
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert current_mode() is Mode.EAGER


def test_default_mesh_returns_registered():
    with _isolated_meshes():
        m = _fake_mesh("model")
        register_mesh(m)
        assert default_mesh() is m


def test_default_mesh_raises_when_unset():
    with _isolated_meshes():
        with pytest.raises(KeyError):
            default_mesh()


def test_use_mesh_switches_default():
    with _isolated_meshes():
        m1 = _fake_mesh("model")
        m2 = _fake_mesh("draft")
        register_mesh(m1)
        register_mesh(m2)
        assert default_mesh() is m1
        with use_mesh("draft"):
            assert default_mesh() is m2
        assert default_mesh() is m1


def test_use_mesh_restores_on_exception():
    with _isolated_meshes():
        m1 = _fake_mesh("model")
        m2 = _fake_mesh("draft")
        register_mesh(m1)
        register_mesh(m2)
        try:
            with use_mesh("draft"):
                raise RuntimeError("x")
        except RuntimeError:
            pass
        assert default_mesh() is m1
