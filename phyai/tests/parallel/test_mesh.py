"""Mock-based tests for Mesh.

These don't require real NCCL — they test the wrapper's Python behaviour
(axis lookups, topology inference defaults).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from phyai.parallel.mesh import Mesh, _infer_topology


def _fake_torch_mesh(sizes: dict[str, int]) -> MagicMock:
    """Build a stand-in for torch.distributed.DeviceMesh."""
    m = MagicMock()
    names = tuple(sizes.keys())
    m.mesh_dim_names = names
    # torch 2.10 regression means phyai.Mesh resolves name → index and
    # queries .size(idx). Both paths should yield the same answer here.
    m.size.side_effect = lambda axis: (
        sizes[axis] if isinstance(axis, str) else sizes[names[axis]]
    )
    m.get_local_rank.side_effect = lambda axis: 0
    m.get_group.side_effect = lambda axis: MagicMock(name=f"pg-{axis}")
    return m


def test_mesh_axis_size():
    m = Mesh(_fake_torch_mesh({"tp": 4, "dp": 2}), name="model")
    assert m.axis_size("tp") == 4
    assert m.axis_size("dp") == 2


def test_mesh_axis_group_caches():
    m = Mesh(_fake_torch_mesh({"tp": 4}), name="model")
    g1 = m.axis_group("tp")
    g2 = m.axis_group("tp")
    assert g1 is g2  # cached, same object


def test_mesh_name():
    m = Mesh(_fake_torch_mesh({"tp": 4}), name="draft")
    assert m.name == "draft"


def test_topology_single_node_assumed_full_nvlink(monkeypatch):
    """Without a real init_process_group, _infer_topology should default to
    a single-node nvlink topology (best-effort)."""
    fake = _fake_torch_mesh({"tp": 4})
    # dist.is_initialized() returns False by default in tests.
    topo = _infer_topology(fake)
    assert topo.is_single_node is True
    assert topo.is_full_nvlink is True
    assert topo.n_nodes == 1
    assert topo.n_gpus_per_node >= 1
