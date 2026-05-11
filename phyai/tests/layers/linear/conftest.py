"""Shared fixtures for phyai.layers.linear tests.

We avoid spinning up a real process group for unit tests — a mock
``Mesh`` registered under the usual ``"model"`` name is enough for
``resolve_mesh`` to find. Layer tests that exercise collectives at
ws>1 live under a separate multiprocess harness (see
``tests/parallel/multiprocess.py``).

The ``FakeKernel`` / ``make_probe`` factories live here so they're
shared without inter-test-file relative imports (which don't work
under pytest ``--import-mode=importlib`` unless ``phyai/tests`` is on
``pythonpath``).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from phyai.layers.linear import _reset_for_test
from phyai.layers.linear.backend import KernelProbe
from phyai.parallel.mesh import Mesh
from phyai.parallel.state import Mode, _meshes, register_mesh


def _fake_mesh(
    *,
    name: str = "model",
    sizes: dict[str, int] | None = None,
    ranks: dict[str, int] | None = None,
) -> Mesh:
    sizes = sizes or {}
    ranks = ranks or {}

    def size_of(axis: str) -> int:
        return sizes.get(axis, 1)

    def rank_of(axis: str) -> int:
        return ranks.get(axis, 0)

    tm = MagicMock()
    tm.mesh_dim_names = tuple(sizes.keys()) if sizes else ()
    _names = tm.mesh_dim_names

    def _size(axis):
        if isinstance(axis, str):
            return size_of(axis)
        return size_of(_names[axis])

    tm.size.side_effect = _size
    tm.get_local_rank.side_effect = rank_of
    tm.get_group.side_effect = lambda axis: MagicMock(name=f"pg-{axis}")
    mesh = Mesh(tm, name=name)
    register_mesh(mesh)
    return mesh


@pytest.fixture
def fake_mesh():
    saved = dict(_meshes)
    try:
        yield _fake_mesh
    finally:
        _meshes.clear()
        _meshes.update(saved)
        _reset_for_test()


# ---------------------------------------------------------------------------
# Shared FakeKernel + probe helper (used by test_registry / test_dispatch)
# ---------------------------------------------------------------------------


class FakeKernel:
    """Configurable stand-in for a real LinearKernel."""

    def __init__(
        self,
        name: str,
        *,
        specs: set[str],
        modes: set[Mode],
        capture: bool = True,
        min_sm: int = 0,
    ) -> None:
        self.name = name
        self._specs = specs
        self._modes = modes
        self._capture = capture
        self._min_sm = min_sm
        self.applied = 0

    def supports_capture(self) -> bool:
        return self._capture

    def can_handle(self, probe: KernelProbe) -> bool:
        if probe.spec_id not in self._specs:
            return False
        if probe.mode not in self._modes:
            return False
        if probe.sm < self._min_sm:
            return False
        return True

    def apply(self, layer, x, bias):  # pragma: no cover
        self.applied += 1
        return torch.empty_like(x)


def make_probe(
    *,
    spec_id: str = "bf16",
    M_bucket: int = 1,
    N: int = 512,
    K: int = 512,
    mode: Mode = Mode.EAGER,
    sm: int = 90,
    in_dtype: torch.dtype = torch.bfloat16,
    out_dtype: torch.dtype = torch.bfloat16,
) -> KernelProbe:
    return KernelProbe(
        spec_id=spec_id,
        M_bucket=M_bucket,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        sm=sm,
        mode=mode,
    )


@pytest.fixture
def fake_kernel():
    """Return the :class:`FakeKernel` class for direct instantiation."""
    return FakeKernel


@pytest.fixture
def probe():
    """Return the :func:`make_probe` helper."""
    return make_probe
