"""Pending-units budget enforcement and crash-recovery tests."""

from __future__ import annotations

import struct

import pytest

from phyai_ext.radix_cache import (
    CacheCapacityError,
    CacheEventKind,
    PrefixCache,
    Tier,
)


def _atoms(*ts: int) -> bytes:
    return struct.pack(f"{len(ts)}I", *ts)


def test_pending_units_tracking_and_release() -> None:
    cache = PrefixCache(
        atom_bytes=4,
        atoms_per_unit=4,
        device_total_units=16,
        host_total_units=32,
    )
    atoms = _atoms(*range(8))
    units = cache.allocate(Tier.DEVICE, 2)
    last_node, _, _ = cache.insert(Tier.DEVICE, atoms, units)
    assert cache.pending_units(Tier.HOST) == 0
    h = cache.start_demote(last_node, Tier.DEVICE, Tier.HOST)
    assert cache.pending_units(Tier.HOST) == 2

    cache.complete_op(h, success=True)
    assert cache.pending_units(Tier.HOST) == 0


def test_max_pending_units_blocks_overflow() -> None:
    cache = PrefixCache(
        atom_bytes=4,
        atoms_per_unit=4,
        device_total_units=16,
        host_total_units=32,
        host_max_pending_units=2,
    )
    a = cache.allocate(Tier.DEVICE, 2)
    node_a, _, _ = cache.insert(Tier.DEVICE, _atoms(*range(8)), a)
    h_a = cache.start_demote(node_a, Tier.DEVICE, Tier.HOST)
    assert cache.pending_units(Tier.HOST) == 2
    assert cache.max_pending_units(Tier.HOST) == 2

    b = cache.allocate(Tier.DEVICE, 2)
    node_b, _, _ = cache.insert(Tier.DEVICE, _atoms(*range(10, 18)), b)
    with pytest.raises(CacheCapacityError):
        cache.start_demote(node_b, Tier.DEVICE, Tier.HOST)

    cache.complete_op(h_a, success=True)
    # Pending budget freed; a new demote now succeeds.
    h_b = cache.start_demote(node_b, Tier.DEVICE, Tier.HOST)
    assert h_b != 0
    cache.complete_op(h_b, success=True)


def test_fail_all_inflight_resets_state() -> None:
    cache = PrefixCache(
        atom_bytes=4,
        atoms_per_unit=4,
        device_total_units=16,
        host_total_units=32,
    )
    a = cache.allocate(Tier.DEVICE, 2)
    node_a, _, _ = cache.insert(Tier.DEVICE, _atoms(*range(8)), a)
    b = cache.allocate(Tier.DEVICE, 2)
    node_b, _, _ = cache.insert(Tier.DEVICE, _atoms(*range(10, 18)), b)
    cache.start_demote(node_a, Tier.DEVICE, Tier.HOST)
    cache.start_demote(node_b, Tier.DEVICE, Tier.HOST)
    assert len(cache.inflight_ops()) == 2

    failed = cache.fail_all_inflight()
    assert failed == 2
    assert cache.inflight_ops() == []
    assert cache.pending_units(Tier.HOST) == 0
    # Source resources still hit on device — only the destination side rolled back.
    m = cache.match(_atoms(*range(8)))
    assert m.matched_atoms[Tier.DEVICE] == 8
    assert m.matched_atoms[Tier.HOST] == 0
    kinds = {e.kind for e in cache.take_events()}
    assert int(CacheEventKind.DEMOTE_FAIL) in kinds


def test_wait_op_returns_true_after_completion() -> None:
    cache = PrefixCache(
        atom_bytes=4,
        atoms_per_unit=4,
        device_total_units=16,
        host_total_units=32,
    )
    units = cache.allocate(Tier.DEVICE, 2)
    last_node, _, _ = cache.insert(Tier.DEVICE, _atoms(*range(8)), units)
    h = cache.start_demote(last_node, Tier.DEVICE, Tier.HOST)
    cache.complete_op(h, success=True)
    # Already done — wait should return True immediately.
    assert cache.wait_op(h, timeout_ms=10)


def test_wait_op_times_out_when_pending() -> None:
    cache = PrefixCache(
        atom_bytes=4,
        atoms_per_unit=4,
        device_total_units=16,
        host_total_units=32,
    )
    units = cache.allocate(Tier.DEVICE, 2)
    last_node, _, _ = cache.insert(Tier.DEVICE, _atoms(*range(8)), units)
    h = cache.start_demote(last_node, Tier.DEVICE, Tier.HOST)
    assert not cache.wait_op(h, timeout_ms=10)
    cache.complete_op(h, success=True)
