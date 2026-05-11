"""Async op state-machine tests (start_demote / start_promote / complete_op)."""

from __future__ import annotations

import struct

import pytest

from phyai_ext.radix_cache import CacheEventKind, PrefixCache, Tier


def _atoms(*ts: int) -> bytes:
    return struct.pack(f"{len(ts)}I", *ts)


@pytest.fixture
def two_tier_cache() -> PrefixCache:
    return PrefixCache(
        atom_bytes=4,
        atoms_per_unit=4,
        device_total_units=16,
        host_total_units=32,
    )


def test_demote_success(two_tier_cache: PrefixCache) -> None:
    cache = two_tier_cache
    atoms = _atoms(*range(8))
    units = cache.allocate(Tier.DEVICE, 2)
    last_node, _, _ = cache.insert(Tier.DEVICE, atoms, units)

    # Pre-demote: device tier has the resource Ready.
    m_before = cache.match(atoms)
    assert m_before.matched_atoms[Tier.DEVICE] == 8

    handle = cache.start_demote(last_node, Tier.DEVICE, Tier.HOST)
    assert handle != 0
    # While Pending, host tier is not yet visible to match.
    m_pending = cache.match(atoms)
    assert m_pending.matched_atoms[Tier.HOST] == 0
    assert handle in cache.inflight_ops()

    cache.complete_op(handle, success=True)
    assert handle not in cache.inflight_ops()
    # Host tier now Ready, device cleared.
    m_done = cache.match(atoms)
    assert m_done.matched_atoms[Tier.HOST] == 8
    assert m_done.matched_atoms[Tier.DEVICE] == 0

    # demote_done event emitted.
    kinds = {e.kind for e in cache.take_events()}
    assert int(CacheEventKind.DEMOTE_START) in kinds
    assert int(CacheEventKind.DEMOTE_DONE) in kinds


def test_demote_fail_rolls_back(two_tier_cache: PrefixCache) -> None:
    cache = two_tier_cache
    atoms = _atoms(*range(8))
    units = cache.allocate(Tier.DEVICE, 2)
    last_node, _, _ = cache.insert(Tier.DEVICE, atoms, units)
    handle = cache.start_demote(last_node, Tier.DEVICE, Tier.HOST)
    cache.complete_op(handle, success=False)
    # Source resource intact; dst tier rolled back.
    m = cache.match(atoms)
    assert m.matched_atoms[Tier.DEVICE] == 8
    assert m.matched_atoms[Tier.HOST] == 0
    kinds = {e.kind for e in cache.take_events()}
    assert int(CacheEventKind.DEMOTE_FAIL) in kinds


def test_promote_success(two_tier_cache: PrefixCache) -> None:
    cache = two_tier_cache
    atoms = _atoms(*range(8))
    # Stage the value on host first via direct insert.
    host_units = cache.allocate(Tier.HOST, 2)
    last_node, _, _ = cache.insert(Tier.HOST, atoms, host_units)
    handle = cache.start_promote(last_node, Tier.HOST, Tier.DEVICE)
    cache.complete_op(handle, success=True)

    m = cache.match(atoms)
    assert m.matched_atoms[Tier.DEVICE] == 8
    assert m.matched_atoms[Tier.HOST] == 8
    kinds = {e.kind for e in cache.take_events()}
    assert int(CacheEventKind.PROMOTE_DONE) in kinds


def test_inflight_ops_listing(two_tier_cache: PrefixCache) -> None:
    cache = two_tier_cache
    atoms = _atoms(*range(8))
    units = cache.allocate(Tier.DEVICE, 2)
    last_node, _, _ = cache.insert(Tier.DEVICE, atoms, units)
    h1 = cache.start_demote(last_node, Tier.DEVICE, Tier.HOST)
    assert cache.inflight_ops() == [h1]
    cache.complete_op(h1, success=True)
    assert cache.inflight_ops() == []
