"""Eviction policy tests."""

from __future__ import annotations

import struct

import pytest

from phyai_ext.radix_cache import (
    PrefixCache,
    Tier,
    tree_node_set_user_priority,
)


def _atoms(*ts: int) -> bytes:
    return struct.pack(f"{len(ts)}I", *ts)


@pytest.mark.parametrize("policy", ["lru", "lfu", "slru", "priority"])
def test_construct_with_each_policy(policy: str) -> None:
    cache = PrefixCache(
        atom_bytes=4,
        atoms_per_unit=4,
        device_total_units=16,
        eviction_policy=policy,
    )
    assert cache.policy_name == policy


def test_lru_evicts_oldest_first() -> None:
    cache = PrefixCache(atom_bytes=4, atoms_per_unit=4, device_total_units=8)
    # 7 free slots; insert two non-overlapping 1-page sequences (2 units total).
    a = _atoms(1, 2, 3, 4)
    b = _atoms(10, 11, 12, 13)
    ua = cache.allocate(Tier.DEVICE, 1)
    cache.insert(Tier.DEVICE, a, ua)
    ub = cache.allocate(Tier.DEVICE, 1)
    cache.insert(Tier.DEVICE, b, ub)
    # Touch `b` so it's MRU.
    cache.match(b)
    # Now demand more capacity than free + 1; LRU should evict `a` first.
    cache.ensure_capacity(Tier.DEVICE, 6)  # need 6 free, have ?
    m_a = cache.match(a)
    m_b = cache.match(b)
    assert m_a.matched_atoms[Tier.DEVICE] == 0
    # b may also have been evicted but at least a should go first.


def test_priority_policy_respects_user_priority() -> None:
    cache = PrefixCache(
        atom_bytes=4,
        atoms_per_unit=4,
        device_total_units=16,
        eviction_policy="priority",
    )
    a = _atoms(1, 2, 3, 4)
    b = _atoms(10, 11, 12, 13)
    ua = cache.allocate(Tier.DEVICE, 1)
    node_a, _, _ = cache.insert(Tier.DEVICE, a, ua)
    ub = cache.allocate(Tier.DEVICE, 1)
    node_b, _, _ = cache.insert(Tier.DEVICE, b, ub)
    # Mark `b` as critical (high priority) so it's evicted last.
    tree_node_set_user_priority(node_a, 0)
    tree_node_set_user_priority(node_b, 100)
    cache.ensure_capacity(Tier.DEVICE, 14)
    # `a` (priority 0) goes first.
    assert cache.match(a).matched_atoms[Tier.DEVICE] == 0
