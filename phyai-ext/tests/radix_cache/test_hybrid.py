"""Hybrid Mamba + Attention shared-tree cache tests."""

from __future__ import annotations

import struct

import pytest

from phyai_ext.radix_cache import (
    HybridPrefixCache,
    PrefixCache,
    Tier,
    tree_node_has_mamba,
    tree_node_mamba_index,
)


def _atoms(*ts: int) -> bytes:
    return struct.pack(f"{len(ts)}I", *ts)


@pytest.fixture
def hybrid() -> HybridPrefixCache:
    kv = PrefixCache(atom_bytes=4, atoms_per_unit=4, device_total_units=64)
    return HybridPrefixCache(kv, num_mamba_slots=8)


def test_mamba_slot_alloc_free(hybrid: HybridPrefixCache) -> None:
    assert hybrid.available_slots == 8
    s1 = hybrid.allocate_mamba_slot()
    assert s1 is not None
    assert s1.valid()
    assert hybrid.available_slots == 7
    del s1
    # RAII frees back to allocator
    assert hybrid.available_slots == 8


def test_attach_mamba_to_node(hybrid: HybridPrefixCache) -> None:
    atoms = _atoms(*range(8))
    units = hybrid.kv.allocate(Tier.DEVICE, 2)
    last_node, _, _ = hybrid.kv.insert(Tier.DEVICE, atoms, units)
    slot = hybrid.allocate_mamba_slot()
    assert slot is not None
    slot_idx = slot.index()
    hybrid.attach_mamba(last_node, slot)
    assert tree_node_has_mamba(last_node)
    assert tree_node_mamba_index(last_node) == slot_idx


def test_mamba_evicts_when_kv_evicts(hybrid: HybridPrefixCache) -> None:
    # Fill up KV cache and attach mambas
    atoms_a = _atoms(*range(8))
    ua = hybrid.kv.allocate(Tier.DEVICE, 2)
    node_a, _, _ = hybrid.kv.insert(Tier.DEVICE, atoms_a, ua)
    s_a = hybrid.allocate_mamba_slot()
    assert s_a is not None
    hybrid.attach_mamba(node_a, s_a)
    assert hybrid.active_slots == 1

    # Now force eviction of node_a's KV resource
    hybrid.kv.ensure_capacity(Tier.DEVICE, 63)  # demand all but 1 slot free
    # mamba slot should have been auto-detached via on_kv_evict observer
    assert hybrid.active_slots == 0


def test_hybrid_match_finds_mamba_ancestor(hybrid: HybridPrefixCache) -> None:
    # Build path: insert two pages, attach mamba on first page only; matching
    # the full sequence should find the mamba on the ancestor.
    atoms = _atoms(*range(8))  # 2 pages
    units = hybrid.kv.allocate(Tier.DEVICE, 2)
    last_node, _, _ = hybrid.kv.insert(Tier.DEVICE, atoms, units)
    # last_node is the deeper page; attach mamba there to make the match obvious
    slot = hybrid.allocate_mamba_slot()
    assert slot is not None
    hybrid.attach_mamba(last_node, slot)

    m = hybrid.match(atoms)
    assert m.kv.matched_atoms[Tier.DEVICE] == 8
    assert m.last_mamba_node == last_node
    assert m.mamba_branching_atoms == 8


def test_ensure_mamba_capacity_evict(hybrid: HybridPrefixCache) -> None:
    # Allocate all 8 slots, attach to fake nodes (we'll attach to one inserted
    # node since attach requires a tree_node).
    atoms = _atoms(*range(4))
    ua = hybrid.kv.allocate(Tier.DEVICE, 1)
    node, _, _ = hybrid.kv.insert(Tier.DEVICE, atoms, ua)
    # exhaust slots
    held = []
    for _ in range(8):
        held.append(hybrid.allocate_mamba_slot())
        assert held[-1] is not None
    # Attach the first slot to the inserted node so the LRU eviction can
    # find it via the mamba_leaves_ set.
    first_slot = held.pop(0)
    hybrid.attach_mamba(node, first_slot)
    # Free the rest by clearing the list (RAII frees them).
    held.clear()
    assert hybrid.available_slots == 7

    # Now ask for capacity that requires evicting the attached one
    assert hybrid.ensure_mamba_capacity_by_evict(8)
    assert hybrid.available_slots == 8
