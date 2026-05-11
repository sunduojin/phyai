"""Core PrefixCache tests: match / insert / lock / evict / observers."""

from __future__ import annotations

import struct

import pytest

from phyai_ext.radix_cache import (
    CacheCapacityError,
    CacheEventKind,
    CacheUsageError,
    OwnedUnits,
    PrefixCache,
    Tier,
    tree_node_atom_count,
    tree_node_depth_in_atoms,
    xxh3_64,
)


def _u32_atoms(*tokens: int) -> bytes:
    return struct.pack(f"{len(tokens)}I", *tokens)


def _tensor_to_list(t) -> list[int]:
    """Materialise a tvm-ffi DLPack int32 tensor as a Python list (test helper)."""
    import torch

    return torch.from_dlpack(t).tolist()


@pytest.fixture
def text_cache() -> PrefixCache:
    return PrefixCache(atom_bytes=4, atoms_per_unit=4, device_total_units=64)


def test_owned_units_lifecycle(text_cache: PrefixCache) -> None:
    assert text_cache.available(Tier.DEVICE) == 64 - 1
    units = text_cache.allocate(Tier.DEVICE, 4)
    assert units.size() == 4
    assert text_cache.available(Tier.DEVICE) == 64 - 1 - 4
    del units
    assert text_cache.available(Tier.DEVICE) == 64 - 1


def test_owned_units_take_first_last_append(text_cache: PrefixCache) -> None:
    a = text_cache.allocate(Tier.DEVICE, 6)
    head = a.take_first(2)
    tail = a.take_last(1)
    assert head.size() == 2
    assert tail.size() == 1
    assert a.size() == 3
    head.append(a)
    assert head.size() == 5
    assert a.size() == 0


def test_basic_match_insert(text_cache: PrefixCache) -> None:
    atoms = _u32_atoms(*range(8))
    units = text_cache.allocate(Tier.DEVICE, 2)
    last_node, inserted_atoms, freed = text_cache.insert(Tier.DEVICE, atoms, units)
    assert inserted_atoms == 8
    assert freed == 0
    assert last_node != 0

    m = text_cache.match(atoms)
    assert m.matched_atoms[Tier.DEVICE] == 8
    ids = text_cache.collect_units(m.last_node[Tier.DEVICE], Tier.DEVICE)
    assert sorted(_tensor_to_list(ids)) == [1, 2]


def test_partial_hit_then_extend(text_cache: PrefixCache) -> None:
    atoms_a = _u32_atoms(*range(8))
    units_a = text_cache.allocate(Tier.DEVICE, 2)
    text_cache.insert(Tier.DEVICE, atoms_a, units_a)

    atoms_b = _u32_atoms(*range(12))
    units_b = text_cache.allocate(Tier.DEVICE, 3)
    _, inserted_atoms, freed = text_cache.insert(Tier.DEVICE, atoms_b, units_b)
    assert inserted_atoms == 4
    assert freed == 2

    m = text_cache.match(atoms_b)
    assert m.matched_atoms[Tier.DEVICE] == 12


def test_lock_pins_against_evict(text_cache: PrefixCache) -> None:
    atoms = _u32_atoms(*range(8))
    units = text_cache.allocate(Tier.DEVICE, 2)
    text_cache.insert(Tier.DEVICE, atoms, units)

    m = text_cache.match(atoms)
    ref = text_cache.lock(Tier.DEVICE, m.last_node[Tier.DEVICE])
    text_cache.ensure_capacity(Tier.DEVICE, 63)
    assert text_cache.available(Tier.DEVICE) == 61
    del ref
    text_cache.ensure_capacity(Tier.DEVICE, 63)
    assert text_cache.available(Tier.DEVICE) >= 63


def test_events_stream(text_cache: PrefixCache) -> None:
    atoms = _u32_atoms(*range(8))
    units = text_cache.allocate(Tier.DEVICE, 2)
    text_cache.insert(Tier.DEVICE, atoms, units)
    events = text_cache.take_events()
    kinds = [e.kind for e in events]
    assert int(CacheEventKind.INSERT) in kinds
    assert text_cache.take_events() == []


def test_events_buffer_cap_drops_oldest() -> None:
    cache = PrefixCache(
        atom_bytes=4, atoms_per_unit=4, device_total_units=8, max_events_buffered=2
    )
    a = cache.allocate(Tier.DEVICE, 1)
    cache.insert(Tier.DEVICE, _u32_atoms(1, 2, 3, 4), a)
    b = cache.allocate(Tier.DEVICE, 1)
    cache.insert(Tier.DEVICE, _u32_atoms(5, 6, 7, 8), b)
    c = cache.allocate(Tier.DEVICE, 1)
    cache.insert(Tier.DEVICE, _u32_atoms(9, 10, 11, 12), c)
    events = cache.take_events()
    assert len(events) == 2
    assert cache.dropped_events == 1


def test_match_empty_cache(text_cache: PrefixCache) -> None:
    atoms = _u32_atoms(*range(8))
    m = text_cache.match(atoms)
    assert m.matched_atoms[Tier.DEVICE] == 0
    ids = text_cache.collect_units(m.last_node[Tier.DEVICE], Tier.DEVICE)
    assert _tensor_to_list(ids) == []


def test_invalid_atoms_misalignment(text_cache: PrefixCache) -> None:
    atoms = _u32_atoms(1, 2, 3)
    with pytest.raises(CacheUsageError):
        text_cache.match(atoms)


def test_disabled_tier_raises(text_cache: PrefixCache) -> None:
    assert not text_cache.tier_enabled(Tier.HOST)
    with pytest.raises(CacheUsageError):
        text_cache.allocate(Tier.HOST, 1)


def test_capacity_error_on_overalloc(text_cache: PrefixCache) -> None:
    with pytest.raises(CacheCapacityError):
        text_cache.allocate(Tier.DEVICE, 10_000)


def test_evict_observer_fires() -> None:
    cache = PrefixCache(atom_bytes=4, atoms_per_unit=4, device_total_units=8)
    seen: list[tuple[int, int]] = []
    obs_id = cache.add_evict_observer(lambda node, t: seen.append((int(node), int(t))))

    a = cache.allocate(Tier.DEVICE, 1)
    cache.insert(Tier.DEVICE, _u32_atoms(1, 2, 3, 4), a)
    cache.ensure_capacity(Tier.DEVICE, 7)
    assert len(seen) >= 1
    assert seen[0][1] == int(Tier.DEVICE)

    assert cache.remove_evict_observer(obs_id)
    seen.clear()
    b = cache.allocate(Tier.DEVICE, 1)
    cache.insert(Tier.DEVICE, _u32_atoms(10, 11, 12, 13), b)
    cache.ensure_capacity(Tier.DEVICE, 7)
    # Removed — observer should no longer receive evictions.
    assert seen == []


def test_node_path_hash_stable_across_calls() -> None:
    cache = PrefixCache(atom_bytes=4, atoms_per_unit=4, device_total_units=16)
    atoms = _u32_atoms(*range(8))
    a = cache.allocate(Tier.DEVICE, 2)
    cache.insert(Tier.DEVICE, atoms, a)
    m = cache.match(atoms)
    h1 = cache.node_path_hash(m.last_node[Tier.DEVICE])
    h2 = cache.node_path_hash(m.last_node[Tier.DEVICE])
    assert h1 == h2
    assert h1 != 0


def test_collect_units_returns_dlpack_tensor() -> None:
    import torch

    cache = PrefixCache(atom_bytes=4, atoms_per_unit=4, device_total_units=64)
    atoms_a = _u32_atoms(*range(8))  # 2 pages
    atoms_b = _u32_atoms(*range(12))  # 3 pages, shares atoms_a's first 2

    ua = cache.allocate(Tier.DEVICE, 2)
    cache.insert(Tier.DEVICE, atoms_a, ua)
    ub = cache.allocate(Tier.DEVICE, 3)
    cache.insert(Tier.DEVICE, atoms_b, ub)

    m = cache.match(atoms_b)
    ids_tensor = cache.collect_units(m.last_node[Tier.DEVICE], Tier.DEVICE)

    # The returned object is a tvm-ffi tensor of int32 on CPU.
    assert ids_tensor.shape == (3,)

    # Zero-copy import into torch.
    ids = torch.from_dlpack(ids_tensor)
    assert ids.dtype == torch.int32
    # First two come from the original atoms_a insert (allocator handed out
    # ids 1..2). The third comes from the suffix appended on the second
    # insert; its exact value depends on allocator LIFO order, but the path
    # must contain three distinct positive ids that match the cache state.
    ids_list = ids.tolist()
    assert len(ids_list) == 3
    assert all(x > 0 for x in ids_list)
    assert ids_list[:2] == [1, 2]


def test_collect_units_empty_for_root() -> None:
    import torch

    cache = PrefixCache(atom_bytes=4, atoms_per_unit=4, device_total_units=8)
    atoms = _u32_atoms(*range(4))
    m = cache.match(atoms)  # no insert → match returns root
    ids_tensor = cache.collect_units(m.last_node[Tier.DEVICE], Tier.DEVICE)
    assert ids_tensor.shape == (0,)
    assert torch.from_dlpack(ids_tensor).numel() == 0


def test_collect_units_outlives_cache_eviction() -> None:
    import torch

    cache = PrefixCache(atom_bytes=4, atoms_per_unit=4, device_total_units=8)
    atoms = _u32_atoms(*range(8))
    u = cache.allocate(Tier.DEVICE, 2)
    cache.insert(Tier.DEVICE, atoms, u)

    m = cache.match(atoms)
    ids_tensor = cache.collect_units(m.last_node[Tier.DEVICE], Tier.DEVICE)
    ids_before = torch.from_dlpack(ids_tensor).clone().tolist()

    # Force eviction. The C++ tensor was allocated as an owned copy, so its
    # contents must remain valid even after the source units are recycled.
    cache.ensure_capacity(Tier.DEVICE, 7)
    ids_after = torch.from_dlpack(ids_tensor).tolist()
    assert ids_after == ids_before


# ---------------------------------------------------------------------------
# Mid-segment node split (radix_tree::walk → tree_node::split_self).
#
# All other tests insert sequences that either match exactly or strictly
# extend an existing node; none of them force the walker to fork an existing
# node mid-segment. These do.


def test_mid_segment_split_forks_existing_node(text_cache: PrefixCache) -> None:
    atoms_a = _u32_atoms(*range(8))  # [0,1,2,3,4,5,6,7]
    atoms_b = _u32_atoms(0, 1, 2, 3, 9, 9, 9, 9)  # shares first 4 atoms

    ua = text_cache.allocate(Tier.DEVICE, 2)
    _, ins_a, free_a = text_cache.insert(Tier.DEVICE, atoms_a, ua)
    assert (ins_a, free_a) == (8, 0)

    ub = text_cache.allocate(Tier.DEVICE, 2)
    _, ins_b, free_b = text_cache.insert(Tier.DEVICE, atoms_b, ub)
    # Only the diverging suffix [9,9,9,9] (1 unit) is new; the 1 prefix unit
    # the caller pre-allocated is redundant and must be returned.
    assert (ins_b, free_b) == (4, 1)

    # The shared prefix lands on a freshly-created mid-node.
    m_pref = text_cache.match(_u32_atoms(0, 1, 2, 3))
    mid = m_pref.last_node[Tier.DEVICE]
    assert m_pref.matched_atoms[Tier.DEVICE] == 4
    assert tree_node_atom_count(mid) == 4
    assert tree_node_depth_in_atoms(mid) == 4

    # Both A and B remain fully matchable; their last_nodes are *different*
    # children of the mid-node (atom_count=4 each, depth=8 each).
    m_a = text_cache.match(atoms_a)
    m_b = text_cache.match(atoms_b)
    a_suffix = m_a.last_node[Tier.DEVICE]
    b_node = m_b.last_node[Tier.DEVICE]
    assert m_a.matched_atoms[Tier.DEVICE] == 8
    assert m_b.matched_atoms[Tier.DEVICE] == 8
    assert a_suffix != b_node
    assert a_suffix != mid and b_node != mid
    for n in (a_suffix, b_node):
        assert tree_node_atom_count(n) == 4
        assert tree_node_depth_in_atoms(n) == 8


def test_mid_segment_split_resources_partition_correctly(
    text_cache: PrefixCache,
) -> None:
    atoms_a = _u32_atoms(*range(8))
    atoms_b = _u32_atoms(0, 1, 2, 3, 9, 9, 9, 9)
    text_cache.insert(Tier.DEVICE, atoms_a, text_cache.allocate(Tier.DEVICE, 2))
    text_cache.insert(Tier.DEVICE, atoms_b, text_cache.allocate(Tier.DEVICE, 2))

    mid = text_cache.match(_u32_atoms(0, 1, 2, 3)).last_node[Tier.DEVICE]
    a_suffix = text_cache.match(atoms_a).last_node[Tier.DEVICE]
    b_node = text_cache.match(atoms_b).last_node[Tier.DEVICE]

    pref_ids = _tensor_to_list(text_cache.collect_units(mid, Tier.DEVICE))
    a_ids = _tensor_to_list(text_cache.collect_units(a_suffix, Tier.DEVICE))
    b_ids = _tensor_to_list(text_cache.collect_units(b_node, Tier.DEVICE))

    # mid owns exactly the 1 prefix unit; A_suffix and B both inherit it via
    # the parent chain, then add their own unit.
    assert len(pref_ids) == 1
    assert len(a_ids) == 2 and a_ids[0] == pref_ids[0]
    assert len(b_ids) == 2 and b_ids[0] == pref_ids[0]
    # Suffix units are distinct (separate tier slots).
    assert a_ids[1] != b_ids[1]
    # Three distinct unit ids in total: prefix + A_suffix + B.
    assert len({pref_ids[0], a_ids[1], b_ids[1]}) == 3


def test_mid_segment_split_path_hash(text_cache: PrefixCache) -> None:
    atoms_a = _u32_atoms(*range(8))
    atoms_b = _u32_atoms(0, 1, 2, 3, 9, 9, 9, 9)
    text_cache.insert(Tier.DEVICE, atoms_a, text_cache.allocate(Tier.DEVICE, 2))
    text_cache.insert(Tier.DEVICE, atoms_b, text_cache.allocate(Tier.DEVICE, 2))

    mid = text_cache.match(_u32_atoms(0, 1, 2, 3)).last_node[Tier.DEVICE]
    a_suffix = text_cache.match(atoms_a).last_node[Tier.DEVICE]
    b_node = text_cache.match(atoms_b).last_node[Tier.DEVICE]

    # node_path_hash is content-addressed over root → node atom bytes; after
    # split it must still reflect the original sequence, not the post-split
    # node boundaries.
    assert text_cache.node_path_hash(mid) == xxh3_64(_u32_atoms(0, 1, 2, 3))
    assert text_cache.node_path_hash(a_suffix) == xxh3_64(atoms_a)
    assert text_cache.node_path_hash(b_node) == xxh3_64(atoms_b)


def test_mid_segment_split_under_lock(text_cache: PrefixCache) -> None:
    atoms_a = _u32_atoms(*range(8))
    atoms_b = _u32_atoms(0, 1, 2, 3, 9, 9, 9, 9)

    text_cache.insert(Tier.DEVICE, atoms_a, text_cache.allocate(Tier.DEVICE, 2))
    pre_split_node = text_cache.match(atoms_a).last_node[Tier.DEVICE]
    ref = text_cache.lock(Tier.DEVICE, pre_split_node)
    assert ref.valid()

    # Triggering the split must not invalidate or move out from under the
    # held NodeRef; both A and B remain reachable afterwards.
    text_cache.insert(Tier.DEVICE, atoms_b, text_cache.allocate(Tier.DEVICE, 2))

    assert ref.valid()
    assert text_cache.match(atoms_a).matched_atoms[Tier.DEVICE] == 8
    assert text_cache.match(atoms_b).matched_atoms[Tier.DEVICE] == 8

    del ref


def test_mid_segment_three_way_fork(text_cache: PrefixCache) -> None:
    atoms_a = _u32_atoms(0, 1, 2, 3, 4, 4, 4, 4)
    atoms_b = _u32_atoms(0, 1, 2, 3, 5, 5, 5, 5)
    atoms_c = _u32_atoms(0, 1, 2, 3, 6, 6, 6, 6)
    for atoms in (atoms_a, atoms_b, atoms_c):
        text_cache.insert(Tier.DEVICE, atoms, text_cache.allocate(Tier.DEVICE, 2))

    # All three full sequences match independently to 8 atoms.
    leaves = []
    for atoms in (atoms_a, atoms_b, atoms_c):
        m = text_cache.match(atoms)
        assert m.matched_atoms[Tier.DEVICE] == 8
        leaves.append(m.last_node[Tier.DEVICE])
    assert len(set(leaves)) == 3

    # Shared prefix matches 4 atoms; mid-node is one common ancestor.
    mid = text_cache.match(_u32_atoms(0, 1, 2, 3)).last_node[Tier.DEVICE]
    assert tree_node_atom_count(mid) == 4
    for leaf in leaves:
        assert leaf != mid
        assert tree_node_depth_in_atoms(leaf) == 8
