"""Native predicate, page hashing, composite NodeRef and storage backend tests."""

from __future__ import annotations

import os
import struct
import tempfile

import pytest

from phyai_ext.radix_cache import (
    CompositeNodeRef,
    PrefixCache,
    Tier,
    file_storage_backend,
    in_memory_storage_backend,
    tree_node_set_user_priority,
    xxh3_64,
)


def _atoms(*ts: int) -> bytes:
    return struct.pack(f"{len(ts)}I", *ts)


def test_evict_by_named_step_le() -> None:
    cache = PrefixCache(atom_bytes=4, atoms_per_unit=4, device_total_units=16)
    atoms = _atoms(*range(8))
    u = cache.allocate(Tier.DEVICE, 2)
    cache.insert(Tier.DEVICE, atoms, u)
    cache.advance_step(100)
    # Touch the inserted node so its step is updated.
    m = cache.match(atoms)
    cache.touch_step(m.last_node[Tier.DEVICE])
    # Now ask the cache to evict everything with step <= 50 — none should match.
    assert cache.evict_by_named_predicate(Tier.DEVICE, "step_le", 50) == 0
    # Evict everything with step <= 200 — should hit our resource.
    freed = cache.evict_by_named_predicate(Tier.DEVICE, "step_le", 200)
    assert freed >= 1


def test_evict_by_named_priority_le() -> None:
    cache = PrefixCache(
        atom_bytes=4,
        atoms_per_unit=4,
        device_total_units=16,
        eviction_policy="priority",
    )
    a = cache.allocate(Tier.DEVICE, 1)
    node_a, _, _ = cache.insert(Tier.DEVICE, _atoms(1, 2, 3, 4), a)
    b = cache.allocate(Tier.DEVICE, 1)
    node_b, _, _ = cache.insert(Tier.DEVICE, _atoms(5, 6, 7, 8), b)
    tree_node_set_user_priority(node_a, 0)
    tree_node_set_user_priority(node_b, 100)
    freed = cache.evict_by_named_predicate(Tier.DEVICE, "priority_le", 0)
    assert freed >= 1
    assert cache.match(_atoms(1, 2, 3, 4)).matched_atoms[Tier.DEVICE] == 0
    assert cache.match(_atoms(5, 6, 7, 8)).matched_atoms[Tier.DEVICE] > 0


def test_evict_by_named_unknown_predicate() -> None:
    cache = PrefixCache(atom_bytes=4, atoms_per_unit=4, device_total_units=8)
    with pytest.raises(ValueError):
        cache.evict_by_named_predicate(Tier.DEVICE, "nope", 0)


def test_xxh3_matches_node_path_hash() -> None:
    cache = PrefixCache(atom_bytes=4, atoms_per_unit=4, device_total_units=16)
    atoms = _atoms(*range(8))
    u = cache.allocate(Tier.DEVICE, 2)
    cache.insert(Tier.DEVICE, atoms, u)
    m = cache.match(atoms)
    h = cache.node_path_hash(m.last_node[Tier.DEVICE])
    expected = xxh3_64(atoms)
    assert h == expected


def test_xxh3_64_known_zero_inputs() -> None:
    # xxh3 of empty input is documented as a fixed constant.
    assert xxh3_64(b"") != 0  # avalanche of seed-mixed value, never zero


def test_lock_multi_pins_two_tiers() -> None:
    cache = PrefixCache(
        atom_bytes=4,
        atoms_per_unit=4,
        device_total_units=8,
        host_total_units=16,
    )
    atoms = _atoms(*range(8))
    udev = cache.allocate(Tier.DEVICE, 2)
    cache.insert(Tier.DEVICE, atoms, udev)
    uhost = cache.allocate(Tier.HOST, 2)
    last_node, _, _ = cache.insert(Tier.HOST, atoms, uhost)

    composite = cache.lock_multi(last_node, [Tier.DEVICE, Tier.HOST])
    assert isinstance(composite, CompositeNodeRef)
    assert composite.size() == 2
    assert composite.valid()
    assert composite.node_handle() == last_node


def test_in_memory_storage_backend_round_trip() -> None:
    backend = in_memory_storage_backend(unit_bytes=8)
    assert backend.name == "in_memory"
    assert backend.unit_bytes == 8
    assert not backend.contains(0xDEADBEEF)
    assert backend.write_sync(op_handle=1, key=0xDEADBEEF, ids=[1, 2, 3])
    assert backend.contains(0xDEADBEEF)
    assert backend.entries() == 1
    assert backend.read_sync(op_handle=2, key=0xDEADBEEF, ids=[1, 2, 3])
    assert not backend.read_sync(op_handle=3, key=0x12345, ids=[1, 2, 3])


def test_file_storage_backend_round_trip() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cache.bin")
        backend = file_storage_backend(path, unit_bytes=16)
        assert backend.name == "file"
        assert backend.write_sync(op_handle=1, key=0x42, ids=[1, 2, 3])
        backend.drain()
        assert backend.read_sync(op_handle=2, key=0x42, ids=[1, 2, 3])
        # Unknown key fails the read.
        assert not backend.read_sync(op_handle=3, key=0x9999, ids=[1, 2, 3])


def test_slru_threshold_is_configurable() -> None:
    cache = PrefixCache(
        atom_bytes=4,
        atoms_per_unit=4,
        device_total_units=16,
        eviction_policy="slru",
        slru_threshold=5,
    )
    assert cache.policy_name == "slru"
    assert cache.slru_threshold == 5
