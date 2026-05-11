"""SWA paired-cache tests."""

from __future__ import annotations

import struct

import pytest

from phyai.radix_cache import (
    CacheConfig,
    PairedCache,
    PairedCacheConfig,
)
from phyai_ext.radix_cache import Tier


def _atoms(*ts: int) -> bytes:
    return struct.pack(f"{len(ts)}I", *ts)


def _make_paired(window_size: int = 8) -> PairedCache:
    full_cfg = CacheConfig(atom_bytes=4, atoms_per_unit=4, device_total_units=64)
    swa_cfg = CacheConfig(atom_bytes=4, atoms_per_unit=4, device_total_units=32)
    return PairedCache(
        PairedCacheConfig(full=full_cfg, swa=swa_cfg, window_size=window_size)
    )


def test_paired_match_full_swa() -> None:
    pc = _make_paired()
    atoms = _atoms(*range(8))
    fu, su = pc.allocate_pair(2)
    pc.insert_pair(atoms, fu, su)
    m = pc.match(atoms)
    assert m.full.matched_atoms[Tier.DEVICE] == 8
    assert m.swa.matched_atoms[Tier.DEVICE] == 8


def test_paired_advance_evicts_swa_only() -> None:
    pc = _make_paired(window_size=4)
    atoms = _atoms(*range(8))
    fu, su = pc.allocate_pair(2)
    pc.insert_pair(atoms, fu, su)
    full_active_before = pc.full.active(Tier.DEVICE)
    swa_active_before = pc.swa.active(Tier.DEVICE)
    assert full_active_before > 0 and swa_active_before > 0

    # Advance 100 atoms past insertion's step → all SWA leaves should be evicted.
    freed = pc.advance(100)
    # Full cache untouched
    assert pc.full.active(Tier.DEVICE) == full_active_before
    # SWA cache shed at least one resource
    assert pc.swa.active(Tier.DEVICE) < swa_active_before
    assert freed >= 1
