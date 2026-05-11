"""Cross-tier eviction cascade and nesting tests."""

from __future__ import annotations

import struct

import pytest

from phyai.radix_cache import (
    CacheConfig,
    HybridCache,
    HybridCacheConfig,
    Modality,
    NestedCache,
    PairedCache,
    PairedCacheConfig,
)
from phyai_ext.radix_cache import PrefixCache, Tier


def _atoms(*ts: int) -> bytes:
    return struct.pack(f"{len(ts)}I", *ts)


def test_cascade_evict_when_destination_full() -> None:
    # Device tier has 8 units, host has 5 (4 usable after the null sentinel).
    # Fill device with three 2-unit segments. Demote the first two so host
    # is full. Asking ensure_capacity to push the third must cascade: the
    # cache should evict the host LRU first to make room.
    cache = PrefixCache(
        atom_bytes=4,
        atoms_per_unit=4,
        device_total_units=8,
        host_total_units=5,
    )
    seqs = [_atoms(*range(0, 8)), _atoms(*range(20, 28)), _atoms(*range(40, 48))]
    nodes = []
    for s in seqs:
        u = cache.allocate(Tier.DEVICE, 2)
        n, _, _ = cache.insert(Tier.DEVICE, s, u)
        nodes.append(n)
    h_a = cache.start_demote(nodes[0], Tier.DEVICE, Tier.HOST)
    cache.complete_op(h_a, success=True)
    h_b = cache.start_demote(nodes[1], Tier.DEVICE, Tier.HOST)
    cache.complete_op(h_b, success=True)
    assert cache.available(Tier.HOST) == 0
    cache.ensure_capacity(Tier.DEVICE, 7, promote_to=Tier.HOST)
    assert cache.active(Tier.DEVICE) <= 4


def test_nested_cache_routes_to_hybrid() -> None:
    text_cfg = CacheConfig(
        atom_bytes=4, atoms_per_unit=4, device_total_units=16, host_total_units=64
    )
    hybrid = HybridCache(
        HybridCacheConfig(
            kv=text_cfg,
            num_mamba_slots=4,
            layer_kinds=("attn", "mamba"),
        )
    )
    image_cache = CacheConfig(
        atom_bytes=32, atoms_per_unit=1, device_total_units=8, host_total_units=64
    ).build()
    nested = NestedCache({Modality.TEXT: hybrid, Modality.IMAGE: image_cache})
    assert nested[Modality.TEXT] is hybrid
    assert nested[Modality.IMAGE] is image_cache


def test_paired_inside_nested() -> None:
    paired = PairedCache(
        PairedCacheConfig(
            full=CacheConfig(atom_bytes=4, atoms_per_unit=4, device_total_units=32),
            swa=CacheConfig(atom_bytes=4, atoms_per_unit=4, device_total_units=16),
            window_size=8,
        )
    )
    nested = NestedCache({"text": paired})
    assert "text" in nested
    p = nested["text"]
    assert isinstance(p, PairedCache)
    assert p.window_size == 8
