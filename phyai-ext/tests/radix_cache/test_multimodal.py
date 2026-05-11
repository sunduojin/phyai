"""Multimodal cache + encoding helpers tests."""

from __future__ import annotations

import hashlib
import struct

import pytest

from phyai.radix_cache import (
    CacheConfig,
    Modality,
    MultimodalCache,
    encoding,
)
from phyai_ext.radix_cache import Tier


def test_multimodal_cache_routing() -> None:
    cache = MultimodalCache(
        {
            Modality.TEXT: CacheConfig(
                atom_bytes=4,
                atoms_per_unit=4,
                device_total_units=64,
                host_total_units=256,
            ),
            Modality.IMAGE: CacheConfig(
                atom_bytes=32,
                atoms_per_unit=1,
                device_total_units=16,
                host_total_units=128,
            ),
            Modality.VIDEO: CacheConfig(
                atom_bytes=8,
                atoms_per_unit=1,
                device_total_units=16,
                host_total_units=128,
            ),
            Modality.AUDIO: CacheConfig(
                atom_bytes=8,
                atoms_per_unit=1,
                device_total_units=16,
                host_total_units=128,
            ),
        }
    )
    for m in Modality:
        assert m in cache
        c = cache[m]
        assert c.tier_enabled(Tier.DEVICE)


def test_image_blob_cache_hit() -> None:
    c = CacheConfig(
        atom_bytes=32, atoms_per_unit=1, device_total_units=8, host_total_units=64
    ).build()
    digest = hashlib.sha256(b"hello").digest()
    assert len(digest) == 32
    units = c.allocate(Tier.DEVICE, 1)
    c.insert(Tier.DEVICE, digest, units)
    m = c.match(digest)
    assert m.matched_atoms[Tier.DEVICE] == 1


def test_video_chunk_cache_prefix_share() -> None:
    c = CacheConfig(
        atom_bytes=8, atoms_per_unit=1, device_total_units=16, host_total_units=128
    ).build()
    chunks_a = encoding.encode_chunk_hash_atoms_u64([0xA1, 0xA2, 0xA3])
    chunks_b = encoding.encode_chunk_hash_atoms_u64(
        [0xA1, 0xA2, 0xB4]
    )  # shares first 2
    ua = c.allocate(Tier.DEVICE, 3)
    c.insert(Tier.DEVICE, chunks_a, ua)
    ub = c.allocate(Tier.DEVICE, 3)
    _, inserted, freed = c.insert(Tier.DEVICE, chunks_b, ub)
    # 2-chunk prefix shared, 1 new
    assert inserted == 1  # one atom in suffix
    assert freed == 2


def test_eagle_bigram_encoding() -> None:
    # 5 token ids → 4 bigram atoms (each 8 bytes)
    bytes_ = encoding.encode_eagle_bigram_atoms([10, 20, 30, 40, 50])
    assert len(bytes_) == 8 * 4
    # Match the wire format: (curr, next) int32 pairs, little-endian
    pairs = [struct.unpack("ii", bytes_[i : i + 8]) for i in range(0, len(bytes_), 8)]
    assert pairs == [(10, 20), (20, 30), (30, 40), (40, 50)]


def test_multimodal_mixed_atom_encoding() -> None:
    text_tokens = [100, 101, 102, 103]  # 4 text tokens
    seg = encoding.ImageSegment(
        image_hash_lo=0xBEEFDEAD, image_hash_hi=0x1234ABCD, insert_position=2, length=2
    )
    atoms = encoding.encode_prompt_to_atoms(text_tokens, [seg])
    # 4 + 2 = 6 atoms × 16 B
    assert len(atoms) == 6 * 16
    # Atom 0 (text token 100): lo=100, hi=0
    lo0, hi0 = struct.unpack("<QQ", atoms[0:16])
    assert lo0 == 100 and hi0 == 0
    # Atom 2 (image first): lo=0xBEEFDEAD, hi=0x1234ABCD ^ 0
    lo2, hi2 = struct.unpack("<QQ", atoms[2 * 16 : 3 * 16])
    assert lo2 == 0xBEEFDEAD and hi2 == 0x1234ABCD
    # Atom 3 (image second): hi has pos=1 XORed in
    lo3, hi3 = struct.unpack("<QQ", atoms[3 * 16 : 4 * 16])
    assert lo3 == 0xBEEFDEAD and hi3 == (0x1234ABCD ^ 1)


def test_page_align() -> None:
    aligned, dropped = encoding.page_align(b"x" * 27, page_bytes=8)
    assert len(aligned) == 24
    assert dropped == 3


def test_text_cache_full_match() -> None:
    c = CacheConfig(atom_bytes=4, atoms_per_unit=4, device_total_units=64).build()
    atoms = encoding.encode_text_atoms_int32([1, 2, 3, 4, 5, 6, 7, 8])
    u = c.allocate(Tier.DEVICE, 2)
    c.insert(Tier.DEVICE, atoms, u)
    m = c.match(atoms)
    assert m.matched_atoms[Tier.DEVICE] == 8
