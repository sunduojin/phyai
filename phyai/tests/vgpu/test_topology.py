"""Round-up + capacity helpers (pure-function unit tests)."""

from __future__ import annotations

import pytest

from phyai.vgpu.topology import (
    get_sm_count_constraint,
    round_up_sm_count,
    validate_total,
)
from phyai.vgpu.exceptions import VGPURuntimeError


def test_constraint_table_matches_flashinfer():
    """Table should match flashinfer.green_ctx.get_sm_count_constraint."""
    assert get_sm_count_constraint((6, 0)) == (1, 1)
    assert get_sm_count_constraint((7, 0)) == (2, 2)
    assert get_sm_count_constraint((7, 5)) == (2, 2)
    assert get_sm_count_constraint((8, 0)) == (4, 2)
    assert get_sm_count_constraint((8, 6)) == (4, 2)
    assert get_sm_count_constraint((9, 0)) == (8, 8)
    assert get_sm_count_constraint((10, 0)) == (8, 8)


def test_constraint_rejects_unsupported_cc():
    with pytest.raises(VGPURuntimeError):
        get_sm_count_constraint((5, 0))


def test_round_up_h20z_examples():
    """H20Z (CC 9.0): minimum 8, alignment 8."""
    cc = (9, 0)
    assert round_up_sm_count(7, cc) == 8  # rounded to minimum
    assert round_up_sm_count(8, cc) == 8  # exact
    assert round_up_sm_count(10, cc) == 16  # next multiple of 8
    assert round_up_sm_count(16, cc) == 16  # exact
    assert round_up_sm_count(17, cc) == 24  # next multiple of 8


def test_round_up_ampere_alignment():
    """CC 8.x: minimum 4, alignment 2."""
    cc = (8, 0)
    assert round_up_sm_count(1, cc) == 4  # below minimum
    assert round_up_sm_count(3, cc) == 4  # below minimum
    assert round_up_sm_count(4, cc) == 4  # exact
    assert round_up_sm_count(5, cc) == 6  # alignment 2
    assert round_up_sm_count(7, cc) == 8


def test_round_up_rejects_non_positive():
    with pytest.raises(VGPURuntimeError):
        round_up_sm_count(0, (9, 0))
    with pytest.raises(VGPURuntimeError):
        round_up_sm_count(-1, (9, 0))


def test_validate_total_passes_within_capacity():
    # 132 SMs (H20Z): 16 + 16 + 16 + 16 = 64 ≤ 132
    validate_total([16, 16, 16, 16], 132)


def test_validate_total_rejects_overflow():
    with pytest.raises(VGPURuntimeError) as ei:
        validate_total([72, 72], 132)
    msg = str(ei.value)
    assert "144" in msg and "132" in msg
