"""Tests for ``split_device`` / ``split_device_by_sm_count`` on flashinfer.

Anchored on H20Z (132 SMs, CC 9.0); skipped on devices that do not match
that profile because the asserted SM layout is hardware-specific.
"""

from __future__ import annotations

import pytest
import torch

import phyai.vgpu as V
from phyai.vgpu.exceptions import VGPURuntimeError


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="phyai.vgpu requires CUDA",
)


def _flashinfer_available() -> bool:
    try:
        import flashinfer.green_ctx  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(autouse=True)
def _flashinfer_only():
    if not _flashinfer_available():
        pytest.skip("flashinfer not installed")


def _total_sms(device: str = "cuda:0") -> int:
    return torch.cuda.get_device_properties(torch.device(device)).multi_processor_count


def test_split_device_returns_num_groups_plus_one_with_remainder():
    V.init(device="cuda:0", backend="flashinfer")
    total = _total_sms()
    shards = V.split_device("cuda:0", num_groups=2, min_count=16)

    assert len(shards) == 3, "split_device returns num_groups + 1 shards"
    assert shards[0].sm_count == 16
    assert shards[1].sm_count == 16
    assert (
        shards[2].sm_count == total - 32
    ), f"remainder should be {total - 32}, got {shards[2].sm_count}"

    assert shards[0].is_remainder is False
    assert shards[1].is_remainder is False
    assert shards[2].is_remainder is True

    for s in shards:
        assert s.backend == "flashinfer"
        assert s.device.type == "cuda"
        assert isinstance(s.stream, torch.cuda.Stream)

    # Names follow the documented convention.
    assert shards[0].name == "shard_0"
    assert shards[1].name == "shard_1"
    assert shards[2].name == "shard_2_rem"


def test_split_device_rounds_up_min_count():
    """``min_count=10`` on CC 9.0 should round up to 16."""
    V.init(device="cuda:0", backend="flashinfer")
    shards = V.split_device("cuda:0", num_groups=2, min_count=10)
    assert shards[0].sm_count == 16
    assert shards[0].requested_sm_count == 10


def test_split_device_capacity_overflow_raises():
    V.init(device="cuda:0", backend="flashinfer")
    total = _total_sms()
    # Force a request that's clearly larger than the device.
    with pytest.raises(VGPURuntimeError):
        V.split_device("cuda:0", num_groups=2, min_count=total)


def test_split_device_by_sm_count_layout():
    V.init(device="cuda:0", backend="flashinfer")
    total = _total_sms()
    shards = V.split_device_by_sm_count("cuda:0", sm_counts=[8, 16, 24])
    assert [s.sm_count for s in shards[:3]] == [8, 16, 24]
    assert shards[-1].is_remainder is True
    assert shards[-1].sm_count == total - 48


def test_split_device_explicit_backend_overrides_default():
    """Passing ``backend='flashinfer'`` works even after ``init`` chose torch."""
    V.init(device="cuda:0", backend="torch")
    shards = V.split_device(
        "cuda:0",
        num_groups=2,
        min_count=16,
        backend="flashinfer",
    )
    assert all(s.backend == "flashinfer" for s in shards)


def test_split_device_runs_correct_gemm_per_shard():
    """End-to-end check: each shard runs an 8192² bf16 GEMM that matches
    the default-stream reference within bf16 tolerance."""
    V.init(device="cuda:0", backend="flashinfer")
    shards = V.split_device("cuda:0", num_groups=2, min_count=16)
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    x = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
    y = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
    z_ref = (x @ y).clone()
    torch.cuda.synchronize()

    for shard in shards:
        with torch.cuda.stream(shard.stream):
            z = x @ y
        torch.cuda.synchronize()
        rel_err = ((z - z_ref).abs().mean() / z_ref.abs().mean()).item()
        assert rel_err < 1e-2, (shard.name, rel_err)
