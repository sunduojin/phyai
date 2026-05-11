"""KernelDispatcher tests — cache behaviour, M-bucket, force env.

``FakeKernel`` / ``make_probe`` come from conftest via fixtures.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from phyai.layers.linear.dispatch import (
    KernelDispatcher,
    _M_bucket,
    get_linear_dispatcher,
)
from phyai.layers.linear.registry import LinearKernelRegistry
from phyai.parallel.exceptions import NoBackendError
from phyai.parallel.state import Mode


def _reg_with(*kernels) -> LinearKernelRegistry:
    r = LinearKernelRegistry()
    for k in kernels:
        r.register(k)
    return r


# ---------------------------------------------------------------------------
# M_bucket
# ---------------------------------------------------------------------------


def test_M_bucket_monotonic_with_bit_length():
    assert _M_bucket(0) == 0
    assert _M_bucket(1) == 1
    assert _M_bucket(4) == 3
    assert _M_bucket(8) == 4
    assert _M_bucket(1024) == 11


# ---------------------------------------------------------------------------
# select / cache
# ---------------------------------------------------------------------------


def test_dispatcher_picks_first_candidate(fake_kernel):
    a = fake_kernel("a", specs={"bf16"}, modes={Mode.EAGER})
    b = fake_kernel("b", specs={"bf16"}, modes={Mode.EAGER})
    d = KernelDispatcher(_reg_with(a, b))
    chosen = d.select(
        spec_id="bf16",
        M=4,
        N=512,
        K=512,
        in_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    assert chosen.name == "a"


def test_dispatcher_caches_decision(fake_kernel):
    a = fake_kernel("a", specs={"bf16"}, modes={Mode.EAGER})
    r = _reg_with(a)
    d = KernelDispatcher(r)

    spy = patch.object(r, "candidates", wraps=r.candidates).start()
    try:
        for _ in range(5):
            d.select(
                spec_id="bf16",
                M=8,
                N=512,
                K=512,
                in_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
            )
        assert spy.call_count == 1
    finally:
        patch.stopall()


def test_dispatcher_separates_decode_vs_prefill(fake_kernel):
    decode_only = fake_kernel("decode", specs={"bf16"}, modes={Mode.EAGER})
    prefill_only = fake_kernel("prefill", specs={"bf16"}, modes={Mode.EAGER})
    r = LinearKernelRegistry()
    r.register(decode_only, prefer_for={("bf16", "decode")})
    r.register(prefill_only, prefer_for={("bf16", "prefill")})
    d = KernelDispatcher(r)

    k1 = d.select(
        spec_id="bf16",
        M=1,
        N=512,
        K=512,
        in_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    k2 = d.select(
        spec_id="bf16",
        M=1024,
        N=512,
        K=512,
        in_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    assert k1.name == "decode"
    assert k2.name == "prefill"


def test_dispatcher_cache_key_includes_shapes_and_dtype(fake_kernel):
    a = fake_kernel("a", specs={"bf16"}, modes={Mode.EAGER})
    r = _reg_with(a)
    d = KernelDispatcher(r)

    spy = patch.object(r, "candidates", wraps=r.candidates).start()
    try:
        d.select(
            spec_id="bf16",
            M=8,
            N=512,
            K=512,
            in_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
        )
        d.select(
            spec_id="bf16",
            M=8,
            N=1024,
            K=512,
            in_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
        )
        d.select(
            spec_id="bf16",
            M=8,
            N=512,
            K=512,
            in_dtype=torch.float16,
            out_dtype=torch.bfloat16,
        )
        assert spy.call_count == 3
    finally:
        patch.stopall()


def test_dispatcher_clear_cache(fake_kernel):
    a = fake_kernel("a", specs={"bf16"}, modes={Mode.EAGER})
    r = _reg_with(a)
    d = KernelDispatcher(r)
    d.select(
        spec_id="bf16",
        M=8,
        N=512,
        K=512,
        in_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    assert d._cache
    d.clear_cache()
    assert not d._cache


def test_dispatcher_raises_when_no_backend(fake_kernel):
    a = fake_kernel("a", specs={"bf16"}, modes={Mode.EAGER})
    d = KernelDispatcher(_reg_with(a))
    with pytest.raises(NoBackendError):
        d.select(
            spec_id="fp8_per_tensor",
            M=8,
            N=512,
            K=512,
            in_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
        )


def test_dispatcher_force_env(fake_kernel, monkeypatch):
    monkeypatch.setenv("PHYAI_FORCE_LINEAR_KERNEL", "b")
    a = fake_kernel("a", specs={"bf16"}, modes={Mode.EAGER})
    b = fake_kernel("b", specs={"bf16"}, modes={Mode.EAGER})
    d = KernelDispatcher(_reg_with(a, b))
    chosen = d.select(
        spec_id="bf16",
        M=8,
        N=512,
        K=512,
        in_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    assert chosen.name == "b"


def test_get_linear_dispatcher_raises_before_init():
    from phyai.layers.linear import _reset_for_test

    _reset_for_test()
    with pytest.raises(RuntimeError, match="init.*not called"):
        get_linear_dispatcher()
