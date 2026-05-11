"""Registry + Policy tests for phyai.layers.linear.

No CUDA / no flashinfer / no dist — all predicates are plain Python.
``FakeKernel`` / ``make_probe`` come from conftest via fixtures.
"""

from __future__ import annotations

import pytest

from phyai.layers.linear.registry import (
    DefaultPolicy,
    ForcedPolicy,
    LinearKernelRegistry,
    _regime_of,
)
from phyai.parallel.exceptions import NoBackendError
from phyai.parallel.state import Mode


# ---------------------------------------------------------------------------
# regime helper
# ---------------------------------------------------------------------------


def test_regime_of_decode_bucket():
    assert _regime_of(0) == "decode"
    assert _regime_of(3) == "decode"
    assert _regime_of(4) == "prefill"
    assert _regime_of(10) == "prefill"


# ---------------------------------------------------------------------------
# Registry ordering
# ---------------------------------------------------------------------------


def test_registry_returns_in_registration_order(fake_kernel, probe):
    a = fake_kernel("a", specs={"bf16"}, modes={Mode.EAGER})
    b = fake_kernel("b", specs={"bf16"}, modes={Mode.EAGER})
    r = LinearKernelRegistry()
    r.register(a)
    r.register(b)
    cands = r.candidates(probe())
    assert [c.name for c in cands] == ["a", "b"]


def test_registry_prefer_for_takes_precedence_per_regime(fake_kernel, probe):
    a = fake_kernel("a", specs={"bf16"}, modes={Mode.EAGER})
    b = fake_kernel("b", specs={"bf16"}, modes={Mode.EAGER})
    r = LinearKernelRegistry()
    r.register(a)
    r.register(b, prefer_for={("bf16", "decode")})

    decode = r.candidates(probe(M_bucket=1))
    assert [c.name for c in decode] == ["b", "a"]

    prefill = r.candidates(probe(M_bucket=10))
    assert [c.name for c in prefill] == ["a", "b"]


def test_registry_filters_capture_unsafe(fake_kernel, probe):
    noncap = fake_kernel(
        "unsafe",
        specs={"bf16"},
        modes={Mode.EAGER, Mode.GRAPH_CAPTURING},
        capture=False,
    )
    cap = fake_kernel(
        "safe",
        specs={"bf16"},
        modes={Mode.EAGER, Mode.GRAPH_CAPTURING},
        capture=True,
    )
    r = LinearKernelRegistry()
    r.register(noncap)
    r.register(cap)

    cands = r.candidates(probe(mode=Mode.GRAPH_CAPTURING))
    assert [c.name for c in cands] == ["safe"]
    eager = r.candidates(probe(mode=Mode.EAGER))
    assert {c.name for c in eager} == {"unsafe", "safe"}


def test_registry_filters_can_handle_false(fake_kernel, probe):
    a = fake_kernel("a", specs={"fp8_per_tensor"}, modes={Mode.EAGER}, min_sm=89)
    b = fake_kernel("b", specs={"fp8_per_tensor"}, modes={Mode.EAGER})
    r = LinearKernelRegistry()
    r.register(a)
    r.register(b)
    cands = r.candidates(probe(spec_id="fp8_per_tensor", sm=80))
    assert [c.name for c in cands] == ["b"]


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


def test_registry_validate_ok_with_single_fallback(fake_kernel):
    fallback = fake_kernel(
        "t",
        specs={"bf16", "fp8_per_tensor"},
        modes={Mode.EAGER, Mode.GRAPH_CAPTURING},
    )
    r = LinearKernelRegistry()
    r.register(fallback)
    r.validate(sample_specs=["bf16", "fp8_per_tensor"], sm=90)


def test_registry_validate_raises_when_no_fallback(fake_kernel):
    narrow = fake_kernel("narrow", specs={"bf16"}, modes={Mode.EAGER})
    r = LinearKernelRegistry()
    r.register(narrow)
    with pytest.raises(NoBackendError):
        r.validate(sample_specs=["bf16", "fp8_per_tensor"], sm=90)


def test_registry_validate_raises_for_unknown_prefer_name(fake_kernel):
    a = fake_kernel(
        "a",
        specs={"bf16"},
        modes={Mode.EAGER, Mode.GRAPH_CAPTURING},
    )
    r = LinearKernelRegistry()
    r.register(a)
    r._prefer[("bf16", "decode")] = ["nonexistent"]
    with pytest.raises(NoBackendError):
        r.validate(sample_specs=["bf16"], sm=90)


def test_registry_validate_skips_capture_when_no_capture_kernel(fake_kernel):
    eager_only = fake_kernel(
        "t",
        specs={"bf16"},
        modes={Mode.EAGER},
        capture=False,
    )
    r = LinearKernelRegistry()
    r.register(eager_only)
    r.validate(sample_specs=["bf16"], sm=90)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_default_policy_picks_first(fake_kernel):
    a = fake_kernel("a", specs=set(), modes=set())
    b = fake_kernel("b", specs=set(), modes=set())
    assert DefaultPolicy().select([a, b]).name == "a"


def test_default_policy_raises_on_empty():
    with pytest.raises(NoBackendError):
        DefaultPolicy().select([])


def test_forced_policy_finds_named_kernel(fake_kernel):
    a = fake_kernel("a", specs=set(), modes=set())
    b = fake_kernel("b", specs=set(), modes=set())
    assert ForcedPolicy("b").select([a, b]).name == "b"


def test_forced_policy_falls_back_when_name_missing(fake_kernel):
    a = fake_kernel("a", specs=set(), modes=set())
    b = fake_kernel("b", specs=set(), modes=set())
    assert ForcedPolicy("nope").select([a, b]).name == "a"
