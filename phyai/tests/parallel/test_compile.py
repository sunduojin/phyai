"""torch.compile integration tests.

Validates that phyai's custom ops are opaque to Dynamo (no graph break)
and survive AOTAutograd functionalization + Inductor codegen.

Coverage:
  - ``backend="eager"``     — Dynamo only, no AOT, no codegen
  - ``backend="aot_eager"`` — Dynamo + AOTAutograd functionalization
  - ``backend="inductor"``  — full pipeline (Dynamo + AOT + Triton codegen)
  - ``torch._dynamo.explain`` — confirms no graph break around our ops
"""

from __future__ import annotations

import pytest
import torch

from .multiprocess import run_distributed


def _make_compiled_block(*, backend: str, axis: str = "tp"):
    import phyai.parallel as P

    @torch.compile(fullgraph=True, backend=backend, dynamic=False)
    def block(x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.relu(x)
        h = P.all_reduce(h, axis=axis)
        return h * 2.0

    return block


# =============================================================================
#  Single-rank tests (world_size=1) — exercise Dynamo + custom-op tracing
#  without requiring real NCCL.
# =============================================================================


def _w_compile_eager_no_break(rank: int, world_size: int) -> None:
    """Verify Dynamo accepts our custom op without graph break in eager
    backend.  AR over a 1-rank group is the identity, so we can also check
    numerical correctness."""
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",))
    block = _make_compiled_block(backend="eager")

    x = torch.tensor([-1.0, 0.5, 2.0, -3.0], device=f"cuda:{rank}", dtype=torch.float32)
    y = block(x)
    expected = torch.relu(x) * 2.0
    assert torch.allclose(y, expected), (y, expected)


def _w_compile_aot_eager(rank: int, world_size: int) -> None:
    """``aot_eager`` runs Dynamo + AOTAutograd's functionalization but no
    Inductor codegen — exercises the FX export path that custom ops feed."""
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",))
    block = _make_compiled_block(backend="aot_eager")

    x = torch.tensor([-1.0, 0.5, 2.0, -3.0], device=f"cuda:{rank}", dtype=torch.float32)
    y = block(x)
    expected = torch.relu(x) * 2.0
    assert torch.allclose(y, expected), (y, expected)


def _w_compile_inductor_single_rank(rank: int, world_size: int) -> None:
    """Full Dynamo + AOTAutograd + Inductor (Triton codegen). Custom op
    must remain opaque so Inductor doesn't try to fuse it; the surrounding
    pointwise ops should be fused into a Triton kernel."""
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",))
    block = _make_compiled_block(backend="inductor")

    x = torch.tensor([-1.0, 0.5, 2.0, -3.0], device=f"cuda:{rank}", dtype=torch.float32)
    y = block(x)
    expected = torch.relu(x) * 2.0
    assert torch.allclose(y, expected), (y, expected)


def _w_compile_explain_no_break(rank: int, world_size: int) -> None:
    """Use torch._dynamo.explain to confirm the custom op shows up as an
    opaque node and there is no graph break."""
    import phyai.parallel as P
    import torch._dynamo as dynamo

    P.init(layout=(world_size,), mesh_dim_names=("tp",))

    def block(x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.relu(x)
        h = P.all_reduce(h, axis="tp")
        return h * 2.0

    x = torch.tensor([-1.0, 0.5, 2.0, -3.0], device=f"cuda:{rank}", dtype=torch.float32)
    explanation = dynamo.explain(block)(x)
    # graph_count == 1 means a single FX graph (no breaks).
    assert explanation.graph_count == 1, (
        f"unexpected graph break(s): graph_count={explanation.graph_count}, "
        f"break_reasons={explanation.break_reasons}"
    )
    assert explanation.graph_break_count == 0, explanation.break_reasons


def _w_compile_multi_op_no_break(rank: int, world_size: int) -> None:
    """A function chaining multiple phyai collectives must remain a single
    FX graph (each custom op is opaque, but the surrounding ops should
    fuse normally)."""
    import phyai.parallel as P
    import torch._dynamo as dynamo

    P.init(layout=(world_size,), mesh_dim_names=("tp",))

    def block(x: torch.Tensor) -> torch.Tensor:
        h = P.all_gather(x, axis="tp", dim=0)
        h = h + 1.0
        h = P.all_reduce(h, axis="tp")
        return h * 0.5

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=f"cuda:{rank}", dtype=torch.float32)
    explanation = dynamo.explain(block)(x)
    assert (
        explanation.graph_count == 1
    ), f"got {explanation.graph_count} graphs, breaks: {explanation.break_reasons}"


# =============================================================================
#  Multi-rank torch.compile correctness — both eager and inductor backends.
# =============================================================================


def _w_compile_eager_multi_rank(rank: int, world_size: int) -> None:
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",))

    @torch.compile(fullgraph=True, backend="eager")
    def block(x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.relu(x)
        return P.all_reduce(h, axis="tp") * 0.5

    x = torch.full(
        (8,), float(rank + 1) - 0.5, device=f"cuda:{rank}", dtype=torch.float32
    )
    y = block(x)
    expected = sum(max(0.0, float(r + 1) - 0.5) for r in range(world_size)) * 0.5
    assert torch.allclose(y, torch.full_like(y, expected)), (y, expected)


def _w_compile_inductor_multi_rank(rank: int, world_size: int) -> None:
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",))

    @torch.compile(fullgraph=True, backend="inductor")
    def block(x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.relu(x)
        return P.all_reduce(h, axis="tp") * 0.5

    x = torch.full(
        (8,), float(rank + 1) - 0.5, device=f"cuda:{rank}", dtype=torch.float32
    )
    y = block(x)
    expected = sum(max(0.0, float(r + 1) - 0.5) for r in range(world_size)) * 0.5
    assert torch.allclose(y, torch.full_like(y, expected)), (y, expected)


# =============================================================================
#  pytest entrypoints
# =============================================================================


def test_compile_eager_single_rank() -> None:
    run_distributed(_w_compile_eager_no_break, world_size=1)


def test_compile_aot_eager_single_rank() -> None:
    run_distributed(_w_compile_aot_eager, world_size=1)


def test_compile_inductor_single_rank() -> None:
    run_distributed(_w_compile_inductor_single_rank, world_size=1, timeout_s=180.0)


def test_compile_explain_no_graph_break() -> None:
    run_distributed(_w_compile_explain_no_break, world_size=1)


def test_compile_multi_op_no_graph_break() -> None:
    run_distributed(_w_compile_multi_op_no_break, world_size=1)


@pytest.mark.parametrize("world_size", [2, 4])
def test_compile_eager_multi_rank(world_size: int) -> None:
    run_distributed(_w_compile_eager_multi_rank, world_size=world_size)


@pytest.mark.parametrize("world_size", [2, 4])
def test_compile_inductor_multi_rank(world_size: int) -> None:
    run_distributed(
        _w_compile_inductor_multi_rank, world_size=world_size, timeout_s=180.0
    )
