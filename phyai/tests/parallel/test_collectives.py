"""Multi-process correctness tests for collective primitives.

Spawns ``world_size`` workers, each running real NCCL through phyai.parallel,
and asserts numerical correctness in both eager and CUDA-graph modes.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from .multiprocess import run_distributed


# =============================================================================
# eager-mode workers
# =============================================================================


def _w_all_reduce_eager(rank: int, world_size: int) -> None:
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",))

    x = torch.full((8,), float(rank + 1), device=f"cuda:{rank}", dtype=torch.bfloat16)
    y = P.all_reduce(x, axis="tp")
    expected = sum(r + 1 for r in range(world_size))
    assert torch.allclose(
        y.float(),
        torch.full_like(y.float(), float(expected)),
    ), f"got {y.tolist()} expected {expected}"


def _w_all_gather_eager(rank: int, world_size: int) -> None:
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",))

    x = torch.full((4,), float(rank + 1), device=f"cuda:{rank}", dtype=torch.bfloat16)
    y = P.all_gather(x, axis="tp", dim=0)
    assert y.shape == (4 * world_size,), y.shape
    # Verify contiguous chunks per rank
    for r in range(world_size):
        chunk = y[r * 4 : (r + 1) * 4]
        assert torch.allclose(chunk.float(), torch.full((4,), float(r + 1))), chunk


def _w_reduce_scatter_eager(rank: int, world_size: int) -> None:
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",))

    # All ranks: x = arange(world_size * 4) * (rank + 1).
    x = torch.arange(
        world_size * 4, device=f"cuda:{rank}", dtype=torch.float32
    ) * float(rank + 1)
    y = P.reduce_scatter(x, axis="tp", dim=0)
    assert y.shape == (4,), y.shape
    # Each rank gets sum_r (chunk_r * (r+1)) for its segment.
    base = torch.arange(rank * 4, (rank + 1) * 4, dtype=torch.float32)
    expected = base * sum(r + 1 for r in range(world_size))
    assert torch.allclose(y.cpu(), expected), (y.cpu(), expected)


def _w_broadcast_eager(rank: int, world_size: int) -> None:
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",))

    src_value = 42.0
    x = torch.full((6,), float(rank + 1), device=f"cuda:{rank}", dtype=torch.float32)
    if rank == 0:
        x.fill_(src_value)
    y = P.broadcast(x, axis="tp", src=0)
    assert torch.allclose(y, torch.full_like(y, src_value)), y


def _w_send_recv_eager(rank: int, world_size: int) -> None:
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",))
    assert world_size == 2

    if rank == 0:
        x = torch.arange(16, device="cuda:0", dtype=torch.float32)
        P.send(x, axis="tp", dst=1)
    else:
        y = P.recv(
            (16,), torch.float32, axis="tp", src=0, device=torch.device("cuda:1")
        )
        assert torch.allclose(y.cpu(), torch.arange(16, dtype=torch.float32))


def _w_all_to_all_eager(rank: int, world_size: int) -> None:
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",))

    # Each rank sends a contiguous chunk of size W to each peer; each rank
    # receives W*W total elements arranged as the gather-along-rank order.
    W = 4
    # x[r] = our rank * 100 + r (so receiver r ends up with all senders' r-th)
    x = torch.cat(
        [
            torch.full((W,), float(rank * 100 + r), dtype=torch.float32)
            for r in range(world_size)
        ]
    ).to(f"cuda:{rank}")
    y = P.all_to_all(x, axis="tp")
    # On rank `r`, we receive sender s's r-th chunk, value = s*100 + r
    expected = torch.cat(
        [
            torch.full((W,), float(s * 100 + rank), dtype=torch.float32)
            for s in range(world_size)
        ]
    )
    assert torch.allclose(y.cpu(), expected), (y.cpu(), expected)


# =============================================================================
# graph-capture workers
# =============================================================================


def _w_all_reduce_graph(rank: int, world_size: int) -> None:
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",))

    static = torch.full(
        (8,), float(rank + 1), device=f"cuda:{rank}", dtype=torch.bfloat16
    )

    # Warmup pass on a side stream
    P.warmup(P.all_reduce, static, axis="tp")

    g = torch.cuda.CUDAGraph()
    with P.graph_capture(), torch.cuda.graph(g):
        out = P.all_reduce(static, axis="tp")
    torch.cuda.synchronize()

    # Replay
    static.fill_(float(rank + 1))
    g.replay()
    torch.cuda.synchronize()
    expected = sum(r + 1 for r in range(world_size))
    assert torch.allclose(
        out.float(), torch.full_like(out.float(), float(expected))
    ), f"got {out.tolist()} expected {expected}"


def _w_all_gather_graph(rank: int, world_size: int) -> None:
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",))

    static = torch.full(
        (4,), float(rank + 1), device=f"cuda:{rank}", dtype=torch.bfloat16
    )
    P.warmup(P.all_gather, static, axis="tp", dim=0)

    g = torch.cuda.CUDAGraph()
    with P.graph_capture(), torch.cuda.graph(g):
        out = P.all_gather(static, axis="tp", dim=0)
    torch.cuda.synchronize()

    static.fill_(float(rank + 1))
    g.replay()
    torch.cuda.synchronize()
    for r in range(world_size):
        chunk = out[r * 4 : (r + 1) * 4]
        assert torch.allclose(chunk.float(), torch.full((4,), float(r + 1))), chunk


def _w_reduce_scatter_graph(rank: int, world_size: int) -> None:
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",))

    # Each rank's input: a vector of length world_size*4, all (rank+1).
    # After reduce-scatter (sum along dim 0, scatter chunks of 4):
    # rank r sees a chunk of 4 = sum_r (r+1) = constant.
    static = torch.full(
        (world_size * 4,),
        float(rank + 1),
        device=f"cuda:{rank}",
        dtype=torch.float32,
    )
    P.warmup(P.reduce_scatter, static, axis="tp", dim=0)

    g = torch.cuda.CUDAGraph()
    with P.graph_capture(), torch.cuda.graph(g):
        out = P.reduce_scatter(static, axis="tp", dim=0)
    torch.cuda.synchronize()

    static.fill_(float(rank + 1))
    g.replay()
    torch.cuda.synchronize()
    expected = float(sum(r + 1 for r in range(world_size)))
    assert out.shape == (4,), out.shape
    assert torch.allclose(
        out, torch.full_like(out, expected)
    ), f"got {out.tolist()} expected {expected}"


def _w_broadcast_graph(rank: int, world_size: int) -> None:
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",))

    src_value = 7.0
    static = torch.full(
        (6,), float(rank + 1), device=f"cuda:{rank}", dtype=torch.float32
    )
    if rank == 0:
        static.fill_(src_value)
    # Warmup: rank 0 broadcasts; on warmup we want all ranks to participate
    P.warmup(P.broadcast, static, axis="tp", src=0)

    # Reset before capture so warmup didn't permanently change state on
    # non-zero ranks.
    if rank == 0:
        static.fill_(src_value)
    else:
        static.fill_(float(rank + 1))

    g = torch.cuda.CUDAGraph()
    with P.graph_capture(), torch.cuda.graph(g):
        out = P.broadcast(static, axis="tp", src=0)
    torch.cuda.synchronize()

    if rank == 0:
        static.fill_(src_value)
    else:
        static.fill_(float(rank + 1))
    g.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out, torch.full_like(out, src_value)), out


def _w_send_recv_graph(rank: int, world_size: int) -> None:
    """Graph-capture send/recv. Both ranks capture their respective op;
    on replay they pair through the PyNCCL communicator just like in
    eager mode."""
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",))
    assert world_size == 2

    if rank == 0:
        static_in = torch.arange(16, device="cuda:0", dtype=torch.float32)
        # Warmup pair (ranks must call together)
        P.warmup(P.send, static_in, axis="tp", dst=1)
    else:
        static_recv = torch.empty(16, device="cuda:1", dtype=torch.float32)
        # We will explicitly use PyNCCL through a recv into a pre-allocated
        # tensor, by calling recv with an "out" argument. ``recv`` allocates
        # internally — for graph capture we need stable pointers. Use the
        # warmup pass to wire up the comm.
        P.warmup(
            lambda: P.recv(
                (16,), torch.float32, axis="tp", src=0, device=torch.device("cuda:1")
            )
        )

    g = torch.cuda.CUDAGraph()
    with P.graph_capture(), torch.cuda.graph(g):
        if rank == 0:
            P.send(static_in, axis="tp", dst=1)
        else:
            static_recv = P.recv(
                (16,),
                torch.float32,
                axis="tp",
                src=0,
                device=torch.device("cuda:1"),
            )
    torch.cuda.synchronize()

    # Replay
    if rank == 0:
        static_in.copy_(torch.arange(16, dtype=torch.float32).cuda())
    g.replay()
    torch.cuda.synchronize()

    if rank == 1:
        assert torch.allclose(
            static_recv.cpu(),
            torch.arange(16, dtype=torch.float32),
        ), static_recv


def _w_all_to_all_graph(rank: int, world_size: int) -> None:
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",))

    W = 4
    # Same payload pattern as eager test
    static = torch.cat(
        [
            torch.full((W,), float(rank * 100 + r), dtype=torch.float32)
            for r in range(world_size)
        ]
    ).to(f"cuda:{rank}")
    P.warmup(P.all_to_all, static, axis="tp")

    g = torch.cuda.CUDAGraph()
    with P.graph_capture(), torch.cuda.graph(g):
        out = P.all_to_all(static, axis="tp")
    torch.cuda.synchronize()

    static.copy_(
        torch.cat(
            [
                torch.full((W,), float(rank * 100 + r), dtype=torch.float32)
                for r in range(world_size)
            ]
        ).to(f"cuda:{rank}")
    )
    g.replay()
    torch.cuda.synchronize()
    expected = torch.cat(
        [
            torch.full((W,), float(s * 100 + rank), dtype=torch.float32)
            for s in range(world_size)
        ]
    )
    assert torch.allclose(out.cpu(), expected), (out.cpu(), expected)


# =============================================================================
# pytest test cases (drive the workers)
# =============================================================================


def _w_dispatch_routing(rank: int, world_size: int) -> None:
    """Verify Dispatcher actually routes through PyNCCL in graph mode and
    TorchDist in eager mode (not just that numerical results match)."""
    import phyai.parallel as P
    from phyai.parallel.backend import Op
    from phyai.parallel.state import Mode

    P.init(layout=(world_size,), mesh_dim_names=("tp",))

    x = torch.full((8,), 1.0, device=f"cuda:{rank}", dtype=torch.bfloat16)

    # Eager: TorchDist should win (PyNCCL.can_handle returns False in eager)
    eager_choice = P.get_dispatcher().select(
        op=Op.ALL_REDUCE,
        mesh=P.default_mesh(),
        axis="tp",
        tensor=x,
    )
    assert eager_choice.name == "nccl", eager_choice.name

    # Capturing: PyNCCL is preferred via prefer_for + supports_capture
    with P.graph_capture():
        graph_choice = P.get_dispatcher().select(
            op=Op.ALL_REDUCE,
            mesh=P.default_mesh(),
            axis="tp",
            tensor=x,
        )
    assert graph_choice.name == "pynccl", graph_choice.name


@pytest.mark.parametrize("world_size", [2, 4])
def test_all_reduce_eager(world_size: int) -> None:
    run_distributed(_w_all_reduce_eager, world_size=world_size)


@pytest.mark.parametrize("world_size", [2, 4])
def test_all_gather_eager(world_size: int) -> None:
    run_distributed(_w_all_gather_eager, world_size=world_size)


@pytest.mark.parametrize("world_size", [2, 4])
def test_reduce_scatter_eager(world_size: int) -> None:
    run_distributed(_w_reduce_scatter_eager, world_size=world_size)


@pytest.mark.parametrize("world_size", [2, 4])
def test_broadcast_eager(world_size: int) -> None:
    run_distributed(_w_broadcast_eager, world_size=world_size)


def test_send_recv_eager() -> None:
    run_distributed(_w_send_recv_eager, world_size=2)


@pytest.mark.parametrize("world_size", [2, 4])
def test_all_to_all_eager(world_size: int) -> None:
    run_distributed(_w_all_to_all_eager, world_size=world_size)


@pytest.mark.parametrize("world_size", [2, 4])
def test_all_reduce_graph(world_size: int) -> None:
    run_distributed(_w_all_reduce_graph, world_size=world_size, timeout_s=120.0)


@pytest.mark.parametrize("world_size", [2, 4])
def test_all_gather_graph(world_size: int) -> None:
    run_distributed(_w_all_gather_graph, world_size=world_size, timeout_s=120.0)


@pytest.mark.parametrize("world_size", [2, 4])
def test_reduce_scatter_graph(world_size: int) -> None:
    run_distributed(_w_reduce_scatter_graph, world_size=world_size, timeout_s=120.0)


@pytest.mark.parametrize("world_size", [2, 4])
def test_broadcast_graph(world_size: int) -> None:
    run_distributed(_w_broadcast_graph, world_size=world_size, timeout_s=120.0)


def test_send_recv_graph() -> None:
    run_distributed(_w_send_recv_graph, world_size=2, timeout_s=120.0)


@pytest.mark.parametrize("world_size", [2, 4])
def test_all_to_all_graph(world_size: int) -> None:
    run_distributed(_w_all_to_all_graph, world_size=world_size, timeout_s=120.0)


def test_dispatch_routing() -> None:
    run_distributed(_w_dispatch_routing, world_size=2)
