"""Gloo (CPU) backend correctness tests.

Spawns workers using the gloo backend on CPU tensors and asserts the
collectives produce the right values. Mirrors a subset of the NCCL tests
in ``test_collectives.py``.
"""

from __future__ import annotations

import os
import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _gloo_worker(
    rank: int,
    world_size: int,
    port: int,
    test_fn,
    args: tuple,
    err_queue,
) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)

        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        try:
            test_fn(rank, world_size, *args)
        finally:
            try:
                dist.destroy_process_group()
            except Exception:
                pass
    except BaseException as e:
        err_queue.put((rank, repr(e), traceback.format_exc()))


def _run_gloo(
    test_fn, *, world_size: int, args: tuple = (), timeout_s: float = 60.0
) -> None:
    """Spawn workers backed by gloo. CPU-only, no CUDA needed."""
    ctx = mp.get_context("spawn")
    err_queue = ctx.Queue()
    port = _free_port()
    procs = []
    for r in range(world_size):
        p = ctx.Process(
            target=_gloo_worker,
            args=(r, world_size, port, test_fn, args, err_queue),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=timeout_s)
        if p.is_alive():
            p.terminate()
            p.join()
            raise TimeoutError(f"gloo worker {p.pid} did not finish")
    errors = []
    while not err_queue.empty():
        errors.append(err_queue.get_nowait())
    if errors:
        msg = "\n".join(
            f"--- gloo worker rank={r} failed: {e} ---\n{tb}" for r, e, tb in errors
        )
        raise AssertionError(f"gloo distributed test failed:\n{msg}")


# =============================================================================
# workers
# =============================================================================


def _w_gloo_init_auto_detects(rank: int, world_size: int) -> None:
    """No explicit backend= → auto detect from world PG."""
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",))  # auto → "gloo"
    backends = [b.name for b in P.get_dispatcher().registry.all()]
    assert backends == ["gloo"], backends


def _w_gloo_all_reduce(rank: int, world_size: int) -> None:
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",), device="cpu", backend="gloo")
    x = torch.full((8,), float(rank + 1), dtype=torch.float32)
    y = P.all_reduce(x, axis="tp")
    expected = sum(r + 1 for r in range(world_size))
    assert torch.allclose(y, torch.full_like(y, float(expected))), (y, expected)


def _w_gloo_all_gather(rank: int, world_size: int) -> None:
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",), device="cpu", backend="gloo")
    x = torch.full((4,), float(rank + 1), dtype=torch.float32)
    y = P.all_gather(x, axis="tp", dim=0)
    assert y.shape == (4 * world_size,), y.shape
    for r in range(world_size):
        chunk = y[r * 4 : (r + 1) * 4]
        assert torch.allclose(chunk, torch.full((4,), float(r + 1))), chunk


def _w_gloo_reduce_scatter(rank: int, world_size: int) -> None:
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",), device="cpu", backend="gloo")
    x = torch.full((world_size * 4,), float(rank + 1), dtype=torch.float32)
    y = P.reduce_scatter(x, axis="tp", dim=0)
    expected = float(sum(r + 1 for r in range(world_size)))
    assert y.shape == (4,), y.shape
    assert torch.allclose(y, torch.full_like(y, expected)), (y, expected)


def _w_gloo_broadcast(rank: int, world_size: int) -> None:
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",), device="cpu", backend="gloo")
    src_value = 9.0
    x = torch.full((6,), float(rank + 1), dtype=torch.float32)
    if rank == 0:
        x.fill_(src_value)
    y = P.broadcast(x, axis="tp", src=0)
    assert torch.allclose(y, torch.full_like(y, src_value)), y


def _w_gloo_send_recv(rank: int, world_size: int) -> None:
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",), device="cpu", backend="gloo")
    assert world_size == 2
    if rank == 0:
        x = torch.arange(16, dtype=torch.float32)
        P.send(x, axis="tp", dst=1)
    else:
        y = P.recv((16,), torch.float32, axis="tp", src=0, device=torch.device("cpu"))
        assert torch.allclose(y, torch.arange(16, dtype=torch.float32)), y


def _w_gloo_all_to_all(rank: int, world_size: int) -> None:
    import phyai.parallel as P

    P.init(layout=(world_size,), mesh_dim_names=("tp",), device="cpu", backend="gloo")
    W = 4
    x = torch.cat(
        [
            torch.full((W,), float(rank * 100 + r), dtype=torch.float32)
            for r in range(world_size)
        ]
    )
    y = P.all_to_all(x, axis="tp")
    expected = torch.cat(
        [
            torch.full((W,), float(s * 100 + rank), dtype=torch.float32)
            for s in range(world_size)
        ]
    )
    assert torch.allclose(y, expected), (y, expected)


def _w_gloo_graph_capture_rejected(rank: int, world_size: int) -> None:
    """``Mode.GRAPH_CAPTURING`` has no gloo support; dispatcher should
    raise ``NoBackendError`` rather than silently choose gloo (which can't
    be captured)."""
    import phyai.parallel as P
    from phyai.parallel.exceptions import NoBackendError

    P.init(layout=(world_size,), mesh_dim_names=("tp",), device="cpu", backend="gloo")

    x = torch.full((8,), float(rank + 1), dtype=torch.float32)
    with P.graph_capture():
        try:
            P.all_reduce(x, axis="tp")
            raise AssertionError("expected NoBackendError under capture")
        except NoBackendError:
            pass  # expected


# =============================================================================
# pytest entrypoints
# =============================================================================


@pytest.mark.parametrize("world_size", [2, 4])
def test_gloo_init_auto_detects(world_size: int) -> None:
    _run_gloo(_w_gloo_init_auto_detects, world_size=world_size)


@pytest.mark.parametrize("world_size", [2, 4])
def test_gloo_all_reduce(world_size: int) -> None:
    _run_gloo(_w_gloo_all_reduce, world_size=world_size)


@pytest.mark.parametrize("world_size", [2, 4])
def test_gloo_all_gather(world_size: int) -> None:
    _run_gloo(_w_gloo_all_gather, world_size=world_size)


@pytest.mark.parametrize("world_size", [2, 4])
def test_gloo_reduce_scatter(world_size: int) -> None:
    _run_gloo(_w_gloo_reduce_scatter, world_size=world_size)


@pytest.mark.parametrize("world_size", [2, 4])
def test_gloo_broadcast(world_size: int) -> None:
    _run_gloo(_w_gloo_broadcast, world_size=world_size)


def test_gloo_send_recv() -> None:
    _run_gloo(_w_gloo_send_recv, world_size=2)


@pytest.mark.parametrize("world_size", [2, 4])
def test_gloo_all_to_all(world_size: int) -> None:
    _run_gloo(_w_gloo_all_to_all, world_size=world_size)


@pytest.mark.parametrize("world_size", [2])
def test_gloo_graph_capture_rejected(world_size: int) -> None:
    _run_gloo(_w_gloo_graph_capture_rejected, world_size=world_size)
