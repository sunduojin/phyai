"""world_size=1 short-circuit tests.

When the mesh axis has only one rank, all collectives should be no-ops
without invoking the dispatcher. We verify via the multiprocess harness
(launching 1 worker) and by inspecting the dispatcher's cache state.
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


def _ws1_worker(rank, world_size, port, test_fn, backend, err_queue):
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        if backend == "nccl":
            torch.cuda.set_device(rank)
        try:
            test_fn(rank, world_size)
        finally:
            try:
                dist.destroy_process_group()
            except Exception:
                pass
    except BaseException as e:
        err_queue.put((rank, repr(e), traceback.format_exc()))


def _run_ws1(test_fn, *, backend: str, timeout_s: float = 30.0) -> None:
    if backend == "nccl" and torch.cuda.device_count() < 1:
        pytest.skip("needs CUDA")
    ctx = mp.get_context("spawn")
    err_queue = ctx.Queue()
    port = _free_port()
    p = ctx.Process(
        target=_ws1_worker,
        args=(0, 1, port, test_fn, backend, err_queue),
    )
    p.start()
    p.join(timeout=timeout_s)
    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError("ws=1 worker hung")
    errors = []
    while not err_queue.empty():
        errors.append(err_queue.get_nowait())
    if errors:
        r, e, tb = errors[0]
        raise AssertionError(f"ws=1 worker failed: {e}\n{tb}")


# =============================================================================
# workers
# =============================================================================


def _w_ws1_all_reduce_returns_clone(rank, world_size):
    """ws=1 AR should return a clone of the input (semantically identity)."""
    import phyai.parallel as P

    P.init(layout=(1,), mesh_dim_names=("tp",))
    x = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
    y = P.all_reduce(x, axis="tp")
    assert torch.allclose(y, x), (y, x)
    # Must be a new tensor (so caller mutating y doesn't affect x).
    assert y.data_ptr() != x.data_ptr(), "AR should return a copy at ws=1"


def _w_ws1_dispatcher_not_invoked(rank, world_size):
    """Verify that ws=1 short-circuits BEFORE the dispatcher cache."""
    import phyai.parallel as P

    P.init(layout=(1,), mesh_dim_names=("tp",))
    disp = P.get_dispatcher()
    disp.clear_cache()
    assert len(disp._cache) == 0

    x = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
    _ = P.all_reduce(x, axis="tp")
    _ = P.all_gather(x, axis="tp", dim=0)
    _ = P.broadcast(x, axis="tp", src=0)

    # Cache must still be empty — dispatcher was bypassed.
    assert len(disp._cache) == 0, dict(disp._cache)


def _w_ws1_all_gather_clone(rank, world_size):
    import phyai.parallel as P

    P.init(layout=(1,), mesh_dim_names=("tp",))
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda:0")
    y = P.all_gather(x, axis="tp", dim=0)
    # ws=1 → output shape == input shape
    assert y.shape == x.shape, (y.shape, x.shape)
    assert torch.allclose(y, x), (y, x)


def _w_ws1_reduce_scatter_clone(rank, world_size):
    import phyai.parallel as P

    P.init(layout=(1,), mesh_dim_names=("tp",))
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda:0")
    y = P.reduce_scatter(x, axis="tp", dim=0)
    assert y.shape == x.shape, (y.shape, x.shape)
    assert torch.allclose(y, x), (y, x)


def _w_ws1_gloo(rank, world_size):
    """Same pattern, gloo backend."""
    import phyai.parallel as P

    P.init(layout=(1,), mesh_dim_names=("tp",), device="cpu", backend="gloo")
    disp = P.get_dispatcher()
    disp.clear_cache()
    x = torch.tensor([1.0, 2.0, 3.0])
    y = P.all_reduce(x, axis="tp")
    assert torch.allclose(y, x), (y, x)
    assert len(disp._cache) == 0, "dispatcher should not be invoked at ws=1"


# =============================================================================
# pytest entrypoints
# =============================================================================


def test_ws1_all_reduce_returns_clone() -> None:
    _run_ws1(_w_ws1_all_reduce_returns_clone, backend="nccl")


def test_ws1_dispatcher_not_invoked() -> None:
    _run_ws1(_w_ws1_dispatcher_not_invoked, backend="nccl")


def test_ws1_all_gather_clone() -> None:
    _run_ws1(_w_ws1_all_gather_clone, backend="nccl")


def test_ws1_reduce_scatter_clone() -> None:
    _run_ws1(_w_ws1_reduce_scatter_clone, backend="nccl")


def test_ws1_gloo() -> None:
    _run_ws1(_w_ws1_gloo, backend="gloo")
