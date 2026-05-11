"""Multiprocess test helper.

Each test that needs real NCCL spawns N worker processes via
``torch.multiprocessing.spawn``; each worker runs ``init_process_group``
+ the test body and asserts on its own.

Failures inside the workers raise to the parent and pytest prints them.
"""

from __future__ import annotations

import os
import socket
import traceback
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker(
    rank: int,
    world_size: int,
    port: int,
    test_fn: Callable[..., None],
    args: tuple,
    err_queue: "mp.Queue[Any]",
) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)

        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        try:
            test_fn(rank, world_size, *args)
        finally:
            try:
                dist.destroy_process_group()
            except Exception:
                pass
    except BaseException as e:
        err_queue.put((rank, repr(e), traceback.format_exc()))


def run_distributed(
    test_fn: Callable[..., None],
    *,
    world_size: int,
    args: tuple = (),
    timeout_s: float = 60.0,
) -> None:
    """Spawn ``world_size`` worker processes and run ``test_fn`` on each.

    ``test_fn(rank, world_size, *args)`` is called in each worker. It must
    raise on failure; any raised exception is collected and re-raised here.

    Skips the test if not enough GPUs are available.
    """
    import pytest

    if torch.cuda.device_count() < world_size:
        pytest.skip(f"need {world_size} GPUs, have {torch.cuda.device_count()}")

    ctx = mp.get_context("spawn")
    err_queue: "mp.Queue[Any]" = ctx.Queue()
    port = _free_port()
    procs: list[mp.Process] = []
    for r in range(world_size):
        p = ctx.Process(
            target=_worker,
            args=(r, world_size, port, test_fn, args, err_queue),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=timeout_s)
        if p.is_alive():
            p.terminate()
            p.join()
            raise TimeoutError(f"worker {p.pid} did not finish in {timeout_s}s")

    errors: list[tuple[int, str, str]] = []
    while not err_queue.empty():
        errors.append(err_queue.get_nowait())

    if errors:
        msg = "\n".join(
            f"--- worker rank={r} failed: {e} ---\n{tb}" for r, e, tb in errors
        )
        raise AssertionError(f"distributed test failed:\n{msg}")
