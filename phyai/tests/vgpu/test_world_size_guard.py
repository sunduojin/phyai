"""``world_size > 1`` guard test.

Spawns two ranks with a gloo process group (no NCCL infra needed). Each
rank calls :func:`phyai.vgpu.init` and is expected to raise
:class:`VGPUNotApplicableError` carrying the literal phrase
``world_size == 1`` so callers can pattern-match.

We launch the workers via ``subprocess.run`` rather than ``mp.spawn``:
under pytest's ``--import-mode=importlib`` the test module is imported as
``tests.vgpu.test_world_size_guard``, which the spawned subprocess has no
way to re-import. A small inline helper script bypasses that and keeps
the test self-contained.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


_WORKER_SCRIPT = textwrap.dedent(
    """
    import os
    import sys

    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])
    port = sys.argv[3]
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=port,
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )

    import torch.distributed as dist

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        import phyai.vgpu as V

        try:
            V.init(device="cuda:0")
            print("NORAISE")
            sys.exit(2)
        except V.VGPUNotApplicableError as e:
            msg = str(e)
            print(f"RAISED::{msg}")
            sys.exit(0)
        except BaseException as e:  # pragma: no cover
            print(f"WRONGERR::{type(e).__name__}::{e}")
            sys.exit(3)
    finally:
        dist.destroy_process_group()
    """
)


def _free_port() -> int:
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_init_rejects_world_size_gt_1():
    port = _free_port()
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER_SCRIPT, str(rank), "2", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in range(2)
    ]
    outputs = []
    for p in procs:
        try:
            out, err = p.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            pytest.fail("worker hung")
        outputs.append((p.returncode, out, err))

    assert len(outputs) == 2
    for rc, out, err in outputs:
        assert rc == 0, f"worker exited with rc={rc}, stdout={out!r}, stderr={err!r}"
        marker_lines = [
            line for line in out.splitlines() if line.startswith("RAISED::")
        ]
        assert marker_lines, f"missing RAISED marker; got stdout={out!r} stderr={err!r}"
        msg = marker_lines[0][len("RAISED::") :]
        assert "world_size == 1" in msg, msg
        assert "got 2" in msg, msg
