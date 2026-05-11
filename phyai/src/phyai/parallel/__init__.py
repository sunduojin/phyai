"""phyai.parallel — distributed primitives by named axis.

Quick start:

    import torch.distributed as dist
    import phyai.parallel as P

    dist.init_process_group("nccl")
    P.init(layout=(8,), mesh_dim_names=("tp",))   # default mesh, NCCL backend

    # In model code:
    y = P.all_reduce(x, axis="tp")

CPU / gloo:

    dist.init_process_group("gloo")
    P.init(layout=(2,), mesh_dim_names=("tp",), device="cpu", backend="gloo")
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from phyai.parallel.backend import Backend, Op, Topology
from phyai.parallel.dispatch import Dispatcher, get_dispatcher, set_dispatcher
from phyai.parallel.exceptions import (
    CaptureUnsafeError,
    CommTimeoutError,
    NoBackendError,
    PhyaiDistError,
)
from phyai.parallel.mesh import Mesh
from phyai.parallel.ops import (
    all_gather,
    all_reduce,
    all_to_all,
    barrier,
    broadcast,
    recv,
    reduce_scatter,
    send,
)
from phyai.parallel.registry import DefaultPolicy, ForcedPolicy, Policy, Registry
from phyai.parallel.state import (
    Mode,
    current_mode,
    default_mesh,
    graph_capture,
    register_mesh,
    use_mesh,
)
from phyai.parallel.backends import (
    GlooBackend,
    NcclBackend,
    PyNCCLBackend,
    TorchDistBackend,  # alias for NcclBackend
)


def _resolve_backend(backend: str | None) -> str:
    """Resolve a ``backend`` choice ('auto' / None / explicit) into a
    concrete name by inspecting the world process group."""
    if backend in (None, "auto"):
        wb = dist.get_backend()
        if wb in ("nccl", "gloo"):
            return wb
        # Unknown / non-standard backend (e.g. mpi, custom) — return as-is;
        # the caller should explicitly opt in if they want phyai support.
        return wb
    return backend


def init(
    *,
    layout: tuple[int, ...] | list[int],
    mesh_dim_names: tuple[str, ...],
    device: str | torch.device | None = None,
    backend: str | None = None,
    enable_pynccl: bool = True,
    pynccl_axes: list[str] | None = None,
    pynccl_library_path: str | None = None,
) -> Mesh:
    """Initialise phyai.parallel.

    Must be called collectively on all ranks AFTER
    ``torch.distributed.init_process_group`` has returned.

    Args:
        layout: mesh shape, e.g. ``(8,)`` for TP=8 or ``(2, 4)`` for
            (DP=2, TP=4). Product must equal ``dist.get_world_size()``.
        mesh_dim_names: names for each axis, e.g. ``("tp",)``.
        device: device type the mesh will run on. If None, derived from
            ``backend`` (``"cuda"`` for nccl, ``"cpu"`` for gloo).
        backend: which torch.distributed backend the registered phyai
            backend(s) should target. One of ``"nccl"``, ``"gloo"``, or
            ``None``/``"auto"`` to auto-detect from the world PG.
        enable_pynccl: register the PyNCCL backend in addition to
            ``NcclBackend``. PyNCCL is preferred under graph capture.
            Automatically disabled when ``backend="gloo"``.
        pynccl_axes: which axes to build PyNCCL communicators for. If
            None, defaults to all ``mesh_dim_names``.
        pynccl_library_path: override path to libnccl.so.

    Returns:
        The default :class:`Mesh`.
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "phyai.parallel.init: torch.distributed must be initialised first "
            "(call torch.distributed.init_process_group(...))"
        )

    expected = 1
    for s in layout:
        expected *= s
    if expected != dist.get_world_size():
        raise ValueError(
            f"layout product ({expected}) != world_size ({dist.get_world_size()})"
        )

    resolved_backend = _resolve_backend(backend)

    if device is None:
        device = "cpu" if resolved_backend == "gloo" else "cuda"
    device_type = str(torch.device(device).type)

    torch_mesh = init_device_mesh(
        device_type,
        tuple(layout),
        mesh_dim_names=tuple(mesh_dim_names),
    )
    mesh = Mesh(torch_mesh, name="model")
    register_mesh(mesh)

    registry = Registry()

    if resolved_backend == "nccl":
        # PyNCCL first (preferred for graph-capture ops), then NcclBackend.
        if enable_pynccl:
            pynccl = PyNCCLBackend(library_path=pynccl_library_path)
            axes = (
                list(pynccl_axes) if pynccl_axes is not None else list(mesh_dim_names)
            )
            local_rank = dist.get_rank() % max(torch.cuda.device_count(), 1)
            dev = torch.device(f"cuda:{local_rank}")
            pynccl.attach(mesh, axes, device=dev)
            registry.register(
                pynccl,
                prefer_for={
                    Op.ALL_REDUCE,
                    Op.ALL_GATHER,
                    Op.REDUCE_SCATTER,
                    Op.BROADCAST,
                    Op.SEND,
                    Op.RECV,
                },
            )
        registry.register(NcclBackend())

    elif resolved_backend == "gloo":
        # PyNCCL is NCCL-only — silently skip.
        registry.register(GlooBackend())

    else:
        raise ValueError(
            f"phyai.parallel.init: unsupported backend {resolved_backend!r}. "
            f"Use 'nccl', 'gloo', or 'auto'."
        )

    registry.validate()
    set_dispatcher(Dispatcher(registry=registry))

    return mesh


@contextmanager
def on_stream(s: torch.cuda.Stream) -> Iterator[None]:
    """Run the body on stream ``s``. Capture-safe (sub-stream usage works
    inside ``torch.cuda.graph`` as long as both streams are in capture
    state)."""
    prev = torch.cuda.current_stream()
    s.wait_stream(prev)
    with torch.cuda.stream(s):
        yield


def warmup(callable, /, *args, **kwargs) -> object:
    """Run ``callable`` once on a side stream, triggering each backend's
    one-time lazy init. Call before entering ``graph_capture()``."""
    if torch.cuda.is_available():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            result = callable(*args, **kwargs)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
    else:
        # CPU-only: just call.
        result = callable(*args, **kwargs)
    return result


__all__ = [
    # init / state / mesh
    "init",
    "Mesh",
    "Mode",
    "default_mesh",
    "use_mesh",
    "current_mode",
    "graph_capture",
    # collectives
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "broadcast",
    "send",
    "recv",
    "barrier",
    # streams / warmup
    "on_stream",
    "warmup",
    # backends (exposed for advanced users to register custom ones)
    "NcclBackend",
    "GlooBackend",
    "PyNCCLBackend",
    "TorchDistBackend",
    "Backend",
    "Op",
    "Topology",
    "Registry",
    "Policy",
    "DefaultPolicy",
    "ForcedPolicy",
    "Dispatcher",
    "get_dispatcher",
    # errors
    "PhyaiDistError",
    "NoBackendError",
    "CommTimeoutError",
    "CaptureUnsafeError",
]
