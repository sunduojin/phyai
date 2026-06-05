"""Run pi0 inference end-to-end through the phyai engine plugin path.

Spins up the pi0 plugin behind ``Engine``, feeds dummy inputs (random
pixels, a single-token "prompt", a random robot state, and optionally a
fixed noise tensor) for ``--batch-size`` robots, and prints per-step
latency. The action numbers are meaningless with random weights/inputs —
the script exists to verify the engine wiring and to time it.

pi0 differs from pi0.5 in its request shape:

* ``pixel_values``  : ``(B, 3, C, H, W)`` — 3 cameras, each ``C`` channels.
* ``state``         : ``(B, max_state_dim)`` — continuous robot state, which
  in pi0 becomes a *suffix* token (in pi0.5 it is folded into the prompt).
* ``noise``         : optional ``(B, chunk, max_action_dim)`` — pass a fixed
  tensor to make a run reproducible (used by parity tests vs OpenPI).

Run::

    # random weights, just exercise the wiring + timing
    uv run python examples/run_pi0.py

    # real weights (HF-style pytorch folder converted from the JAX ckpt)
    uv run python examples/run_pi0.py --checkpoint /path/to/pi0_pytorch/
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.pi0.configuration_pi0 import PI0Config
from phyai.models.pi0.main_pi0 import PI0Args
from phyai.models.pi0.scheduler_ws1_pi0 import PI0Request
from phyai.utils import load_config


def make_dummy_request(
    *,
    batch_size: int,
    plugin_cfg: PI0Config,
    device: torch.device,
    dtype: torch.dtype,
    fixed_noise: bool = False,
) -> PI0Request:
    """Build a placeholder ``PI0Request``.

    Random pixels + a one-token prompt + a random state. When
    ``fixed_noise`` is set, the flow-matching noise is drawn under a fixed
    seed and passed explicitly, so two runs (or two implementations) start
    from identical noise — the basis for an OpenPI parity check.
    """

    pixel_values = torch.rand(
        batch_size,
        3,  # cameras: base / left_wrist / right_wrist
        plugin_cfg.vision.num_channels,
        plugin_cfg.vision.image_size,
        plugin_cfg.vision.image_size,
        dtype=dtype,
        device=device,
    )
    input_ids = torch.zeros(
        batch_size, plugin_cfg.tokenizer_max_length, dtype=torch.int64, device=device
    )
    input_ids[:, 0] = 2  # any non-pad token id
    lang_lens = torch.ones(batch_size, dtype=torch.int64, device=device)
    state = torch.rand(
        batch_size, plugin_cfg.max_state_dim, dtype=dtype, device=device
    )

    noise = None
    if fixed_noise:
        gen = torch.Generator(device=device).manual_seed(0)
        noise = torch.randn(
            batch_size,
            plugin_cfg.chunk_size,
            plugin_cfg.max_action_dim,
            dtype=dtype,
            device=device,
            generator=gen,
        )

    return PI0Request(
        pixel_values=pixel_values,
        input_ids=input_ids,
        lang_lens=lang_lens,
        state=state,
        noise=noise,
    )


def benchmark(
    engine: Engine,
    request: PI0Request,
    *,
    n_warmup: int,
    n_timed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Warm + ``n_timed`` ``engine.step`` calls; return last action + ms stats."""

    actions: torch.Tensor | None = None
    for _ in range(n_warmup):
        actions = engine.step(request)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(n_timed):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        actions = engine.step(request)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    assert actions is not None
    return actions, {
        "mean": statistics.fmean(times_ms),
        "median": statistics.median(times_ms),
        "stdev": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min": min(times_ms),
        "max": max(times_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional HF-style pi0 pytorch checkpoint folder (config.json + "
            "model.safetensors[.index.json]). Omit to run with random weights "
            "for a pure wiring/timing smoke test."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--fixed-noise",
        action="store_true",
        help="Use a seed-0 noise tensor for a reproducible run.",
    )
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-timed", type=int, default=30)
    args = parser.parse_args()

    if args.checkpoint is not None:
        if not args.checkpoint.is_dir():
            raise NotADirectoryError(
                f"--checkpoint must be a directory, got: {args.checkpoint}"
            )
        plugin_cfg = load_config(args.checkpoint, PI0Config)
    else:
        plugin_cfg = PI0Config()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    engine = Engine(
        EngineArgs(
            plugin="pi0",
            plugin_args=PI0Args(
                checkpoint_dir=args.checkpoint,
                max_batch_size=args.batch_size,
            ),
            config=EngineConfig(
                device=DeviceConfig(target="cuda", params_dtype=dtype),
                runtime=RuntimeConfig(use_cuda_graph=True),
            ),
        )
    )
    try:
        request = make_dummy_request(
            batch_size=args.batch_size,
            plugin_cfg=plugin_cfg,
            device=device,
            dtype=dtype,
            fixed_noise=args.fixed_noise,
        )
        actions, stats = benchmark(
            engine, request, n_warmup=args.n_warmup, n_timed=args.n_timed
        )

        print(f"weights            : {'random' if args.checkpoint is None else args.checkpoint}")
        print(f"action chunk shape : {tuple(actions.shape)}")
        print(f"action chunk dtype : {actions.dtype}")
        print(f"action chunk device: {actions.device}")
        print(
            f"step latency       : mean={stats['mean']:.2f} ms  "
            f"median={stats['median']:.2f} ms  std={stats['stdev']:.2f} ms  "
            f"min={stats['min']:.2f} ms  max={stats['max']:.2f} ms  "
            f"(n_warmup={args.n_warmup}, n_timed={args.n_timed})"
        )
        print(f"first action row   : {actions[0, 0].float().tolist()}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
