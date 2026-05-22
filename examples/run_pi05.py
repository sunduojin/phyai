"""Run pi0.5 inference end-to-end through the phyai engine plugin path.

This example exercises the engine + plugin contract:

    EngineArgs(plugin_args=PI05Args(...))  ->  Engine(engine_args)
        ->  engine.step(PI05Request(...))   ->  action chunk

The inputs here are *dummy* — random pixel values and zero-padded
``input_ids`` with a single non-pad token. The intent is to show the
calling convention and verify that the full graph runs; the action
output is therefore not meaningful. Wire a real image preprocessor
and tokenizer on top to get sane actions.

Run::

    uv run python examples/run_pi05.py
"""

from __future__ import annotations

import statistics
from pathlib import Path

import torch

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.pi05.main_pi05 import PI05Args
from phyai.models.pi05.scheduler_single_batch_pi05 import PI05Request


PI05_BASE_WEIGHTS = Path(
    "/mnt/bos-multimodal/wangchenghua/hf_models/pi05_base/model.safetensors"
)


def make_dummy_request(
    *,
    batch_size: int,
    image_size: int,
    tokenizer_max_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> PI05Request:
    """Build a placeholder :class:`PI05Request`.

    Three random images per robot, a single-token "prompt" padded to
    ``tokenizer_max_length`` with zeros. ``noise=None`` lets the
    scheduler sample a fresh Gaussian for the flow-matching loop.
    """
    pixel_values = torch.rand(
        batch_size, 3, 3, image_size, image_size, dtype=dtype, device=device
    )
    input_ids = torch.zeros(
        batch_size, tokenizer_max_length, dtype=torch.int64, device=device
    )
    input_ids[:, 0] = 2  # any non-pad token id; defaults to <bos>-ish
    lang_lens = torch.ones(batch_size, dtype=torch.int64, device=device)
    return PI05Request(
        pixel_values=pixel_values, input_ids=input_ids, lang_lens=lang_lens
    )


def main() -> None:
    if not PI05_BASE_WEIGHTS.exists():
        raise FileNotFoundError(
            f"pi05_base weights not found at {PI05_BASE_WEIGHTS}; edit the "
            f"PI05_BASE_WEIGHTS constant or copy the safetensors file."
        )

    engine_args = EngineArgs(
        plugin="pi05",
        plugin_args=PI05Args(
            weights_paths=[PI05_BASE_WEIGHTS],
            batch_size=1,
        ),
        config=EngineConfig(
            device=DeviceConfig(target="cuda", params_dtype=torch.bfloat16),
            runtime=RuntimeConfig(use_cuda_graph=True),
        ),
    )

    engine = Engine(engine_args)
    try:
        cfg = engine_args.plugin_args.config
        engine_config = engine.config

        # Seed before sampling pixels and before the first ``engine.step``
        # so the dummy inputs and the per-step ``torch.randn`` noise are
        # reproducible. Match the seed used by the LeRobot comparison
        # script (.tmp/lerobot/.tmp/run_pi05.py) to line the two up.
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        request = make_dummy_request(
            batch_size=engine_args.plugin_args.batch_size,
            image_size=cfg.vision.image_size,
            tokenizer_max_length=cfg.tokenizer_max_length,
            device=torch.device(engine_config.device.target),
            dtype=engine_config.device.params_dtype,
        )

        # Warmup: covers any lazy allocations the captured graphs deferred,
        # plus first-call autotuning. ``n_timed`` separate ``cuda.Event``
        # pairs pin the per-step latency so we can report mean / median /
        # std rather than a single noisy sample.
        n_warmup = 3
        n_timed = 30
        for _ in range(n_warmup):
            _ = engine.step(request)
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

        mean_ms = statistics.fmean(times_ms)
        median_ms = statistics.median(times_ms)
        stdev_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        min_ms = min(times_ms)
        max_ms = max(times_ms)

        print(f"action chunk shape : {tuple(actions.shape)}")
        print(f"action chunk dtype : {actions.dtype}")
        print(f"action chunk device: {actions.device}")
        print(
            f"step latency       : mean={mean_ms:.2f} ms  "
            f"median={median_ms:.2f} ms  std={stdev_ms:.2f} ms  "
            f"min={min_ms:.2f} ms  max={max_ms:.2f} ms  "
            f"(n_warmup={n_warmup}, n_timed={n_timed})"
        )
        print(f"first action row   : {actions[0, 0].float().tolist()}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
