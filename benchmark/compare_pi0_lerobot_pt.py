"""Compare phyai pi0 outputs against a LeRobot-saved random test payload.

The expected ``.pt`` file is produced by LeRobot's
``examples/pi0_random_lerobot_compare.py`` and contains:

* ``raw_batch`` with raw state/images/tasks.
* ``processed_batch`` with tokenized language inputs.
* ``sample_noise`` used by LeRobot sampling.
* ``actions`` as the LeRobot reference action chunk.

Example:

    uv run python benchmark/compare_pi0_lerobot_pt.py \
        --checkpoint /data/share/pi0_base \
        --pt /data/share/pi0_random_lerobot_compare.pt \
        --dtype float32
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import BackendConfig, DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.pi0.configuration_pi0 import PI0Config
from phyai.models.pi0.main_pi0 import PI0Args
from phyai.models.pi0.scheduler_ws1_pi0 import PI0Request
from phyai.utils import load_config as load_phyai_config


IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

TOKEN_ALIASES = (
    "observation.language.tokens",
    "observation.language_tokens",
    "language_tokens",
    "input_ids",
)

MASK_ALIASES = (
    "observation.language.attention_mask",
    "observation.language_attention_mask",
    "language_attention_mask",
    "attention_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare phyai pi0 action output with a LeRobot .pt payload."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="pi0 checkpoint directory.")
    parser.add_argument("--pt", type=Path, required=True, help="LeRobot comparison .pt file.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Target device passed to phyai Engine. Usually cuda or cpu.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "bf16", "bfloat16", "fp16", "float16"),
        help="phyai model parameter/runtime dtype.",
    )
    parser.add_argument(
        "--attn_backend",
        default=None,
        choices=("auto", "flashinfer", "sdpa", "eager"),
        help=(
            "Attention backend. Default auto uses eager for float32 because "
            "flashinfer paged attention does not support torch.float32."
        ),
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Engine max batch size. Defaults to the payload batch size.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for torch.allclose.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for torch.allclose.",
    )
    parser.add_argument(
        "--weight-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use strict checkpoint loading in PI0Args.",
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA graph in the phyai runtime.",
    )
    parser.add_argument(
        "--save-output",
        type=Path,
        default=None,
        help="Optional path to save phyai output, reference output, diff, and metrics.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def choose_attn_backend(dtype: torch.dtype, requested: str | None) -> str:
    if requested is not None and requested != "auto":
        return requested
    if dtype is torch.float32:
        return "eager"
    return "flashinfer"


def require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Expected payload[{key!r}] to be a dict.")
    return value


def pick_tensor(mapping: dict[str, Any], aliases: tuple[str, ...], label: str) -> torch.Tensor:
    for key in aliases:
        value = mapping.get(key)
        if torch.is_tensor(value):
            return value

    available = ", ".join(sorted(mapping))
    raise KeyError(
        f"Cannot find {label}. Tried {aliases}. Available processed_batch keys: {available}"
    )


def build_pixel_values(raw_batch: dict[str, Any]) -> torch.Tensor:
    images: list[torch.Tensor] = []
    for image_key in IMAGE_KEYS:
        key = f"observation.images.{image_key}"
        image = raw_batch.get(key)
        if not torch.is_tensor(image):
            raise KeyError(f"Expected raw_batch[{key!r}] to be a tensor.")

        image = image.detach().to(dtype=torch.float32)
        if image.ndim != 4:
            raise ValueError(f"{key} must have shape (B, C, H, W), got {tuple(image.shape)}.")
        if image.shape[1] != 3:
            raise ValueError(f"{key} must be channels-first RGB, got {tuple(image.shape)}.")

        images.append(image.mul(2.0).sub(1.0))

    return torch.stack(images, dim=1)


def build_request(payload: dict[str, Any], device: torch.device) -> tuple[PI0Request, torch.Tensor]:
    raw_batch = require_mapping(payload, "raw_batch")
    processed_batch = require_mapping(payload, "processed_batch")

    state = raw_batch.get("observation.state")
    if not torch.is_tensor(state):
        raise KeyError("Expected raw_batch['observation.state'] to be a tensor.")

    sample_noise = payload.get("sample_noise")
    if not torch.is_tensor(sample_noise):
        raise KeyError("Expected payload['sample_noise'] to be a tensor.")

    reference_actions = payload.get("actions")
    if not torch.is_tensor(reference_actions):
        raise KeyError("Expected payload['actions'] to be a tensor.")

    input_ids = pick_tensor(processed_batch, TOKEN_ALIASES, "language token ids")
    attention_mask = pick_tensor(processed_batch, MASK_ALIASES, "language attention mask")
    lang_lens = attention_mask.to(dtype=torch.long).sum(dim=-1)

    request = PI0Request(
        pixel_values=build_pixel_values(raw_batch).to(device=device),
        input_ids=input_ids.to(device=device, dtype=torch.long),
        lang_lens=lang_lens.to(device=device, dtype=torch.long),
        state=state.to(device=device, dtype=torch.float32),
        noise=sample_noise.to(device=device, dtype=torch.float32),
    )
    return request, reference_actions.detach().cpu()


def load_config(checkpoint: Path, payload: dict[str, Any]) -> PI0Config:
    config = load_phyai_config(checkpoint, PI0Config)
    meta = payload.get("meta")
    if isinstance(meta, dict) and "num_steps" in meta:
        config = replace(config, num_inference_steps=int(meta["num_steps"]))
    return config


def align_shapes(actual: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if actual.shape == reference.shape:
        return actual, reference

    if (
        actual.ndim == reference.ndim == 3
        and actual.shape[0] == reference.shape[0]
        and actual.shape[1] == reference.shape[1]
        and actual.shape[2] >= reference.shape[2]
    ):
        return actual[..., : reference.shape[2]], reference

    raise ValueError(
        f"Cannot compare tensors with shapes actual={tuple(actual.shape)} "
        f"reference={tuple(reference.shape)}."
    )


def compute_metrics(actual: torch.Tensor, reference: torch.Tensor, rtol: float, atol: float) -> dict[str, Any]:
    actual, reference = align_shapes(actual.float().cpu(), reference.float().cpu())
    diff = actual - reference
    abs_diff = diff.abs()
    rel_diff = abs_diff / reference.abs().clamp_min(1e-6)

    actual_flat = actual.reshape(-1)
    reference_flat = reference.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(actual_flat, reference_flat, dim=0)

    return {
        "shape": list(actual.shape),
        "finite_actual": bool(torch.isfinite(actual).all().item()),
        "finite_reference": bool(torch.isfinite(reference).all().item()),
        "finite_diff": bool(torch.isfinite(diff).all().item()),
        "allclose": bool(torch.allclose(actual, reference, rtol=rtol, atol=atol)),
        "rtol": rtol,
        "atol": atol,
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
        "rmse": float(torch.sqrt((diff * diff).mean()).item()),
        "max_rel": float(rel_diff.max().item()),
        "mean_rel": float(rel_diff.mean().item()),
        "cosine": float(cosine.item()),
    }


def print_preview(actual: torch.Tensor, reference: torch.Tensor) -> None:
    actual, reference = align_shapes(actual.float().cpu(), reference.float().cpu())
    first_actual = actual[0, 0].tolist()
    first_reference = reference[0, 0].tolist()
    first_diff = (actual[0, 0] - reference[0, 0]).tolist()

    print("first action actual   :", first_actual)
    print("first action reference:", first_reference)
    print("first action diff     :", first_diff)


def main() -> None:
    args = parse_args()
    dtype = dtype_from_name(args.dtype)
    attn_backend = choose_attn_backend(dtype, args.attn_backend)
    device = torch.device(args.device)

    payload = torch.load(args.pt, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected .pt payload to be a dict, got {type(payload)!r}.")

    request, reference_actions = build_request(payload, device)
    batch_size = int(reference_actions.shape[0])
    max_batch_size = args.max_batch_size or batch_size
    config = load_config(args.checkpoint, payload)

    engine = Engine(
        EngineArgs(
            plugin="pi0",
            plugin_args=PI0Args(
                checkpoint_dir=args.checkpoint,
                config=config,
                max_batch_size=max_batch_size,
                weight_strict=args.weight_strict,
            ),
            config=EngineConfig(
                backends=BackendConfig(attn=attn_backend),
                device=DeviceConfig(target=args.device, params_dtype=dtype),
                runtime=RuntimeConfig(use_cuda_graph=args.cuda_graph),
            ),
        )
    )

    try:
        with torch.inference_mode():
            actual_actions = engine.step(request).detach().cpu()
    finally:
        close = getattr(engine, "close", None)
        if close is not None:
            close()

    metrics = compute_metrics(actual_actions, reference_actions, rtol=args.rtol, atol=args.atol)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print_preview(actual_actions, reference_actions)

    if args.save_output is not None:
        actual_aligned, reference_aligned = align_shapes(actual_actions.float().cpu(), reference_actions.float().cpu())
        args.save_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "phyai_actions": actual_aligned,
                "lerobot_actions": reference_aligned,
                "diff": actual_aligned - reference_aligned,
                "metrics": metrics,
            },
            args.save_output,
        )
        print(f"saved output: {args.save_output}")


if __name__ == "__main__":
    main()
