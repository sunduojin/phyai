#!/usr/bin/env python
"""Run LeRobot PI0 on fixed random inputs for cross-framework comparison.

This script writes the raw random inputs and LeRobot outputs to one .pt file.
Use the saved tensors as the source of truth when running another PI0
implementation, so both frameworks see exactly the same images, state, actions,
flow-matching noise, and inference noise.

Run from the lerobot repo root:

    uv run python examples/pi0_random_lerobot_compare.py \
        --out ../pt/pi0bf16ini.pt \
        --deterministic 
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi0 import PI0Policy
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


class FixedLanguageInputsProcessorStep(ProcessorStep):
    def __init__(self, *, max_length: int, vocab_size: int) -> None:
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = dict(new_transition.get(TransitionKey.OBSERVATION) or {})
        batch_size = _infer_batch_size(observation, new_transition.get(TransitionKey.ACTION))
        device = _infer_device(observation, new_transition.get(TransitionKey.ACTION))
        tokens, attention_mask = make_language_inputs(
            batch_size=batch_size,
            max_length=self.max_length,
            vocab_size=self.vocab_size,
            device=device,
        )
        observation[OBS_LANGUAGE_TOKENS] = tokens
        observation[OBS_LANGUAGE_ATTENTION_MASK] = attention_mask
        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def transform_features(self, features):
        return features


def seed_everything(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_dataset_stats(device: torch.device) -> dict:
    image_stats = {
        "mean": torch.zeros(3, 224, 224, device=device),
        "std": torch.ones(3, 224, 224, device=device),
        "q01": torch.zeros(3, 224, 224, device=device),
        "q99": torch.ones(3, 224, 224, device=device),
    }
    return {
        OBS_STATE: {
            "mean": torch.zeros(32, device=device),
            "std": torch.ones(32, device=device),
            "q01": torch.zeros(32, device=device),
            "q99": torch.ones(32, device=device),
        },
        ACTION: {
            "mean": torch.zeros(32, device=device),
            "std": torch.ones(32, device=device),
            "q01": torch.zeros(32, device=device),
            "q99": torch.ones(32, device=device),
        },
        "images": {key: {stat: value.clone() for stat, value in image_stats.items()} for key in IMAGE_KEYS},
    }


def make_raw_batch(batch_size: int, device: torch.device, prompt: str) -> dict:
    batch = {
        OBS_STATE: torch.randn(batch_size, 32, dtype=torch.float32, device=device),
        ACTION: torch.randn(batch_size, 50, 32, dtype=torch.float32, device=device),
        "task": [prompt for _ in range(batch_size)],
    }
    for key in IMAGE_KEYS:
        batch[f"observation.images.{key}"] = torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        )
    return batch


def _infer_batch_size(observation: dict, action) -> int:
    if OBS_STATE in observation:
        return int(observation[OBS_STATE].shape[0])
    if isinstance(action, torch.Tensor):
        return int(action.shape[0])
    for value in observation.values():
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
    raise ValueError("Cannot infer batch size for fixed language inputs.")


def _infer_device(observation: dict, action) -> torch.device:
    if OBS_STATE in observation:
        return observation[OBS_STATE].device
    if isinstance(action, torch.Tensor):
        return action.device
    for value in observation.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def make_language_inputs(
    *,
    batch_size: int,
    max_length: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_id = min(2, vocab_size - 1)
    tokens = torch.full((batch_size, max_length), token_id, dtype=torch.long, device=device)
    attention_mask = torch.ones((batch_size, max_length), dtype=torch.bool, device=device)
    return tokens, attention_mask


def make_preprocessor(policy: PI0Policy, dataset_stats: dict) -> PolicyProcessorPipeline[dict, dict]:
    embed_tokens = policy.model.paligemma_with_expert.paligemma.model.language_model.embed_tokens
    relative_step = RelativeActionsProcessorStep(
        enabled=policy.config.use_relative_actions,
        exclude_joints=getattr(policy.config, "relative_exclude_joints", []),
        action_names=getattr(policy.config, "action_feature_names", None),
    )
    return PolicyProcessorPipeline[dict, dict](
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            FixedLanguageInputsProcessorStep(
                max_length=policy.config.tokenizer_max_length,
                vocab_size=embed_tokens.num_embeddings,
            ),
            DeviceProcessorStep(device=policy.config.device),
            relative_step,
            NormalizerProcessorStep(
                features={**policy.config.input_features, **policy.config.output_features},
                norm_map=policy.config.normalization_mapping,
                stats=dataset_stats,
            ),
        ],
    )


def clone_batch(batch: dict) -> dict:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else list(value) for key, value in batch.items()
    }


@contextlib.contextmanager
def fixed_flow_sampling(model, *, noise: torch.Tensor, time: torch.Tensor) -> Iterator[None]:
    original_sample_noise = model.sample_noise
    original_sample_time = model.sample_time

    def sample_noise(shape, device):
        if tuple(shape) != tuple(noise.shape):
            raise ValueError(f"Expected noise shape {tuple(noise.shape)}, got {tuple(shape)}")
        return noise.to(device=device)

    def sample_time(batch_size, device):
        if batch_size != time.shape[0]:
            raise ValueError(f"Expected time batch size {time.shape[0]}, got {batch_size}")
        return time.to(device=device)

    model.sample_noise = sample_noise
    model.sample_time = sample_time
    try:
        yield
    finally:
        model.sample_noise = original_sample_noise
        model.sample_time = original_sample_time


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return list(value)
    return value


def tensor_summary(tensor: torch.Tensor) -> dict[str, float | list[int]]:
    tensor = tensor.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "l2": float(torch.linalg.vector_norm(tensor)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="/data/share/pi0_base")
    parser.add_argument("--output", type=Path, default=Path("pi0_lerobot_random_outputs.pt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--prompt", default="Pick up the red block and place it in the bin")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    seed_everything(args.seed, deterministic=args.deterministic)

    config = PreTrainedConfig.from_pretrained(args.model_id)
    config.device = str(device)
    config.dtype = args.dtype
    config.compile_model = args.compile

    policy = PI0Policy.from_pretrained(args.model_id, config=config, strict=True)
    policy.to(device).eval()
    policy.config.device = str(device)

    dataset_stats = make_dataset_stats(device)
    preprocessor = make_preprocessor(policy, dataset_stats)

    raw_batch = make_raw_batch(args.batch_size, device, args.prompt)
    lerobot_batch = preprocessor(clone_batch(raw_batch))

    forward_noise = torch.randn(args.batch_size, 50, 32, dtype=torch.float32, device=device)
    forward_time = torch.linspace(0.2, 0.8, args.batch_size, dtype=torch.float32, device=device)
    sample_noise = torch.randn(args.batch_size, 50, 32, dtype=torch.float32, device=device)

    with torch.no_grad():
        with fixed_flow_sampling(policy.model, noise=forward_noise, time=forward_time):
            loss, loss_dict = policy(lerobot_batch, reduction="none")
        actions = policy.predict_action_chunk(lerobot_batch, noise=sample_noise, num_steps=args.num_steps)

    payload = {
        "meta": {
            "model_id": args.model_id,
            "seed": args.seed,
            "device": str(device),
            "dtype": args.dtype,
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "image_keys": IMAGE_KEYS,
        },
        "raw_batch": cpu_tree(raw_batch),
        "processed_batch": cpu_tree(lerobot_batch),
        "forward_noise": cpu_tree(forward_noise),
        "forward_time": cpu_tree(forward_time),
        "sample_noise": cpu_tree(sample_noise),
        "loss": cpu_tree(loss),
        "loss_dict": loss_dict,
        "actions": cpu_tree(actions),
        "summary": {
            "loss": tensor_summary(loss),
            "actions": tensor_summary(actions),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    print(json.dumps(payload["meta"], indent=2))
    print(json.dumps(payload["summary"], indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
