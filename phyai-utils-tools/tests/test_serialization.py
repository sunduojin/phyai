"""Tests for lerobot-format serialization round-trip + real-ckpt load."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from phyai_utils_tools.processing import (
    ProcessorPipeline,
    ProcessorStepError,
    ProcessorStepRegistry,
)
from phyai_utils_tools.processing.steps import NormalizerStep
from phyai_utils_tools.processing.transition import STATE

_STATE_FEAT = "observation.state"
_PI05_BASE = Path("/mnt/bos-multimodal/wangchenghua/hf_models/pi05_base")


def test_unknown_step_raises(tmp_path):
    """A config with an unregistered step name raises a clear ProcessorStepError."""
    cfg = {"name": "p", "steps": [{"registry_name": "does_not_exist", "config": {}}]}
    (tmp_path / "p.json").write_text(json.dumps(cfg))
    with pytest.raises(ProcessorStepError, match="Unknown processor step"):
        ProcessorPipeline.from_pretrained(tmp_path, "p.json")


def test_save_load_roundtrip_with_stats(tmp_path):
    """save_pretrained -> from_pretrained reproduces config + stats sidecar."""
    feats = {_STATE_FEAT: {"type": "STATE", "shape": [3]}}
    stats = {_STATE_FEAT: {"mean": [1.0, 2.0, 3.0], "std": [0.5, 0.5, 0.5]}}
    pipe = ProcessorPipeline(
        steps=[
            NormalizerStep(features=feats, norm_map={"STATE": "MEAN_STD"}, stats=stats)
        ],
        name="policy_preprocessor",
    )
    json_path = pipe.save_pretrained(
        tmp_path, config_filename="policy_preprocessor.json"
    )
    assert json_path.exists()

    # A stats sidecar .safetensors was written.
    sidecars = list(tmp_path.glob("*.safetensors"))
    assert len(sidecars) == 1

    loaded = ProcessorPipeline.from_pretrained(tmp_path, "policy_preprocessor.json")
    assert len(loaded) == 1
    assert loaded.get_config() == pipe.get_config()
    # Stats survived the round-trip: normalize gives the same result.
    x = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(loaded[0]({STATE: x})[STATE], pipe[0]({STATE: x})[STATE])
    # mean==x so normalized == 0.
    assert torch.allclose(loaded[0]({STATE: x})[STATE], torch.zeros_like(x), atol=1e-6)


def test_config_json_schema(tmp_path):
    """Emitted json matches lerobot's {name, steps:[{registry_name, config}]}."""
    pipe = ProcessorPipeline(
        steps=[
            NormalizerStep(
                features={_STATE_FEAT: {"type": "STATE", "shape": [2]}},
                norm_map={"STATE": "QUANTILES"},
                stats=None,
            )
        ],
        name="policy_preprocessor",
    )
    pipe.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    cfg = json.loads((tmp_path / "policy_preprocessor.json").read_text())
    assert cfg["name"] == "policy_preprocessor"
    step = cfg["steps"][0]
    assert step["registry_name"] == "normalizer_processor"
    assert set(step["config"]) == {"eps", "features", "norm_map"}
    assert step["config"]["norm_map"] == {"STATE": "QUANTILES"}
    assert "state_file" not in step  # no stats => no sidecar


class _StubTokenizer:
    def __call__(
        self, prompts, max_length, padding, padding_side, truncation, return_tensors
    ):
        b = len(prompts)
        ids = torch.zeros(b, max_length, dtype=torch.long)
        mask = torch.zeros(b, max_length, dtype=torch.long)
        mask[:, 0] = 1
        return {"input_ids": ids, "attention_mask": mask}


@pytest.mark.skipif(not _PI05_BASE.is_dir(), reason="pi05_base checkpoint not present")
def test_load_real_pi05_base_policy():
    """Load the real lerobot policy_*processor.json shipped in pi05_base."""
    # Importing the pi05 model package registers its model-specific step
    # (pi05_prepare_state_tokenizer_processor_step) — required to resolve the
    # pi05 json. This is the deliberate boundary: loading a model's checkpoint
    # needs that model's package imported.
    import phyai_utils_tools.models.pi05  # noqa: F401

    step_kwargs = {
        "tokenizer_processor": {"tokenizer": _StubTokenizer()},
        "device_processor": {"device": "cpu"},
    }
    pre = ProcessorPipeline.from_pretrained(
        _PI05_BASE, "policy_preprocessor.json", step_kwargs=step_kwargs
    )
    post = ProcessorPipeline.from_pretrained(
        _PI05_BASE, "policy_postprocessor.json", step_kwargs=step_kwargs
    )
    pre_names = [type(s)._registry_name for s in pre.steps]
    assert pre_names == [
        "rename_observations_processor",
        "to_batch_processor",
        "normalizer_processor",
        "pi05_prepare_state_tokenizer_processor_step",
        "tokenizer_processor",
        "device_processor",
    ]
    post_names = [type(s)._registry_name for s in post.steps]
    assert post_names == ["unnormalizer_processor", "device_processor"]
    # Empty features in the checkpoint => identity normalize (no sidecar).
    norm = pre.steps[2]
    x = torch.tensor([[0.5, -0.5]])
    assert torch.equal(norm({STATE: x})[STATE], x)


def test_registry_has_lerobot_names():
    """All lerobot pi05 step names are registered.

    The generic steps register on ``import ...processing.steps``; pi0.5's
    model-specific step registers on ``import ...models.pi05``.
    """
    import phyai_utils_tools.models.pi05  # noqa: F401
    import phyai_utils_tools.processing.steps  # noqa: F401

    names = set(ProcessorStepRegistry.list())
    for n in (
        "rename_observations_processor",
        "to_batch_processor",
        "normalizer_processor",
        "unnormalizer_processor",
        "pi05_prepare_state_tokenizer_processor_step",
        "tokenizer_processor",
        "device_processor",
    ):
        assert n in names, n
