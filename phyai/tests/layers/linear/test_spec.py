"""WeightSpec tests — allocate shapes, process_after_loading, activation quant.

No CUDA required; float8_e4m3fn tensors allocate on CPU in newer PyTorch.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from phyai.layers.linear.backend import Granularity
from phyai.layers.linear.spec import (
    ActivationView,
    Bf16Spec,
    Fp8Spec,
    _convert_to_channelwise,
)


# ---------------------------------------------------------------------------
# Bf16Spec
# ---------------------------------------------------------------------------


def test_bf16_allocate_plain():
    layer = nn.Module()
    Bf16Spec().allocate(
        layer,
        input_size_per_partition=32,
        output_partition_sizes=[64],
        input_size_global=32,
        output_size_global=64,
        params_dtype=torch.bfloat16,
        weight_loader=None,
    )
    assert layer.weight.shape == (64, 32)
    assert layer.weight.dtype == torch.bfloat16
    assert not hasattr(layer, "weight_scale")
    assert layer.input_size_per_partition == 32
    assert layer.output_size_per_partition == 64
    assert layer.logical_widths == [64]


def test_bf16_allocate_fused_sizes():
    layer = nn.Module()
    Bf16Spec().allocate(
        layer,
        input_size_per_partition=16,
        output_partition_sizes=[32, 16, 16],
        input_size_global=16,
        output_size_global=64,
        params_dtype=torch.bfloat16,
        weight_loader=None,
    )
    assert layer.weight.shape == (64, 16)


def test_bf16_respects_params_dtype():
    layer = nn.Module()
    Bf16Spec().allocate(
        layer,
        input_size_per_partition=16,
        output_partition_sizes=[16],
        input_size_global=16,
        output_size_global=16,
        params_dtype=torch.float16,
        weight_loader=None,
    )
    assert layer.weight.dtype == torch.float16


def test_bf16_process_after_loading_noop():
    layer = nn.Module()
    Bf16Spec().allocate(
        layer,
        input_size_per_partition=16,
        output_partition_sizes=[16],
        input_size_global=16,
        output_size_global=16,
        params_dtype=torch.bfloat16,
        weight_loader=None,
    )
    # Fill with deterministic values so torch.equal doesn't trip over NaNs
    # that torch.empty may leave behind.
    layer.weight.data.fill_(0.5)
    before = layer.weight.data.clone()
    Bf16Spec().process_after_loading(layer)
    assert torch.equal(layer.weight.data, before)


def test_bf16_quantize_activation_identity():
    layer = nn.Module()
    x = torch.randn(4, 16, dtype=torch.bfloat16)
    act = Bf16Spec().quantize_activation(x, layer)
    assert isinstance(act, ActivationView)
    assert act.x is x
    assert act.x_scale is None
    assert act.granularity == Granularity.PER_TENSOR


# ---------------------------------------------------------------------------
# Fp8Spec
# ---------------------------------------------------------------------------


def test_fp8_per_tensor_shapes_pre_and_post_loading():
    spec = Fp8Spec(granularity=Granularity.PER_TENSOR)
    layer = nn.Module()
    spec.allocate(
        layer,
        input_size_per_partition=64,
        output_partition_sizes=[32, 16],  # two logical matrices
        input_size_global=64,
        output_size_global=48,
        params_dtype=torch.bfloat16,
        weight_loader=None,
    )
    assert layer.weight.shape == (48, 64)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.shape == (2,)  # one per logical matrix
    assert layer.input_scale.shape == (1,)
    assert spec.spec_id == "fp8_per_tensor"

    # After loading, weight_scale is fanned out to per-channel.
    spec.process_after_loading(layer)
    assert layer.weight_scale.shape == (48,)


def test_fp8_per_channel_shapes():
    spec = Fp8Spec(granularity=Granularity.PER_CHANNEL)
    layer = nn.Module()
    spec.allocate(
        layer,
        input_size_per_partition=64,
        output_partition_sizes=[128],
        input_size_global=64,
        output_size_global=128,
        params_dtype=torch.bfloat16,
        weight_loader=None,
    )
    assert layer.weight.shape == (128, 64)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.shape == (128,)
    assert not hasattr(layer, "input_scale")  # computed at runtime
    assert spec.spec_id == "fp8_per_channel"


def test_fp8_block_shapes():
    spec = Fp8Spec(granularity=Granularity.BLOCK, block_shape=(128, 128))
    layer = nn.Module()
    spec.allocate(
        layer,
        input_size_per_partition=256,
        output_partition_sizes=[384],
        input_size_global=256,
        output_size_global=384,
        params_dtype=torch.bfloat16,
        weight_loader=None,
    )
    assert layer.weight.shape == (384, 256)
    assert layer.weight_scale.shape == (3, 2)  # 384/128 x 256/128
    assert spec.spec_id == "fp8_block_128_128"


def test_fp8_block_requires_block_shape():
    with pytest.raises(ValueError, match="block_shape"):
        Fp8Spec(granularity=Granularity.BLOCK)


def test_fp8_block_enforces_divisibility():
    spec = Fp8Spec(granularity=Granularity.BLOCK, block_shape=(128, 128))
    layer = nn.Module()
    with pytest.raises(ValueError, match="not divisible"):
        spec.allocate(
            layer,
            input_size_per_partition=100,  # not /128
            output_partition_sizes=[256],
            input_size_global=100,
            output_size_global=256,
            params_dtype=torch.bfloat16,
            weight_loader=None,
        )


def test_convert_to_channelwise_basic():
    scales = torch.tensor([0.25, 0.5, 1.0])
    out = _convert_to_channelwise(scales, [2, 1, 3])
    assert out.tolist() == [0.25, 0.25, 0.5, 1.0, 1.0, 1.0]


def test_fp8_quantize_activation_per_tensor():
    spec = Fp8Spec(granularity=Granularity.PER_TENSOR)
    layer = nn.Module()
    spec.allocate(
        layer,
        input_size_per_partition=16,
        output_partition_sizes=[32],
        input_size_global=16,
        output_size_global=32,
        params_dtype=torch.bfloat16,
        weight_loader=None,
    )
    x = torch.randn(4, 16)
    act = spec.quantize_activation(x, layer)
    assert act.x.dtype == torch.float8_e4m3fn
    assert act.x_scale is layer.input_scale
    assert act.granularity == Granularity.PER_TENSOR


def test_fp8_quantize_activation_per_channel_rowwise():
    spec = Fp8Spec(granularity=Granularity.PER_CHANNEL)
    layer = nn.Module()
    spec.allocate(
        layer,
        input_size_per_partition=16,
        output_partition_sizes=[32],
        input_size_global=16,
        output_size_global=32,
        params_dtype=torch.bfloat16,
        weight_loader=None,
    )
    x = torch.randn(4, 16) * 5.0
    act = spec.quantize_activation(x, layer)
    assert act.x.dtype == torch.float8_e4m3fn
    assert act.x_scale.shape == (4, 1)
    assert act.granularity == Granularity.PER_CHANNEL


def test_fp8_needs_act_quant_true():
    assert Fp8Spec(granularity=Granularity.PER_CHANNEL).needs_act_quant is True
    assert Bf16Spec().needs_act_quant is False


def test_fp8_spec_id_format():
    assert Fp8Spec(granularity=Granularity.PER_TENSOR).spec_id == "fp8_per_tensor"
    assert Fp8Spec(granularity=Granularity.PER_CHANNEL).spec_id == "fp8_per_channel"
    assert (
        Fp8Spec(granularity=Granularity.BLOCK, block_shape=(128, 128)).spec_id
        == "fp8_block_128_128"
    )
    assert (
        Fp8Spec(granularity=Granularity.BLOCK, block_shape=(64, 256)).spec_id
        == "fp8_block_64_256"
    )
