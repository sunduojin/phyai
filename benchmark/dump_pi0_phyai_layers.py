"""Dump phyai pi0 intermediate tensors for layer-wise comparison with LeRobot.

Wraps the live scheduler/runner objects to capture intermediates while running
the normal ``engine.step``, so the dump reflects exactly what the engine
computes. CUDA graph is force-disabled (wrappers cannot intercept graph
replay).

Run from the phyai repo root:

    uv run python benchmark/dump_pi0_phyai_layers.py \
        --checkpoint /data/share/pi0_base \
        --pt pt/pi0bf16ini.pt \
        --dtype bfloat16 \
        --out pt/phyai_bf16.pt

Then compare with the lerobot dump:

    uv run python benchmark/compare_pi0_layer_dumps.py \
        pt/phyai_bf16.pt pt/lerobot_bf16.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from compare_pi0_lerobot_pt import (
    build_request,
    choose_attn_backend,
    dtype_from_name,
    load_config,
)
from phyai.engine import Engine, EngineArgs
from phyai.engine_config import BackendConfig, DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.pi0.main_pi0 import PI0Args


def cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().cpu()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--pt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("benchmark/pi0_phyai_layer_dump.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=("float32", "bf16", "bfloat16"))
    ap.add_argument("--attn_backend", default=None, choices=("auto", "flashinfer", "sdpa", "eager"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    dtype = dtype_from_name(args.dtype)
    attn_backend = choose_attn_backend(dtype, args.attn_backend)
    device = torch.device(args.device)

    payload = torch.load(args.pt, map_location="cpu", weights_only=False)
    request, reference_actions = build_request(payload, device)
    batch_size = int(reference_actions.shape[0])
    config = load_config(args.checkpoint, payload)

    engine = Engine(
        EngineArgs(
            plugin="pi0",
            plugin_args=PI0Args(
                checkpoint_dir=args.checkpoint,
                config=config,
                max_batch_size=batch_size,
                weight_strict=True,
            ),
            config=EngineConfig(
                backends=BackendConfig(attn=attn_backend),
                device=DeviceConfig(target=args.device, params_dtype=dtype),
                runtime=RuntimeConfig(use_cuda_graph=False),
            ),
        )
    )

    sched = engine.entry.scheduler
    if sched.llm_runner.graph is not None or sched.expert_runner.graph is not None:
        raise RuntimeError("CUDA graph is active; wrappers cannot capture. Disable cuda graph.")

    dump: dict = {}

    # ---- cut 1: vision tower outputs, one call per sample ----
    vis_outs: list[torch.Tensor] = []
    orig_vis_fwd = sched.vision_runner.forward

    def vis_fwd(batch):
        out = orig_vis_fwd(batch)
        vis_outs.append(cpu(out))  # (n_cam, P, D)
        return out

    sched.vision_runner.forward = vis_fwd

    # ---- cut 2: language embeddings ----
    lm = sched.model.paligemma_lm
    orig_embed_lang = lm.embed_lang

    def embed_lang(input_ids):
        out = orig_embed_lang(input_ids)
        dump["lang_emb"] = cpu(out)
        return out

    lm.embed_lang = embed_lang

    # ---- cut 3: packed prefix (input to the LLM) ----
    import phyai.models.pi0.scheduler_ws1_pi0 as sched_mod

    orig_pack = sched_mod.pack_prefix_per_sample_padded

    def pack(*a, **kw):
        out = orig_pack(*a, **kw)
        dump["packed_flat"] = cpu(out)  # (max_B * n_per_sample, D)
        return out

    sched_mod.pack_prefix_per_sample_padded = pack

    # ---- cut 4: LLM prefix output (post final norm) ----
    orig_llm_fwd = sched.llm_runner._fwd

    def llm_fwd(**kw):
        out = orig_llm_fwd(**kw)
        dump["prefix_hidden_flat"] = cpu(out)
        return out

    sched.llm_runner._fwd = llm_fwd

    # ---- cut 5: suffix embeddings (first denoise step only) ----
    heads = sched.expert_runner.heads
    orig_embed_state = heads.embed_state
    orig_embed_at = heads.embed_action_time

    def embed_state(state):
        out = orig_embed_state(state)
        dump.setdefault("state_emb", cpu(out))
        return out

    def embed_action_time(x_t, time):
        out = orig_embed_at(x_t, time)
        if "action_time_emb" not in dump:
            dump["x_t_step0"] = cpu(x_t)
            dump["action_time_emb"] = cpu(out)
        return out

    heads.embed_state = embed_state
    heads.embed_action_time = embed_action_time

    # ---- cut 6: denoise step velocities ----
    orig_expert_fwd = sched.expert_runner.forward
    denoise_step = 0

    def expert_fwd(batch):
        nonlocal denoise_step
        out = orig_expert_fwd(batch)
        dump[f"v_t_step{denoise_step}"] = cpu(out)
        denoise_step += 1
        return out

    sched.expert_runner.forward = expert_fwd

    # ---- run ----
    try:
        with torch.inference_mode():
            actions = engine.step(request).detach().float().cpu()
    finally:
        sched_mod.pack_prefix_per_sample_padded = orig_pack
        close = getattr(engine, "close", None)
        if close is not None:
            close()

    # ---- reshape flat buffers to lerobot-comparable layouts ----
    B = batch_size
    n_per_sample = sched.n_per_sample
    n_img = sched.image_token_count
    lang_len = int(request.lang_lens.max())
    real_len = n_img + lang_len

    dump["img_emb"] = torch.stack(vis_outs, dim=0)[:B]  # (B, n_cam, P, D)
    dump["lang_emb"] = dump["lang_emb"][:B]
    packed = dump.pop("packed_flat").view(-1, n_per_sample, dump["lang_emb"].shape[-1])
    dump["prefix_embs"] = packed[:B, :real_len]
    hidden = dump.pop("prefix_hidden_flat").view(-1, n_per_sample, packed.shape[-1])
    dump["prefix_hidden"] = hidden[:B, :real_len]
    dump["state_emb"] = dump["state_emb"][:B]
    dump["x_t_step0"] = dump["x_t_step0"][:B]
    dump["action_time_emb"] = dump["action_time_emb"][:B]
    for key in list(dump):
        if key.startswith("v_t_step"):
            dump[key] = dump[key][:B]
    dump["actions"] = actions

    dump["meta"] = {
        "side": "phyai",
        "dtype": args.dtype,
        "attn_backend": attn_backend,
        "image_keys_order": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
        "n_per_sample": n_per_sample,
        "n_img": n_img,
        "lang_len": lang_len,
        "num_steps": config.num_inference_steps,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dump, args.out)
    for k, v in dump.items():
        if torch.is_tensor(v):
            print(f"{k:18s} {tuple(v.shape)}  l2={v.norm():.4f}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
