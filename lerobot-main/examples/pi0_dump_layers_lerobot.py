#!/usr/bin/env python
"""Dump LeRobot PI0 intermediate tensors for layer-wise comparison with phyai.

Uses the same .pt payload produced by ``pi0_random_lerobot_compare.py`` so both
sides see identical inputs. Dumps standardized keys (img_emb, lang_emb,
prefix_embs, prefix_hidden, state_emb, action_time_emb, v_t_step0, ...) that
``compare_pi0_layer_dumps.py`` (in the phyai repo) consumes.

Run from the lerobot repo root:

    uv run python examples/pi0_dump_layers_lerobot.py \
        --pt ../pt/pi0bf16ini.pt \
        --out ../pt/lerobot_bf16.pt \
        --dtype bfloat16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi0 import PI0Policy
from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().cpu()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=Path, required=True, help="payload from pi0_random_lerobot_compare.py")
    ap.add_argument("--out", type=Path, default=Path("pi0_lerobot_layer_dump.pt"))
    ap.add_argument("--model_id", default="/data/share/pi0_base", help="defaults to payload meta model_id")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--num_steps", type=int, default=None, help="defaults to payload meta num_steps")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    payload = torch.load(args.pt, map_location="cpu", weights_only=False)
    model_id = args.model_id or payload["meta"]["model_id"]

    batch = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in payload["processed_batch"].items()
    }
    noise = payload["sample_noise"].to(device=device, dtype=torch.float32)

    config = PreTrainedConfig.from_pretrained(model_id)
    config.device = str(device)
    config.dtype = args.dtype
    config.compile_model = False
    policy = PI0Policy.from_pretrained(model_id, config=config, strict=True)
    policy.to(device).eval()
    model = policy.model

    dump: dict = {}

    with torch.no_grad():
        # ---- inputs, exactly as predict_action_chunk prepares them ----
        images, img_masks = policy._preprocess_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        state = policy.prepare_state(batch)

        present_img_keys = [k for k in policy.config.image_features if k in batch]

        # ---- cut 1: per-camera image embeddings (B, n_cam, P, D) ----
        img_embs = [model.paligemma_with_expert.embed_image(img) for img in images]
        dump["img_emb"] = torch.stack([cpu(e) for e in img_embs], dim=1)

        # ---- cut 2: language token embeddings (B, L, D) ----
        dump["lang_emb"] = cpu(model.paligemma_with_expert.embed_language_tokens(lang_tokens))

        # ---- cut 3: packed prefix (B, n_img*n_cam + L, D) ----
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        dump["prefix_embs"] = cpu(prefix_embs)

        # ---- cut 4: LLM prefix forward (post final norm) + KV cache ----
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        mask4d = model._prepare_attention_masks_4d(prefix_att_2d)
        model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        (prefix_out, _), past_key_values = model.paligemma_with_expert.forward(
            attention_mask=mask4d,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        dump["prefix_hidden"] = cpu(prefix_out)

        # ---- cut 5: suffix embeddings at t=1.0 with the fixed noise ----
        bsize = state.shape[0]
        time = torch.ones(bsize, dtype=torch.float32, device=device)
        suffix_embs, _, _, _ = model.embed_suffix(state, noise, time)
        dump["x_t_step0"] = cpu(noise)
        dump["state_emb"] = cpu(suffix_embs[:, 0])
        dump["action_time_emb"] = cpu(suffix_embs[:, 1:])

        # ---- cut 6: denoise step velocities ----
        num_steps = args.num_steps if args.num_steps is not None else int(
            payload.get("meta", {}).get("num_steps", policy.config.num_inference_steps)
        )
        if num_steps <= 0:
            raise ValueError(f"--num-steps must be positive, got {num_steps}")
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            t = 1.0 + step * dt
            time = torch.full((bsize,), t, dtype=torch.float32, device=device)
            v_t = model.denoise_step(
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time,
            )
            dump[f"v_t_step{step}"] = cpu(v_t)
            x_t = x_t + dt * v_t

    # reference final actions from the original payload
    dump["actions"] = payload["actions"].detach().float().cpu()
    dump["meta"] = {
        "side": "lerobot",
        "model_id": model_id,
        "dtype": args.dtype,
        "image_keys_order": present_img_keys,
        "tokenizer_len": int(lang_tokens.shape[1]),
        "chunk_size": int(policy.config.chunk_size),
        "num_steps": num_steps,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dump, args.out)
    for k, v in dump.items():
        if torch.is_tensor(v):
            print(f"{k:18s} {tuple(v.shape)}  l2={v.norm():.4f}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
