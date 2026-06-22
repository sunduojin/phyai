"""Compare lerobot vs phyai pi0 layer dumps and locate the first divergence.

    uv run python benchmark/compare_pi0_layer_dumps.py \
        pi0_lerobot_layer_dump.pt benchmark/pi0_phyai_layer_dump.pt
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch
import torch.nn.functional as F

# (key, description) in pipeline order: the first bad cut localises the bug.
BASE_CUTS = [
    ("img_emb", "image tokens (SigLIP + projector)"),
    ("lang_emb", "language token embeddings"),
    ("prefix_embs", "packed prefix (LLM input)"),
    ("prefix_hidden", "LLM prefix output (post norm)"),
    ("x_t_step0", "noise fed to expert at step 0"),
    ("state_emb", "state token embedding"),
    ("action_time_emb", "action+time token embeddings"),
]


def align(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if a.shape == b.shape:
        return a, b, False
    if a.ndim != b.ndim:
        raise ValueError(f"rank mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    sl = tuple(slice(0, min(da, db)) for da, db in zip(a.shape, b.shape))
    return a[sl], b[sl], True


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b, clipped = align(a.float(), b.float())
    af, bf = a.flatten(), b.flatten()
    diff = (af - bf).abs()
    return {
        "cos": float(F.cosine_similarity(af, bf, dim=0)),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "l2_a": float(af.norm()),
        "l2_b": float(bf.norm()),
        "clipped": clipped,
    }

def _meta_num_steps(*dumps: dict) -> int | None:
    for dump in dumps:
        meta = dump.get("meta", {})
        if not isinstance(meta, dict):
            continue
        for key in ("num_steps", "num_inference_steps"):
            value = meta.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


def _max_numbered_step(dumps: tuple[dict, ...], prefix: str) -> int | None:
    max_step: int | None = None
    for dump in dumps:
        for key in dump:
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix) :]
            if suffix.isdigit():
                step = int(suffix)
                max_step = step if max_step is None else max(max_step, step)
    return max_step


def infer_vt_steps(*dumps: dict) -> int:
    meta_steps = _meta_num_steps(*dumps)
    if meta_steps is not None:
        return meta_steps

    max_step = _max_numbered_step(dumps, "v_t_step")
    if max_step is not None:
        return max_step + 1

    return 1


def build_cuts(*, vt_steps: int) -> list[tuple[str, str]]:
    vt_cuts = [(f"v_t_step{step}", f"denoise step {step} velocity") for step in range(vt_steps)]
    return [*BASE_CUTS, *vt_cuts, ("actions", "final action chunk")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("lerobot_dump", type=Path)
    ap.add_argument("phyai_dump", type=Path)
    ap.add_argument("--threshold", type=float, default=0.99, help="cosine below this = diverged")
    ap.add_argument(
        "--vt-steps",
        type=int,
        default=None,
        help="number of v_t_stepN tensors to compare; defaults to dump meta num_steps or present keys",
    )
    args = ap.parse_args()

    L = torch.load(args.lerobot_dump, map_location="cpu", weights_only=False)
    P = torch.load(args.phyai_dump, map_location="cpu", weights_only=False)
    vt_steps = args.vt_steps if args.vt_steps is not None else infer_vt_steps(L, P)
    if vt_steps <= 0:
        raise ValueError(f"--vt-steps must be positive, got {vt_steps}")
    cuts = build_cuts(vt_steps=vt_steps)
    desc_by_key = dict(cuts)

    lk = L.get("meta", {}).get("image_keys_order")
    pk = P.get("meta", {}).get("image_keys_order")
    print(f"lerobot image order: {lk}")
    print(f"phyai   image order: {pk}")
    if lk and pk and list(lk) != list(pk):
        print("!! camera order differs between the two sides - fix this first\n")

    print(
        f"{'cut':18s} {'cosine':>9s} {'max_abs':>10s} {'mean_abs':>10s} "
        f"{'l2(lerobot)':>12s} {'l2(phyai)':>12s}"
    )
    print("-" * 77)

    first_bad: str | None = None
    for key, desc in cuts:
        if key not in L or key not in P:
            print(f"{key:18s} {'missing':>9s}   (lerobot={key in L}, phyai={key in P})")
            continue
        m = metrics(L[key], P[key])
        flag = ""
        if m["cos"] < args.threshold and first_bad is None:
            first_bad = key
            flag = "  <-- FIRST DIVERGENCE"
        elif m["cos"] < args.threshold:
            flag = "  (bad)"
        clip = " [shape-clipped]" if m["clipped"] else ""
        print(
            f"{key:18s} {m['cos']:>9.5f} {m['max_abs']:>10.4f} {m['mean_abs']:>10.4f} "
            f"{m['l2_a']:>12.3f} {m['l2_b']:>12.3f}{flag}{clip}"
        )

    # ---- extra diagnostics on the first bad cut ----
    if first_bad is None:
        print("\nAll requested cuts match.")
        return

    print(f"\nfirst divergence: {first_bad} ({desc_by_key[first_bad]})")

    if first_bad == "img_emb" and L["img_emb"].ndim == 4 and L["img_emb"].shape[1] == 3:
        print("\ncamera-permutation check (cos of phyai vs lerobot[perm]):")
        for perm in itertools.permutations(range(3)):
            m = metrics(L["img_emb"][:, list(perm)], P["img_emb"])
            print(f"  lerobot order {perm}: cos={m['cos']:.5f}")

    if first_bad in ("prefix_embs", "prefix_hidden"):
        n_img = L["img_emb"].shape[1] * L["img_emb"].shape[2]
        a, b, _ = align(L[first_bad].float(), P[first_bad].float())
        mi = metrics(a[:, :n_img], b[:, :n_img])
        ml = metrics(a[:, n_img:], b[:, n_img:])
        print(f"\n  image segment [:{n_img}]  cos={mi['cos']:.5f}  l2 {mi['l2_a']:.2f} vs {mi['l2_b']:.2f}")
        print(f"  lang  segment [{n_img}:]  cos={ml['cos']:.5f}  l2 {ml['l2_a']:.2f} vs {ml['l2_b']:.2f}")

    # per-sample cosine helps spot batch-element mixups (e.g. padding bugs)
    a, b, _ = align(L[first_bad].float(), P[first_bad].float())
    if a.shape[0] > 1:
        print("\n  per-sample cosine:")
        for i in range(a.shape[0]):
            c = float(F.cosine_similarity(a[i].flatten(), b[i].flatten(), dim=0))
            print(f"    sample {i}: {c:.5f}")


if __name__ == "__main__":
    main()
