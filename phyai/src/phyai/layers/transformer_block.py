"""NoStateTransformerBlock — a highly configurable prefill-only transformer block.

One block, two normalisation topologies, optional Q/K head-dim norm, no
KV cache:

* **Pre-norm** (Llama / Qwen2 / Qwen2.5 / Qwen3 / Mistral / Phi3 /
  SigLIP encoder)::

      h = x + attn(input_norm(x))
      y = h + mlp(pre_ff_norm(h))

* **Sandwich norm** (``sandwich_norm=True``; Gemma2 / Gemma3)::

      h = x + post_attn_norm(attn(input_norm(x)))
      y = h + post_ff_norm(mlp(pre_ff_norm(h)))

When ``attn_qk_norm=True`` the block additionally normalises Q and K
on the per-head ``head_dim`` axis after the QKV projection and before
RoPE — used by Gemma3 (gemma-style ``(1+w)`` RMS) and Qwen3 (standard
RMS).

Knob matrix
-----------

================  ========================================================
group             knobs
================  ========================================================
norm              ``norm_type`` (rmsnorm / gemma_rmsnorm / layernorm),
                  ``norm_eps``, ``norm_bias`` (LN only), ``norm_backend``,
                  ``sandwich_norm`` (off → 2 norms; on → 4 norms)
attention         ``num_heads``, ``num_kv_heads``, ``head_dim``,
                  ``attn_causal``, ``attn_sliding_window``,
                  ``attn_logits_soft_cap``, ``attn_scale``,
                  ``attn_bias`` (q/k/v), ``attn_out_bias`` (o-proj),
                  ``attn_qk_norm`` (per-head Q/K norm), ``attn_backend``
RoPE              ``rope`` (a :class:`RotaryEmbedding` instance shared
                  across layers, or ``None`` for vision encoders)
MLP               ``intermediate_size``, ``mlp_gated``, ``mlp_activation``,
                  ``mlp_bias``
TP                ``axis`` / ``sp_axis`` / ``mesh``
quant             ``spec_qkv`` / ``spec_o`` / ``spec_mlp_in`` /
                  ``spec_mlp_out`` (per-leg :class:`WeightSpec`)
HF naming         **required**: ``norm_hf_names`` (per-position),
                  ``attn_out_hf_name`` (Llama/Gemma ``"o_proj"`` vs SigLIP
                  ``"out_proj"``); optional: ``attn_qkv_hf_names``,
                  ``mlp_gated_hf_names``
================  ========================================================

The block is a **structural primitive** — HF naming conventions belong
to the model that uses it. The two highly-divergent knobs are required
arguments; the two truly-universal-ish ones (Q/K/V, gated MLP) keep
sensible defaults that match every modern decoder phyai targets (and
SigLIP, for the QKV side).

Forward
-------
``forward(x, position_ids=None, cu_seqlens_q=None, cu_seqlens_kv=None) -> y``

* ``x``: ``(B, S, hidden_size)`` for padded batches, ``(nnz, hidden_size)``
  for ragged. Output preserves the leading shape.
* ``position_ids``: required when ``rope`` is set — ``(B, S)`` / ``(S,)``
  for padded, ``(nnz,)`` for ragged.
* ``cu_seqlens_q`` / ``cu_seqlens_kv``: int32 ``(B+1,)`` for ragged input,
  passed through to :class:`NoStateAttention`.

Norm-position keys
------------------

================  ===============================  ===============================
position           pre-norm                          sandwich norm
================  ===============================  ===============================
``input_norm``     ✓                                ✓
``post_attn_norm`` —                                ✓
``pre_ff_norm``    ✓                                ✓
``post_ff_norm``   —                                ✓
================  ===============================  ===============================

``norm_hf_names`` must contain *exactly* the keys for the chosen
topology (``{input_norm, pre_ff_norm}`` for pre-norm,
``{input_norm, post_attn_norm, pre_ff_norm, post_ff_norm}`` for
sandwich) — extra or missing keys are rejected at construction. When
``attn_qk_norm=True``, the q/k norms are HF-named ``q_norm`` / ``k_norm``
inside ``self_attn`` (universal across Gemma3 / Qwen3).

The attention sub-prefix is always ``self_attn`` and the MLP sub-prefix
is always ``mlp``.

Limitations
-----------
* Prefill only. No KV cache, no radix.
* No append-prefill mode. Q and K must share token count.
* No fused FP8 RoPE / Q/K quant.
* No cross-attention (decoder-only / encoder-only).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

import torch
import torch.nn as nn

from phyai.layers.attention.no_state_attention import NoStateAttention
from phyai.layers.layer_norm import GemmaRMSNorm, LayerNorm, RMSNorm
from phyai.layers.linear.layers import (
    QKVParallelLinear,
    RowParallelLinear,
)
from phyai.layers.mlp.dense_mlp import DenseMLP


# HuggingFace de-facto norm naming defaults. Llama / Qwen2 / Qwen3 / Mistral /
# Phi3 / Olmo all use ``input_layernorm`` + ``post_attention_layernorm``;
# Gemma2 / Gemma3 add ``pre_feedforward_layernorm`` + ``post_feedforward_layernorm``
# for the sandwich-norm topology. The mapping is ``phyai_slot -> hf_default``.
# The user-facing override dict is keyed by ``hf_default`` (recognisable to anyone
# coming from a HuggingFace modeling file) and gets resolved to the slot internally.
_HF_NORM_NAMES_PRE: Mapping[str, str] = MappingProxyType(
    {
        "input_norm": "input_layernorm",
        "pre_ff_norm": "post_attention_layernorm",
    }
)
_HF_NORM_NAMES_SANDWICH: Mapping[str, str] = MappingProxyType(
    {
        "input_norm": "input_layernorm",
        "post_attn_norm": "post_attention_layernorm",
        "pre_ff_norm": "pre_feedforward_layernorm",
        "post_ff_norm": "post_feedforward_layernorm",
    }
)

_VALID_NORM_TYPES: tuple[str, ...] = ("rmsnorm", "gemma_rmsnorm", "layernorm")


def _make_norm(
    norm_type: str,
    hidden_size: int,
    eps: float,
    *,
    bias: bool,
    backend: str,
    dtype: torch.dtype | None,
    prefix: str,
) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(
            hidden_size, eps=eps, backend=backend, dtype=dtype, prefix=prefix
        )
    if norm_type == "gemma_rmsnorm":
        return GemmaRMSNorm(
            hidden_size, eps=eps, backend=backend, dtype=dtype, prefix=prefix
        )
    if norm_type == "layernorm":
        return LayerNorm(
            hidden_size,
            eps=eps,
            backend=backend,
            bias=bias,
            dtype=dtype,
            prefix=prefix,
        )
    raise ValueError(
        f"Unknown norm_type {norm_type!r}; expected one of {_VALID_NORM_TYPES!r}."
    )


class NoStateTransformerBlock(nn.Module):
    """Configurable pre-norm / sandwich-norm transformer block (no KV cache).

    See module docstring for the full topology and knob list. The class
    composes existing phyai primitives:
    :class:`QKVParallelLinear` →
    (optional Q/K norm) →
    (optional :class:`RotaryEmbedding`) →
    :class:`NoStateAttention` →
    :class:`RowParallelLinear` (output proj) →
    :class:`DenseMLP`, with norms inserted per the chosen topology.
    """

    def __init__(
        self,
        # ---- Core dims -------------------------------------------------- #
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        *,
        # ---- HF naming (defaults match Llama / Qwen / Gemma conventions) ---- #
        norm_hf_names: Mapping[str, str] | None = None,
        attn_out_hf_name: str = "o_proj",
        # ---- Attention dims --------------------------------------------- #
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        # ---- Topology --------------------------------------------------- #
        sandwich_norm: bool = False,
        # ---- Attention behaviour --------------------------------------- #
        attn_causal: bool = True,
        attn_sliding_window: int | None = None,
        attn_logits_soft_cap: float | None = None,
        attn_scale: float | None = None,
        attn_bias: bool = False,
        attn_out_bias: bool | None = None,
        attn_qk_norm: bool = False,
        attn_backend: str = "flashinfer",
        # ---- RoPE ------------------------------------------------------- #
        rope: nn.Module | None = None,
        # ---- MLP -------------------------------------------------------- #
        mlp_gated: bool = True,
        mlp_activation: str = "silu",
        mlp_bias: bool = False,
        # ---- Norm ------------------------------------------------------- #
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-6,
        norm_bias: bool = False,
        norm_backend: str = "flashinfer",
        # ---- TP / mesh -------------------------------------------------- #
        axis: str = "tp",
        sp_axis: str | None = None,
        mesh: str = "model",
        # ---- Quant specs ------------------------------------------------ #
        spec_qkv: object | None = None,
        spec_o: object | None = None,
        spec_mlp_in: object | None = None,
        spec_mlp_out: object | None = None,
        # ---- Optional HF naming overrides (defaults are universal) ----- #
        attn_qkv_hf_names: Mapping[str, str] | None = None,
        mlp_gated_hf_names: tuple[str, str] = ("gate_proj", "up_proj"),
        # ---- Misc ------------------------------------------------------- #
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        if norm_type not in _VALID_NORM_TYPES:
            raise ValueError(
                f"Unknown norm_type {norm_type!r}; expected one of "
                f"{_VALID_NORM_TYPES!r}."
            )
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if head_dim is None:
            if hidden_size % num_heads != 0:
                raise ValueError(
                    f"hidden_size={hidden_size} not divisible by "
                    f"num_heads={num_heads}; pass head_dim explicitly."
                )
            head_dim = hidden_size // num_heads
        if attn_out_bias is None:
            attn_out_bias = attn_bias

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.sandwich_norm = sandwich_norm
        self.norm_type = norm_type
        self.attn_qk_norm_enabled = attn_qk_norm
        self.prefix = prefix
        self.rope = rope

        # Naming knobs. ``norm_hf_names=None`` uses HF defaults for the
        # chosen topology (covers Llama / Qwen / Gemma / Mistral / Phi3).
        # When passed, keys are HF default names (``"input_layernorm"``
        # etc.) so the override dict reads naturally; values are the
        # actual HF source names in this checkpoint.
        self.attn_out_hf_name = attn_out_hf_name
        self.attn_qkv_hf_names = (
            dict(attn_qkv_hf_names) if attn_qkv_hf_names is not None else None
        )
        self.mlp_gated_hf_names = tuple(mlp_gated_hf_names)
        self._norm_hf_names = self._validate_norm_hf_names(norm_hf_names, sandwich_norm)

        # Sub-prefixes
        attn_prefix = f"{prefix}.self_attn" if prefix else "self_attn"
        mlp_prefix = f"{prefix}.mlp" if prefix else "mlp"

        # ---- Norms (hidden_size) --------------------------------------- #
        norm_kwargs: dict[str, Any] = dict(
            hidden_size=hidden_size,
            eps=norm_eps,
            bias=norm_bias,
            backend=norm_backend,
            dtype=params_dtype,
        )
        self.input_norm = _make_norm(
            norm_type, prefix=self._norm_prefix("input_norm"), **norm_kwargs
        )
        self.pre_ff_norm = _make_norm(
            norm_type, prefix=self._norm_prefix("pre_ff_norm"), **norm_kwargs
        )
        if sandwich_norm:
            self.post_attn_norm = _make_norm(
                norm_type,
                prefix=self._norm_prefix("post_attn_norm"),
                **norm_kwargs,
            )
            self.post_ff_norm = _make_norm(
                norm_type,
                prefix=self._norm_prefix("post_ff_norm"),
                **norm_kwargs,
            )
        else:
            self.post_attn_norm = None
            self.post_ff_norm = None

        # ---- Attention: QKV → (Q/K norm) → (RoPE) → attn → O ----------- #
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            axis=axis,
            sp_axis=sp_axis,
            gather_output=False,
            bias=attn_bias,
            params_dtype=params_dtype,
            spec=spec_qkv,
            hf_legs=self.attn_qkv_hf_names,
            mesh=mesh,
            prefix=f"{attn_prefix}.qkv_proj",
        )
        tp_size = self.qkv_proj.tp_size
        self.q_heads_local = num_heads // tp_size
        self.kv_heads_local = max(1, num_kv_heads // tp_size)

        # Optional per-head Q/K norm (Gemma3 / Qwen3). Operates on
        # ``head_dim`` (each Q/K head normalised independently). Names
        # ``q_norm`` / ``k_norm`` are universal across the families that
        # use this hook today.
        if attn_qk_norm:
            self.q_norm = _make_norm(
                norm_type,
                hidden_size=head_dim,
                eps=norm_eps,
                bias=norm_bias,
                backend=norm_backend,
                dtype=params_dtype,
                prefix=f"{attn_prefix}.q_norm",
            )
            self.k_norm = _make_norm(
                norm_type,
                hidden_size=head_dim,
                eps=norm_eps,
                bias=norm_bias,
                backend=norm_backend,
                dtype=params_dtype,
                prefix=f"{attn_prefix}.k_norm",
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.attn = NoStateAttention(
            num_heads=self.q_heads_local,
            head_dim=head_dim,
            num_kv_heads=self.kv_heads_local,
            scale=attn_scale,
            causal=attn_causal,
            sliding_window=attn_sliding_window,
            logits_soft_cap=attn_logits_soft_cap,
            backend=attn_backend,
        )

        self.o_proj = RowParallelLinear(
            in_features=num_heads * head_dim,
            out_features=hidden_size,
            axis=axis,
            sp_axis=sp_axis,
            input_is_parallel=True,
            reduce_results=True,
            bias=attn_out_bias,
            params_dtype=params_dtype,
            spec=spec_o,
            mesh=mesh,
            prefix=f"{attn_prefix}.{attn_out_hf_name}",
        )

        # ---- MLP -------------------------------------------------------- #
        self.mlp = DenseMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=mlp_activation,
            gated=mlp_gated,
            bias=mlp_bias,
            axis=axis,
            sp_axis=sp_axis,
            params_dtype=params_dtype,
            spec_in=spec_mlp_in,
            spec_out=spec_mlp_out,
            gated_hf_legs=self.mlp_gated_hf_names,
            mesh=mesh,
            prefix=mlp_prefix,
        )

    # ------------------------------------------------------------------ #
    # Naming validation                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_norm_hf_names(
        overrides: Mapping[str, str] | None, sandwich: bool
    ) -> dict[str, str]:
        """Resolve the actual HF source name for each norm slot.

        ``overrides`` is keyed by **HuggingFace default names**
        (``"input_layernorm"`` etc.) — the same names users see in any
        HF modeling file. A missing entry uses the HF default itself.
        ``None`` means use HF defaults for everything (covers Llama /
        Qwen / Gemma / Mistral / Phi3 — the common case).

        Returns a ``{phyai_slot -> actual_hf_source_name}`` map for
        internal use by :meth:`_norm_prefix`.
        """
        defaults = _HF_NORM_NAMES_SANDWICH if sandwich else _HF_NORM_NAMES_PRE
        # Start with the identity mapping (use the HF default for every slot).
        out = dict(defaults)
        if overrides is None:
            return out
        # User keys are HF default names; flip the mapping so we can find the slot.
        hf_to_slot = {hf_default: slot for slot, hf_default in defaults.items()}
        unknown = set(overrides) - set(hf_to_slot)
        if unknown:
            topo = "sandwich-norm" if sandwich else "pre-norm"
            raise ValueError(
                f"norm_hf_names has unknown keys {sorted(unknown)!r} for "
                f"{topo} topology; expected subset of "
                f"{sorted(hf_to_slot.keys())!r}."
            )
        for hf_default, actual in overrides.items():
            out[hf_to_slot[hf_default]] = actual
        return out

    def _norm_prefix(self, position: str) -> str:
        own = self._norm_hf_names[position]
        return f"{self.prefix}.{own}" if self.prefix else own

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() not in (2, 3):
            raise ValueError(
                f"x must be 2-D (ragged) or 3-D (padded), got shape "
                f"{tuple(x.shape)}."
            )
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"x last dim {x.shape[-1]} != hidden_size={self.hidden_size}."
            )
        if self.rope is not None and position_ids is None:
            raise ValueError("rope is set but no position_ids passed to forward.")

        # ---- Attention sub-block --------------------------------------- #
        residual = x
        h = self.input_norm(x)
        q, k, v = self._qkv_split(h)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q, k, position_ids)
        attn_out = self.attn(
            q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv
        )
        # (..., q_heads_local, head_dim) -> (..., q_heads_local * head_dim)
        attn_out = attn_out.reshape(
            *attn_out.shape[:-2], self.q_heads_local * self.head_dim
        )
        attn_out, _ = self.o_proj(attn_out)
        if self.post_attn_norm is not None:
            attn_out = self.post_attn_norm(attn_out)
        h = residual + attn_out

        # ---- MLP sub-block --------------------------------------------- #
        residual = h
        m = self.pre_ff_norm(h)
        m = self.mlp(m)
        if self.post_ff_norm is not None:
            m = self.post_ff_norm(m)
        return residual + m

    def _qkv_split(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run QKVParallelLinear and split the fused output into (q, k, v).

        Output shapes:
            * 3-D padded ``h (B, S, hidden_size)``:
              q ``(B, S, q_heads_local, head_dim)``,
              k/v ``(B, S, kv_heads_local, head_dim)``.
            * 2-D ragged ``h (nnz, hidden_size)``:
              q ``(nnz, q_heads_local, head_dim)``,
              k/v ``(nnz, kv_heads_local, head_dim)``.
        """
        fused, _ = self.qkv_proj(h)
        q_dim = self.q_heads_local * self.head_dim
        kv_dim = self.kv_heads_local * self.head_dim
        q, k, v = fused.split([q_dim, kv_dim, kv_dim], dim=-1)
        leading = h.shape[:-1]
        q = q.reshape(*leading, self.q_heads_local, self.head_dim)
        k = k.reshape(*leading, self.kv_heads_local, self.head_dim)
        v = v.reshape(*leading, self.kv_heads_local, self.head_dim)
        return q, k, v

    def extra_repr(self) -> str:
        s = (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, "
            f"intermediate_size={self.intermediate_size}, "
            f"norm_type={self.norm_type!r}, sandwich_norm={self.sandwich_norm}"
        )
        if self.attn_qk_norm_enabled:
            s += ", attn_qk_norm=True"
        return s


__all__ = ["NoStateTransformerBlock"]
