"""Weight-shard loaders.

Each loader knows how to narrow a full on-disk tensor into the per-rank
slice of an ``nn.Parameter``. Every ``allocate`` call on a WeightSpec
attaches an instance of one of these to ``param._loader``, and the
model-loading layer calls ``loader.load_*`` instead of doing string
attribute checks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ColumnShardLoader:
    """ColumnParallelLinear loader — slices along the output dim (dim=0).

    ``output_partition_sizes`` is the per-rank size of each *logical*
    matrix; for plain :class:`ColumnParallelLinear` that's a one-element
    list, for :class:`MergedColumnParallelLinear` it's one entry per
    fused sub-matrix.
    """

    def __init__(
        self,
        *,
        output_partition_sizes: list[int],
        tp_rank: int,
        tp_size: int,
    ) -> None:
        self.output_partition_sizes = output_partition_sizes
        self.tp_rank = tp_rank
        self.tp_size = tp_size

    def load_full(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        """Load a single un-fused weight whose global output dim = sum(sizes) * tp_size."""
        global_out = sum(self.output_partition_sizes) * self.tp_size
        if loaded.shape[0] != global_out:
            raise ValueError(
                f"ColumnShardLoader.load_full: loaded.shape[0]={loaded.shape[0]} "
                f"!= global_out={global_out}"
            )
        per_rank = sum(self.output_partition_sizes)
        sliced = loaded.narrow(0, self.tp_rank * per_rank, per_rank)
        param.data.copy_(sliced)

    def load_shard(
        self,
        param: nn.Parameter,
        loaded: torch.Tensor,
        shard_id: int,
    ) -> None:
        """Load a single logical sub-matrix into its slot of a fused param.

        ``loaded`` is the *global* un-sharded tensor for this one logical
        matrix; we first take the current rank's slice, then write it into
        the fused param at the correct offset.
        """
        if shard_id < 0 or shard_id >= len(self.output_partition_sizes):
            raise IndexError(
                f"shard_id={shard_id} out of range for "
                f"output_partition_sizes={self.output_partition_sizes}"
            )
        per_rank = self.output_partition_sizes[shard_id]
        global_size = per_rank * self.tp_size
        if loaded.shape[0] != global_size:
            raise ValueError(
                f"ColumnShardLoader.load_shard({shard_id}): loaded.shape[0]="
                f"{loaded.shape[0]} != global_size={global_size}"
            )
        offset = sum(self.output_partition_sizes[:shard_id])
        sliced = loaded.narrow(0, self.tp_rank * per_rank, per_rank)
        param.data.narrow(0, offset, per_rank).copy_(sliced)


class RowShardLoader:
    """RowParallelLinear loader — slices along the input dim (dim=1)."""

    def __init__(self, *, tp_rank: int, tp_size: int) -> None:
        self.tp_rank = tp_rank
        self.tp_size = tp_size

    def load_full(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        shard = loaded.shape[1] // self.tp_size
        if shard * self.tp_size != loaded.shape[1]:
            raise ValueError(
                f"RowShardLoader.load_full: in_dim={loaded.shape[1]} "
                f"not divisible by tp_size={self.tp_size}"
            )
        sliced = loaded.narrow(1, self.tp_rank * shard, shard)
        param.data.copy_(sliced)


class QKVShardLoader(ColumnShardLoader):
    """Q/K/V fused loader; ``shard_id ∈ {"q", "k", "v"}``.

    GQA is supported via ``num_kv_replicas``: when ``tp_size`` exceeds the
    number of KV heads, each K/V shard is replicated rather than split.
    """

    _QKV_IDX = {"q": 0, "k": 1, "v": 2}

    def __init__(
        self,
        *,
        q_size: int,
        kv_size: int,
        num_kv_replicas: int,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        super().__init__(
            output_partition_sizes=[q_size, kv_size, kv_size],
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if num_kv_replicas < 1:
            raise ValueError(f"num_kv_replicas must be ≥1, got {num_kv_replicas}")
        self.num_kv_replicas = num_kv_replicas

    def load_qkv(
        self,
        param: nn.Parameter,
        loaded: torch.Tensor,
        shard_id: str,
    ) -> None:
        if shard_id not in self._QKV_IDX:
            raise ValueError(f"shard_id must be one of q/k/v, got {shard_id!r}")
        idx = self._QKV_IDX[shard_id]
        per_rank = self.output_partition_sizes[idx]

        if idx == 0:
            # Q is straightforwardly column-sharded.
            global_size = per_rank * self.tp_size
            if loaded.shape[0] != global_size:
                raise ValueError(
                    f"QKVShardLoader.load_qkv('q'): loaded.shape[0]="
                    f"{loaded.shape[0]} != global_q={global_size}"
                )
            offset = 0
            sliced = loaded.narrow(0, self.tp_rank * per_rank, per_rank)
        else:
            # K/V: with GQA, disk weight is the un-replicated width
            # ``per_rank * tp_size // num_kv_replicas``; we pick our slice
            # from the ``tp_rank // num_kv_replicas`` partition.
            disk_width = per_rank * self.tp_size // self.num_kv_replicas
            if loaded.shape[0] != disk_width:
                raise ValueError(
                    f"QKVShardLoader.load_qkv({shard_id!r}): loaded.shape[0]="
                    f"{loaded.shape[0]} != disk_width={disk_width}"
                )
            offset = sum(self.output_partition_sizes[:idx])
            kv_rank = self.tp_rank // self.num_kv_replicas
            sliced = loaded.narrow(0, kv_rank * per_rank, per_rank)

        param.data.narrow(0, offset, per_rank).copy_(sliced)
