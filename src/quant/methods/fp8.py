"""Fp8Method：FP8 W8A8 量化方法（per-tensor / per-block 128×128）。

覆盖 2D 线性层（``replicated`` / ``column`` / ``row`` / ``merged_column`` /
``qkv``）；MoE 3D 专家的 FP8 在 ``src/moe/experts.py`` 内按同族 scale 语义
单独实现（权重布局不同）。

权重加载两态：
- **预量化 checkpoint**：``*.weight`` 为 fp8、``*.weight_scale`` 为 fp32
  网格，按 TP 规则切片直写。
- **在线量化**（``quantize_at_load``）：BF16/FP16 权重先写 staging buffer，
  全部 shard 到齐后 :meth:`finalize` 统一 per-block 量化并释放 staging。

per-tensor 在线量化的 scale 按 **rank 本地权重** 计算（各 rank 独立、
内部自洽）；预量化 checkpoint 的 scale 一律以 checkpoint 为准。
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn

from ..config import QuantConfig
from ..kernels.fp8_gemm import (
    FP8_DTYPE,
    quantize_fp8_per_block,
    w8a8_block_fp8_matmul,
    w8a8_tensor_fp8_matmul,
)
from .base import QuantMethod

# kind → (out 维被 TP 切分, in 维被 TP 切分)
_KIND_SHARD = {
    "replicated": (False, False),
    "column": (True, False),
    "merged_column": (True, False),
    "qkv": (True, False),
    "row": (False, True),
}

_SCALE = 128  # scale 网格边长


class Fp8Method(QuantMethod):
    """FP8 W8A8；``forward`` 只做量化 GEMM，TP 通信由调用层处理。"""

    def __init__(
        self,
        config: QuantConfig,
        *,
        kind: str,
        out_size: int,
        in_size: int,
        tp_size: int,
        tp_rank: int,
        merged_sizes: list[int] | None = None,
        qkv_meta: tuple[int, int, int, int] | None = None,
    ) -> None:
        super().__init__(config)
        if kind not in _KIND_SHARD:
            raise ValueError(f"Fp8Method does not support kind {kind!r}")
        self.kind = kind
        self.out_size = out_size  # rank 本地输出维
        self.in_size = in_size    # rank 本地输入维
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        # per-partition 段尺寸（merged_column 的 gate/up、qkv 的 q/kv）
        self.merged_sizes = merged_sizes
        # (num_heads_total, num_kv_heads_total, head_dim, num_kv_head_replicas)
        self.qkv_meta = qkv_meta
        self.per_block = config.weight_scale == "per_block_128x128"
        self._shard_out, self._shard_in = _KIND_SHARD[kind]

    # ------------------------------------------------------------------ #
    # scale 网格几何
    # ------------------------------------------------------------------ #

    @property
    def _grid(self) -> tuple[int, int]:
        if self.per_block:
            return (
                (self.out_size + _SCALE - 1) // _SCALE,
                (self.in_size + _SCALE - 1) // _SCALE,
            )
        return (1, 1)

    def _seg_grid(self, shard_id) -> tuple[int, int]:
        """(段在全局网格中的行数, 段在本地网格中的行偏移)。"""
        if self.kind == "merged_column":
            seg_local = self.merged_sizes[shard_id]
            seg_go = (seg_local * self.tp_size + _SCALE - 1) // _SCALE
            off = sum(
                (s + _SCALE - 1) // _SCALE for s in self.merged_sizes[:shard_id]
            )
            return seg_go, off
        if self.kind == "qkv":
            heads_total, kv_total, head_dim, replicas = self.qkv_meta
            if shard_id == "q":
                seg_go = (heads_total * head_dim + _SCALE - 1) // _SCALE
                return seg_go, 0
            kv_go = (kv_total * head_dim + _SCALE - 1) // _SCALE
            return kv_go, None  # q/k/v 偏移由调用方按 shard_id 计算
        raise ValueError(f"kind {self.kind!r} has no segments")

    # ------------------------------------------------------------------ #
    # 参数创建
    # ------------------------------------------------------------------ #

    def create_weights(self, layer: nn.Module) -> None:
        go, gi = self._grid
        layer.weight = nn.Parameter(
            torch.empty(self.out_size, self.in_size, dtype=FP8_DTYPE)
        )
        layer.weight_scale = nn.Parameter(
            torch.empty(go, gi, dtype=torch.float32)
        )
        # 在线量化 staging：首次收到 BF16 权重时按权重 dtype 懒创建
        layer.register_buffer("_quant_staging", None, persistent=False)
        setattr(layer.weight, "weight_loader", partial(self._load_weight, layer=layer))
        setattr(
            layer.weight_scale,
            "weight_loader",
            partial(self._load_scale, layer=layer),
        )

    def extra_params(self, layer: nn.Module) -> list[nn.Parameter]:
        return [layer.weight, layer.weight_scale]

    # ------------------------------------------------------------------ #
    # 权重加载：数据
    # ------------------------------------------------------------------ #

    def _shard_data(
        self, loaded: torch.Tensor, shard_id
    ) -> torch.Tensor:
        """按 kind 取本 rank 的数据切片（与 linear.py 的切分语义一致）。"""
        r, tp = self.tp_rank, self.tp_size
        if self.kind == "replicated":
            return loaded
        if self.kind == "column":
            return loaded.narrow(0, r * self.out_size, self.out_size)
        if self.kind == "row":
            return loaded.narrow(1, r * self.in_size, self.in_size)
        if self.kind == "merged_column":
            seg = self.merged_sizes[shard_id]
            return loaded.narrow(0, r * seg, seg)
        if self.kind == "qkv":
            heads_total, kv_total, head_dim, replicas = self.qkv_meta
            if shard_id == "q":
                q_local = heads_total * head_dim // tp
                return loaded.narrow(0, r * q_local, q_local)
            kv_local = kv_total * head_dim // tp
            head_idx = r // replicas
            return loaded.narrow(0, head_idx * kv_local, kv_local)
        raise ValueError(f"kind {self.kind!r} has no data shard rule")

    def _load_weight(self, param, loaded_weight, shard_id=None, *, layer) -> None:
        if loaded_weight.dtype == FP8_DTYPE:
            piece = self._shard_data(loaded_weight, shard_id)
            off = self._seg_offset(shard_id)
            if off is None:
                param.data.copy_(piece)
            else:
                param.data[off : off + piece.shape[0]].copy_(piece)
            return
        if self.config.quantize_at_load:
            self._write_staging(layer, loaded_weight, shard_id)
            return
        raise ValueError(
            f"fp8 layer got {loaded_weight.dtype} weights but the checkpoint is "
            "not marked quantize_at_load; pass a pre-quantized checkpoint or "
            "quantization=fp8 with a BF16 model"
        )

    def _seg_offset(self, shard_id) -> int | None:
        """段起点（本地参内行偏移）；非分段 kind 返回 None。"""
        if self.kind == "merged_column":
            return sum(self.merged_sizes[:shard_id])
        if self.kind == "qkv":
            heads_total, kv_total, head_dim, _ = self.qkv_meta
            q_local = heads_total * head_dim // self.tp_size
            kv_local = kv_total * head_dim // self.tp_size
            if shard_id == "q":
                return 0
            if shard_id == "k":
                return q_local
            return q_local + kv_local
        return None

    def _write_staging(self, layer, loaded_weight, shard_id) -> None:
        if layer._quant_staging is None:
            layer.register_buffer(
                "_quant_staging",
                torch.zeros(
                    self.out_size, self.in_size, dtype=loaded_weight.dtype
                ),
                persistent=False,
            )
        staging = layer._quant_staging
        piece = self._shard_data(loaded_weight, shard_id)
        if self.kind == "replicated" or self.kind in ("column", "row"):
            staging.copy_(piece)
            return
        # merged / qkv：按段写入本地 staging 的对应行块
        if self.kind == "merged_column":
            off = sum(self.merged_sizes[:shard_id])
            staging[off : off + piece.shape[0]].copy_(piece)
            return
        heads_total, kv_total, head_dim, replicas = self.qkv_meta
        q_local = heads_total * head_dim // self.tp_size
        kv_local = kv_total * head_dim // self.tp_size
        if shard_id == "q":
            staging[:q_local].copy_(piece)
        elif shard_id == "k":
            staging[q_local : q_local + kv_local].copy_(piece)
        else:
            staging[q_local + kv_local :].copy_(piece)

    # ------------------------------------------------------------------ #
    # 权重加载：scale（预量化 checkpoint）
    # ------------------------------------------------------------------ #

    def _load_scale(self, param, loaded_scale, shard_id=None, *, layer) -> None:
        if self.config.quantize_at_load:
            return  # 在线量化无独立 scale 张量，finalize 统一生成
        if loaded_scale.dtype != torch.float32:
            loaded_scale = loaded_scale.float()
        r, tp = self.tp_rank, self.tp_size
        if self.kind == "replicated":
            param.data.copy_(loaded_scale.view(param.shape))
            return
        if self.kind == "column":
            go = param.shape[0]
            param.data.copy_(loaded_scale[r * go : (r + 1) * go])
            return
        if self.kind == "row":
            gi = param.shape[1]
            param.data.copy_(loaded_scale[:, r * gi : (r + 1) * gi])
            return
        if self.kind == "merged_column":
            seg_go, off = self._seg_grid(shard_id)
            param.data[off : off + seg_go].copy_(
                loaded_scale[r * seg_go : (r + 1) * seg_go]
            )
            return
        if self.kind == "qkv":
            heads_total, kv_total, head_dim, replicas = self.qkv_meta
            q_go = (heads_total * head_dim // tp + _SCALE - 1) // _SCALE
            kv_go = (kv_total * head_dim // tp + _SCALE - 1) // _SCALE
            if shard_id == "q":
                param.data[:q_go].copy_(loaded_scale[r * q_go : (r + 1) * q_go])
                return
            head_idx = r // replicas
            if shard_id == "k":
                param.data[q_go : q_go + kv_go].copy_(
                    loaded_scale[head_idx * kv_go : (head_idx + 1) * kv_go]
                )
            else:
                param.data[q_go + kv_go :].copy_(
                    loaded_scale[head_idx * kv_go : (head_idx + 1) * kv_go]
                )
            return
        raise ValueError(f"kind {self.kind!r} has no scale shard rule")

    # ------------------------------------------------------------------ #
    # 在线量化收尾
    # ------------------------------------------------------------------ #

    def finalize(self, layer: nn.Module) -> None:
        """staging → per-block 量化 → weight / weight_scale，释放 staging。"""
        staging = getattr(layer, "_quant_staging", None)
        if staging is None:
            return
        if self.per_block:
            q, s = quantize_fp8_per_block(staging)
            layer.weight_scale.data.copy_(s)
        else:
            amax = staging.abs().amax().clamp(min=1e-12)
            s = (amax / 448.0).to(torch.float32)
            q = (staging / s).to(FP8_DTYPE)
            layer.weight_scale.data.fill_(s.item())
        layer.weight.data.copy_(q)
        layer._quant_staging = None

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        if self.per_block:
            out = w8a8_block_fp8_matmul(x2, layer.weight, layer.weight_scale)
        else:
            out = w8a8_tensor_fp8_matmul(x2, layer.weight, layer.weight_scale)
        return out.view(*orig_shape[:-1], out.shape[-1])


def create(
    quant: QuantConfig,
    *,
    kind: str,
    out_size: int,
    in_size: int,
    tp_size: int,
    tp_rank: int,
    prefix: str = "",
    merged_sizes: list[int] | None = None,
    qkv_meta: tuple[int, int, int, int] | None = None,
) -> Fp8Method:
    """注册表工厂：构造 Fp8Method（2D 层）。"""
    return Fp8Method(
        quant,
        kind=kind,
        out_size=out_size,
        in_size=in_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        merged_sizes=merged_sizes,
        qkv_meta=qkv_meta,
    )
