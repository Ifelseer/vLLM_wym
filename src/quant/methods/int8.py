"""Int8Method：INT8 W8A8 量化方法（per-token 动态激活 × per-channel 静态权重）。

覆盖 2D 线性层（``replicated`` / ``column`` / ``row`` / ``merged_column`` /
``qkv``）；MoE 3D 专家的 INT8 未实现——:class:`src.moe.experts.Experts` 对
int8 显式 raise（MoE 模型请用 fp8 / awq / gptq）。

权重加载两态：
- **预量化 checkpoint**（compressed-tensors int8）：``*.weight`` 为 int8、
  ``*.weight_scale`` 为 fp32 ``[out]``，按 TP 规则切片直写。
  带 ``input_scale`` 的 SmoothQuant checkpoint 显式拒绝（其权重已按校准
  统计量预缩放，直接用动态激活量化会错；需先做 scale 吸收，未实现）。
- **在线量化**（``quantize_at_load``）：BF16/FP16 权重先写 staging buffer，
  全部 shard 到齐后 :meth:`finalize` 统一 per-channel 量化并释放 staging。

TP 切分语义：per-channel scale 随**输出维**切（column/merged/qkv），
row 并行只切输入维故 scale 不切。
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn

from ..config import QuantConfig
from ..kernels.int8_gemm import (
    INT8_DTYPE,
    quantize_per_channel,
    w8a8_int8_matmul,
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


class Int8Method(QuantMethod):
    """INT8 W8A8；``forward`` 只做量化 GEMM，TP 通信由调用层处理。"""

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
            raise ValueError(f"Int8Method does not support kind {kind!r}")
        if config.weight_scale != "per_channel":
            raise ValueError(
                f"Int8Method expects per_channel weight scale, got "
                f"{config.weight_scale!r}"
            )
        self.kind = kind
        self.out_size = out_size  # rank 本地输出维
        self.in_size = in_size    # rank 本地输入维
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.merged_sizes = merged_sizes
        self.qkv_meta = qkv_meta
        self._shard_out, self._shard_in = _KIND_SHARD[kind]

    # ------------------------------------------------------------------ #
    # 参数创建
    # ------------------------------------------------------------------ #

    def create_weights(self, layer: nn.Module) -> None:
        layer.weight = nn.Parameter(
            torch.empty(self.out_size, self.in_size, dtype=INT8_DTYPE),
            requires_grad=False,
        )
        layer.weight_scale = nn.Parameter(
            torch.empty(self.out_size, dtype=torch.float32), requires_grad=False
        )
        # SmoothQuant 的静态激活 scale（``*.input_scale``）不支持：其权重已按
        # 校准统计量预缩放，直接用动态 per-token 激活量化会错。检测放在模型级
        # load_weights 路由（这里无法挂 loader——named_parameters 不含 buffer）。
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

    @staticmethod
    def _reject_input_scale() -> NotImplementedError:
        """SmoothQuant 的 ``input_scale`` 不被支持（见 create_weights 注释）。"""
        return NotImplementedError(
            "SmoothQuant int8 checkpoints (with input_scale) are not "
            "supported: their weights are pre-scaled by calibrated input "
            "scales, which would be incorrect with dynamic per-token "
            "activation quantization. Use a compressed-tensors int8 "
            "checkpoint (no input_scale) or online quantization "
            "(quantization=int8 on a BF16 model)."
        )

    # ------------------------------------------------------------------ #
    # 权重加载：数据
    # ------------------------------------------------------------------ #

    def _shard_data(self, loaded: torch.Tensor, shard_id) -> torch.Tensor:
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

    def _load_weight(self, param, loaded_weight, shard_id=None, *, layer) -> None:
        if loaded_weight.dtype == INT8_DTYPE:
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
            f"int8 layer got {loaded_weight.dtype} weights but the checkpoint is "
            "not marked quantize_at_load; pass a pre-quantized int8 checkpoint "
            "or quantization=int8 with a BF16 model"
        )

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
        if self.kind in ("replicated", "column", "row"):
            staging.copy_(piece)
            return
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
    # 权重加载：scale（预量化 checkpoint，per-channel [out]）
    # ------------------------------------------------------------------ #

    def _load_scale(self, param, loaded_scale, shard_id=None, *, layer) -> None:
        if self.config.quantize_at_load:
            return  # 在线量化无独立 scale 张量，finalize 统一生成
        if loaded_scale.dtype != torch.float32:
            loaded_scale = loaded_scale.float()
        r, tp = self.tp_rank, self.tp_size
        if self.kind == "replicated":
            param.data.copy_(loaded_scale.reshape(-1))
            return
        if self.kind == "column":
            go = param.shape[0]
            param.data.copy_(loaded_scale[r * go : (r + 1) * go])
            return
        if self.kind == "row":
            # row 并行只切输入维 → per-output scale 不切
            param.data.copy_(loaded_scale.reshape(-1))
            return
        if self.kind == "merged_column":
            seg_go = self.merged_sizes[shard_id]
            off = sum(self.merged_sizes[:shard_id])
            param.data[off : off + seg_go].copy_(
                loaded_scale[r * seg_go : (r + 1) * seg_go]
            )
            return
        if self.kind == "qkv":
            heads_total, kv_total, head_dim, replicas = self.qkv_meta
            q_go = heads_total * head_dim // tp
            kv_go = kv_total * head_dim // tp
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
        """staging → per-channel 量化 → weight / weight_scale，释放 staging。"""
        staging = getattr(layer, "_quant_staging", None)
        if staging is None:
            return
        q, s = quantize_per_channel(staging)
        layer.weight.data.copy_(q)
        layer.weight_scale.data.copy_(s)
        layer._quant_staging = None

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        out = w8a8_int8_matmul(x2, layer.weight, layer.weight_scale)
        return out.view(*orig_shape[:-1], out.shape[-1])


def create(
    config: QuantConfig,
    *,
    kind: str,
    out_size: int,
    in_size: int,
    tp_size: int,
    tp_rank: int,
    prefix: str = "",
    merged_sizes: list[int] | None = None,
    qkv_meta: tuple[int, int, int, int] | None = None,
) -> Int8Method:
    """注册表入口（``methods/__init__.make_method`` 调用）。"""
    return Int8Method(
        config,
        kind=kind,
        out_size=out_size,
        in_size=in_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        merged_sizes=merged_sizes,
        qkv_meta=qkv_meta,
    )
