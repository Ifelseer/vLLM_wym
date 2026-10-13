"""INT4 weight-only 量化方法（AWQ / GPTQ）。

内部统一 in-major 布局（见 ``kernels/int4_gemm.py`` 的模块文档）：

    qweight : int32 [in, out/8]
    scales  : fp16  [in/G, out]
    zeros   : uint8 [in/G, out]（asym 时）；sym 时不建该参数
    g_idx   : int32 [in]（GPTQ desc_act=true 时；本版本显式拒绝）

前向：``int4_weight_only_matmul``（Triton fused dequant-GEMM，W4A16）。
激活保持模型 dtype（weight-only 不改激活通路），因此与既有 TP 通信、
CUDA Graph、FP8 KV 等全部正交。
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import QuantConfig
from ..kernels.int4_gemm import (
    INT4_SYM_BIAS,
    int4_weight_only_matmul,
    qweight_to_in_major,
    qzeros_to_in_major,
    scales_to_in_major,
    unpack_int4_packed_cols,
)
from .base import QuantMethod, bind_loader

# 每个 int32 打包 4bit 的个数
_PACK = 8


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


class Int4Method(QuantMethod):
    """AWQ / GPTQ 的 weight-only INT4 方法（group_size=128）。"""

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
        prefix: str = "",
    ) -> None:
        super().__init__(config)
        if config.group_size is None:
            raise ValueError("INT4 requires group_size in QuantConfig")
        if config.desc_act:
            raise NotImplementedError(
                "GPTQ desc_act=true (act-order / g_idx) is not supported; "
                "use a desc_act=false checkpoint"
            )
        self.kind = kind
        self.out_size = out_size
        self.in_size = in_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.merged_sizes = merged_sizes or []
        self.qkv_meta = qkv_meta
        self.group = int(config.group_size)
        # 零点语义：
        #   AWQ            非对称，零点来自 qzeros（w = (q - z) * s）
        #   GPTQ sym=true  对称 uint4b8（存值 = 量化值 + 8，无 qzeros 张量）
        #   GPTQ sym=false 非对称，零点来自 qzeros
        # 注意从 `sym` 推导而非 `zero_point`——HF 的 GPTQ config 只给 sym，
        # zero_point 会停在 dataclass 默认值，据此判断会走错分支。
        asym = (config.method == "awq") or (not config.sym)
        self.has_zeros = asym
        self.zero_bias = 0.0 if asym else INT4_SYM_BIAS
        if in_size % self.group != 0:
            raise ValueError(
                f"INT4 group_size={self.group} must divide local in_size={in_size}"
            )

    # ------------------------------------------------------------------ #
    # 参数创建
    # ------------------------------------------------------------------ #

    def create_weights(
        self,
        layer: nn.Module,
        *,
        out_size: int | None = None,
        in_size: int | None = None,
        tp_size: int | None = None,
        tp_rank: int | None = None,
        merged_sizes: list[int] | None = None,
    ) -> None:
        out_s = out_size if out_size is not None else self.out_size
        in_s = in_size if in_size is not None else self.in_size
        g = _cdiv(in_s, self.group)
        layer.qweight = nn.Parameter(
            torch.empty(in_s, out_s // _PACK, dtype=torch.int32), requires_grad=False
        )
        layer.scales = nn.Parameter(torch.empty(g, out_s, dtype=torch.float16), requires_grad=False)
        if self.has_zeros:
            layer.qzeros = nn.Parameter(
                torch.zeros(g, out_s, dtype=torch.uint8), requires_grad=False
            )
        else:
            layer.register_parameter("qzeros", None)
        bind_loader(layer.qweight, self._make_loader("qweight"))
        bind_loader(layer.scales, self._make_loader("scales"))
        if self.has_zeros:
            bind_loader(layer.qzeros, self._make_loader("qzeros"))

    def extra_params(self, layer: nn.Module) -> list[nn.Parameter]:
        return [p for p in (layer.qweight, layer.scales, layer.qzeros) if p is not None]

    # ------------------------------------------------------------------ #
    # 加载：归一化 → TP 切分 → 写入
    # ------------------------------------------------------------------ #

    def _make_loader(self, field: str):
        def loader(param, loaded_weight, shard_id=None):
            self._load_field(param, loaded_weight, shard_id, field)

        return loader

    def _full_dims(self) -> tuple[int, int]:
        """checkpoint 中的全局 (in_full, out_full)。"""
        if self.kind in ("column", "merged_column", "qkv"):
            return self.in_size, self.out_size * self.tp_size
        if self.kind == "row":
            return self.in_size * self.tp_size, self.out_size
        return self.in_size, self.out_size  # replicated

    def _checkpoint_dims(self, shard_id) -> tuple[int, int]:
        """本字段所在 checkpoint 张量的全局 (in_full, out_full)。

        fused 层（merged_column / qkv）的 checkpoint 是**按段**存的
        （gate/up 或 q/k/v 各自一个张量），因此维度取段而非融合后总维。
        """
        in_full, out_full = self._full_dims()
        if self.kind == "merged_column":
            return in_full, self.merged_sizes[int(shard_id)] * self.tp_size
        if self.kind == "qkv":
            _heads, kv_heads, head_dim, _r = self.qkv_meta
            seg = self.qkv_meta[0] if shard_id == "q" else kv_heads
            return in_full, seg * head_dim
        return in_full, out_full

    def _segment(self, shard_id):
        """(源段内 tp 偏移, 目标段偏移, 本段本地长度) —— 非分段 kind 返回 None。"""
        if self.kind == "merged_column":
            assert isinstance(shard_id, int), "merged_column expects int shard_id"
            dest = sum(self.merged_sizes[:shard_id])
            seg_full = self.merged_sizes[shard_id] * self.tp_size
            seg_local = self.merged_sizes[shard_id]
            return self.tp_rank * seg_local, dest, seg_local, seg_full
        if self.kind == "qkv":
            heads, kv_heads, head_dim, replicas = self.qkv_meta
            q_full = heads * head_dim
            kv_full = kv_heads * head_dim
            q_local = q_full // self.tp_size
            kv_local = kv_full // self.tp_size
            if shard_id == "q":
                return self.tp_rank * q_local, 0, q_local, q_full
            head = (self.tp_rank // replicas) if replicas > 1 else self.tp_rank
            if shard_id == "k":
                return head * kv_local, q_local, kv_local, kv_full
            return head * kv_local, q_local + kv_local, kv_local, kv_full
        return None

    def _normalize(self, loaded: torch.Tensor, field: str, shard_id) -> torch.Tensor:
        """把 checkpoint 张量归一化到内部 in-major 布局（不切分）。

        qzeros 额外解包为 uint8 [in/G, out]——热路径不再做位运算。
        """
        in_full, out_full = self._checkpoint_dims(shard_id)
        if field == "qweight":
            return qweight_to_in_major(
                loaded,
                in_full=in_full,
                out_full=out_full,
                method=self.config.method,
            )
        if field == "scales":
            return scales_to_in_major(
                loaded, in_full=in_full, out_full=out_full, group_size=self.group
            )
        if field == "qzeros":
            packed = qzeros_to_in_major(
                loaded,
                in_full=in_full,
                out_full=out_full,
                group_size=self.group,
                method=self.config.method,
            )
            return unpack_int4_packed_cols(packed, out_full)  # [in/G, out] uint8
        raise ValueError(f"unknown INT4 field {field!r}")

    def _load_field(self, param, loaded_weight, shard_id, field: str) -> None:
        norm = self._normalize(loaded_weight, field, shard_id)
        if self.kind == "row":
            # 行并行切输入维：qweight 按行切（in），scales/qzeros 按组切
            g_local = self.in_size // self.group
            if field == "qweight":
                piece = norm[
                    self.tp_rank * self.in_size : (self.tp_rank + 1) * self.in_size
                ]
            else:
                piece = norm[self.tp_rank * g_local : (self.tp_rank + 1) * g_local]
            param.data.copy_(piece.to(param.dtype))
            return
        # 输出维按 TP 切（column / merged_column / qkv / replicated）
        seg = self._segment(shard_id)
        if seg is None:
            src_out, dest_out, n_local = self.tp_rank * self.out_size, 0, self.out_size
        else:
            src_out, dest_out, n_local, _seg_full = seg
        if field == "qweight":
            c0, cl = src_out // _PACK, n_local // _PACK
            param.data[
                :, dest_out // _PACK : dest_out // _PACK + cl
            ].copy_(norm[:, c0 : c0 + cl])
            return
        param.data[:, dest_out : dest_out + n_local].copy_(
            norm[:, src_out : src_out + n_local].to(param.dtype)
        )

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        return int4_weight_only_matmul(
            x,
            layer.qweight,
            layer.scales,
            layer.qzeros if self.has_zeros else None,
            group_size=self.group,
            zero_bias=self.zero_bias,
        )


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
) -> Int4Method:
    """注册表入口（``methods/__init__.make_method`` 调用）。"""
    return Int4Method(
        config,
        kind=kind,
        out_size=out_size,
        in_size=in_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        merged_sizes=merged_sizes,
        qkv_meta=qkv_meta,
        prefix=prefix,
    )
