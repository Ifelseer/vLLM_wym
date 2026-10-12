"""QuantMethod：线性层量化方法抽象接口。

每个量化方法（fp8 / int8 / awq / gptq）实现本接口，由
:func:`src.quant.get_quant_method` 按 layer 类型构造并注入层内。

层与方法的职责划分::

    层（compute/layers/linear.py）   持有参数、决定 TP 通信语义
                                    （gather / all-reduce / input_is_parallel）
    方法（quant/methods/*）         创建量化参数与 scale buffer、实现
                                    weight_loader（TP/EP 切分 + 量化）、
                                    forward（量化 GEMM kernel 分派）

因此 ``forward`` 只负责「量化矩阵乘」，输出未做 TP 通信，由层接着处理。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from ..config import QuantConfig

# 层类型 → QuantMethod 构造参数语义
LAYER_KINDS = (
    "replicated",     # ReplicatedLinear：整份复制，无 TP 切分
    "column",         # ColumnParallelLinear：按输出维（dim=0）切
    "row",            # RowParallelLinear：按输入维（dim=1）切
    "merged_column",  # MergedColumnParallelLinear：多段按输出维切
    "qkv",            # QKVParallelLinear：q/k/v 三段按输出维切
    "experts_w13",    # MoE w13：[E, 2i, h]，按 dim=1 切（TP）+ EP 过滤
    "experts_w2",     # MoE w2：[E, h, i]，按 dim=2 切（TP）+ EP 过滤
)


class QuantMethod(ABC):
    """线性层量化方法基类。"""

    def __init__(self, config: QuantConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ #
    # 参数创建
    # ------------------------------------------------------------------ #

    @abstractmethod
    def create_weights(
        self,
        layer: nn.Module,
        *,
        out_size: int,
        in_size: int,
        tp_size: int,
        tp_rank: int,
        merged_sizes: list[int] | None = None,
    ) -> None:
        """在 layer 上创建量化参数（替代原 ``nn.Parameter`` 权重）。

        约定：
        - 数据参数命名为 ``layer.weight``（dtype 随方法：fp8/int8 packed/int32）。
        - scale 参数命名为 ``layer.weight_scale``（block/per-channel/tensor）。
        - 每个参数通过 ``setattr(param, "weight_loader", ...)`` 绑定加载器，
          模型 ``load_weights`` 的既有路由（按参数名查表）无需改动。
        - 需要额外 buffer（如 g_idx）时用 ``register_buffer(..., persistent=False)``。
        """

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #

    @abstractmethod
    def forward(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        """量化 GEMM：``x @ W^T``（含 scale 融合），输出 dtype 与激活一致。

        不做 TP 通信（gather / all-reduce 由调用层处理）。
        """

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #

    def _bind_loader(self, param: nn.Parameter, loader) -> None:
        setattr(param, "weight_loader", loader)

    @staticmethod
    def _bind_default_loader(param: nn.Parameter) -> None:
        """无 TP 切分语义的参数（如 per-tensor scale）用直接拷贝。"""
        setattr(param, "weight_loader", lambda p, w, *a: p.data.copy_(w))

    def extra_params(self, layer: nn.Module) -> list[nn.Parameter]:
        """除 weight / weight_scale 外的附加参数（供测试与断点调试）。"""
        return []


def bind_loader(param: nn.Parameter, loader) -> None:
    """把 weight_loader 挂到参数上（与 linear.py 既有约定一致）。"""
    setattr(param, "weight_loader", loader)


def default_loader(param: nn.Parameter, loaded_weight: torch.Tensor, *args) -> None:
    param.data.copy_(loaded_weight)
