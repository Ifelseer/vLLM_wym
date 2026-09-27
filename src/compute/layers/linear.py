from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from ...distributed import (
    divide,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)

# ==============================================================================
# 模块说明：
# 本模块实现了LLM中常用的张量并行（Tensor Parallelism, TP）线性层组件。
# 主要包含：
# 1. ColumnParallelLinear: 列并行（按输出维度切分权重），常用于 FFN 的 gate/up 投影或 Attention 的 QKV 投影。
# 2. RowParallelLinear: 行并行（按输入维度切分权重），常用于 FFN 的 down 投影或 Attention 的 out 投影。
# 3. QKVParallelLinear: 融合的 QKV 投影层，支持 GQA/MQA 注意力机制的 Head 切分。
# 4. MergedColumnParallelLinear: 多列融合并行（如将 gate 和 up 融合），用于减少 Kernel Launch 开销。
# ==============================================================================



def _tp_meta(tp_size: int | None) -> tuple[int, int]:
    tp = get_tp_group()
    if tp_size is None:
        return tp.size, tp.group_rank
    if tp_size == 1:
        return 1, 0
    if tp_size != tp.size:
        raise ValueError(f"tp_size={tp_size} does not match process group {tp.size}")
    return tp.size, tp.group_rank


def _shard(
    weight: torch.Tensor,
    dim: int,
    *,
    tp_size: int | None = None,
    tp_rank: int | None = None,
) -> torch.Tensor:
    """按 TP rank 取指定维上的分片 view。``tp_size==1`` 时返回原张量。"""
    if tp_size is None or tp_rank is None:
        size, rank = _tp_meta(None)
    else:
        size, rank = tp_size, tp_rank
    if size == 1:
        return weight
    chunk = weight.shape[dim] // size
    return weight.narrow(dim, chunk * rank, chunk)


class ReplicatedLinear(nn.Module):
    """整份复制、无通信。MoE router / EP 下的 shared expert 用这个。"""

    def __init__(self, in_size: int, out_size: int, *, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_size, in_size))
        setattr(self.weight, "weight_loader", self.weight_loader)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_size))
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class ColumnParallelLinear(nn.Module):
    """
        输出并行

        【数学原理】
        权重 W 按照输出维度（第0维）切分， W ∈ R^(out x in)，切分为 W_i ∈ R^((out/TP) x in。
        每个 Rank 独立计算局部的  Y_i = X * W_i^T。
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        bias: bool = False,
        gather_output: bool = True,
        tp_size: int | None = None,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.gather_output = gather_output
        self._tp_size, self._tp_rank = _tp_meta(tp_size)
        self.output_size_per_partition = divide(out_size, self._tp_size)
        # 初始化权重，仅持有当前 Rank 负责的权重部分
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, in_size)
        )
        # 初始化偏置，仅持有当前 Rank 负责的偏置部分
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)
        # 设置权重加载器
        setattr(self.weight, "weight_loader", self.weight_loader)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int = 0
    ) -> None:
        param.data.copy_(
            _shard(loaded_weight, dim=0, tp_size=self._tp_size, tp_rank=self._tp_rank)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.linear(x, self.weight, self.bias)

        # 跨 TP 通信，将所有 Rank 的局部输出拼接为完整输出
        if self.gather_output:
            return tensor_model_parallel_all_gather(out, dim=-1)
        return out


class RowParallelLinear(nn.Module):
    """
        输入并行

        【数学原理】
        权重 W 按照输出维度（第1维）切分， W ∈ R^(out x in)，切分为W_i ∈ R^(out x (in/TP))。
        
        【使用场景】
        通常紧跟在 ColumnParallelLinear 之后（例如 FFN 的 down_proj 或 Attention 的 out_proj）。
        输出前 all-reduce 求和，确保各 rank 的输出累加为全局输出。
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        bias: bool = False,
        input_is_parallel: bool = False,
        reduce_results: bool = True,
        tp_size: int | None = None,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        self._tp_size, self._tp_rank = _tp_meta(tp_size)
        self.input_size_per_partition = divide(in_size, self._tp_size)
        self.weight = nn.Parameter(
            torch.empty(out_size, self.input_size_per_partition)
        )
        # 行并行的Bias是全量持有
        if bias:
            self.bias = nn.Parameter(torch.empty(out_size))
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)
        setattr(self.weight, "weight_loader", self.weight_loader)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int = 0
    ) -> None:
        if param is self.bias:
            param.data.copy_(loaded_weight)
            return
        param.data.copy_(
            _shard(loaded_weight, dim=1, tp_size=self._tp_size, tp_rank=self._tp_rank)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel and self._tp_size > 1:
            shard = self.input_size_per_partition
            x = x.narrow(-1, self._tp_rank * shard, shard).contiguous()
        out = torch.nn.functional.linear(x, self.weight)
        if self.reduce_results and self._tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        # 加上偏置
        if self.bias is not None:
            out = out + self.bias
        return out


class QKVParallelLinear(nn.Module):
    """
    融合 QKV 投影层
    
    【设计目的】
    将 Query, Key, Value 的投影矩阵融合为一个大矩阵，减少 Kernel Launch 次数，提高显存带宽利用率。
    
    【特性】
    支持 MHA、MQA 和 GQA。
    当 KV Head 数量小于 TP Size 时，会自动在多个 Rank 之间复制 KV Head。
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        *,
        bias: bool = False,
    ):
        super().__init__()
        tp_size = get_tp_group().size
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_heads_total = total_num_heads
        self.num_kv_heads_total = total_num_kv_heads
        # Q 均分到每个 rank
        self.num_heads = divide(total_num_heads, tp_size)
        # 处理 KV Head 数量小于 TP Size 的情况
        if tp_size > total_num_kv_heads:
            self.num_kv_heads = 1
            # 记录每个 KV Head 需要被复制的次数 (例如 2个KV Head, 4个Rank -> 每个复制 2 次)
            self.num_kv_head_replicas = divide(tp_size, total_num_kv_heads)
        else:
            self.num_kv_heads = divide(total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        self.q_size = self.num_heads * head_size
        self.kv_size = self.num_kv_heads * head_size

        # 本地持有的融合权重形状: [q_size + 2*kv_size, hidden_size]
        self.weight = nn.Parameter(
            torch.empty(self.q_size + 2 * self.kv_size, hidden_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.q_size + 2 * self.kv_size))
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)
        setattr(self.weight, "weight_loader", self.weight_loader)

    def _shard_qkv(
        self, loaded_weight: torch.Tensor, shard_id: str
    ) -> tuple[int, torch.Tensor]:
        """
        根据 shard_id ('q', 'k', 'v') 从全量单体权重中切出本 Rank 负责的部分
        """

        tp_rank = get_tp_group().group_rank
        if shard_id == "q":
            begin = 0
            return begin, loaded_weight[
                tp_rank * self.q_size : (tp_rank + 1) * self.q_size
            ]
        if shard_id == "k":
            begin = self.q_size
        else:  # "v"
            begin = self.q_size + self.kv_size
        
        # 处理 GQA/MQA 下的 KV Head 复制逻辑
        if self.num_kv_head_replicas > 1:
            # 例：nkv=2, tp=4 时，head0 归 rank0,1 共享、head1 归 rank2,3 共享
            # 通过整除计算当前 Rank 应该获取哪一个全局 KV Head
            head_idx = tp_rank // self.num_kv_head_replicas
            start = head_idx * self.kv_size
        else:
            start = tp_rank * self.kv_size
        return begin, loaded_weight[start : start + self.kv_size]

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str
    ) -> None:
        assert shard_id in ("q", "k", "v", "bias")
        if shard_id == "bias":
            self._load_bias(loaded_weight)
            return
        begin, shard = self._shard_qkv(loaded_weight, shard_id)
        param.data[begin : begin + shard.shape[0]].copy_(shard)

    def _load_bias(self, loaded_weight: torch.Tensor) -> None:
        """
        加载合并形式的 QKV Bias。
        假设传入的 loaded_weight 格式为：[q_全量 | k_全量 | v_全量] 拼接而成的单段张量。
        """
        q_total = self.num_heads_total * self.head_size
        kv_total = self.num_kv_heads_total * self.head_size
        # 切分并加载 Q Bias
        _, q_shard = self._shard_qkv(loaded_weight[:q_total], "q")
        self.bias.data[: self.q_size].copy_(q_shard)
        k_seg = loaded_weight[q_total : q_total + kv_total]
        v_seg = loaded_weight[q_total + kv_total : q_total + 2 * kv_total]
        for idx, (seg, shard_id) in enumerate(((k_seg, "k"), (v_seg, "v"))):
            _, shard = self._shard_qkv(seg, shard_id)
            lo = self.q_size + idx * self.kv_size
            self.bias.data[lo : lo + self.kv_size].copy_(shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：直接进行融合矩阵乘法，不进行 Gather。
        返回的张量包含当前 Rank 负责的 [q_local | k_local | v_local]，由上层模型负责 Split。
        """
        return torch.nn.functional.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(nn.Module):
    """
    多列融合并行线性层。
    
    【设计目的】
    将多个独立的列并行层融合为一个巨大的权重矩阵，减少 GPU Kernel 的 Launch 次数，提升硬件计算利用率。
    """

    def __init__(
        self,
        in_size: int,
        output_sizes: list[int],
        *,
        bias: bool = False,
        tp_size: int | None = None,
    ):
        super().__init__()
        self.in_size = in_size
        self.output_sizes = output_sizes
        self._tp_size, self._tp_rank = _tp_meta(tp_size)
        self.output_sizes_per_partition = [
            divide(s, self._tp_size) for s in output_sizes
        ]
        total = sum(self.output_sizes_per_partition)
        self.weight = nn.Parameter(torch.empty(total, in_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(total))
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)
        setattr(self.weight, "weight_loader", self.weight_loader)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int
    ) -> None:
        shard_offset = sum(self.output_sizes_per_partition[:shard_id])
        shard_size = self.output_sizes_per_partition[shard_id]
        param.data[shard_offset : shard_offset + shard_size].copy_(
            _shard(loaded_weight, dim=0, tp_size=self._tp_size, tp_rank=self._tp_rank)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


def _default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight)


def get_weight_loader(param: nn.Parameter) -> Callable:
    return getattr(param, "weight_loader", _default_weight_loader)
