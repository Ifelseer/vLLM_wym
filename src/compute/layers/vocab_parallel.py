
from __future__ import annotations

import torch
from torch import nn

from ...distributed import (
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)

# ==============================================================================
# Qwen3 张量并行核心组件实现。
# 包含 Embedding 层和 LM Head 层，针对显存带宽和通信开销做了优化。
# ==============================================================================

def _pad_vocab_size(vocab_size: int, tp_size: int, *, multiple: int = 16) -> int:
    """
        计算填充后的词表总大小。
        保证能被 tp_size 整除，并且每个 Rank 分到的词表大小是 multiple (如 16) 的倍数，
        确保契合 NVIDIA GPU Tensor Cores 的矩阵乘法对齐要求，最大化算力利用率。
    """
    # 1. 计算每个 Rank 基础需要处理的词表大小 (向上取整)
    per = (vocab_size + tp_size - 1) // tp_size
    # 2. 向上取整到 multiple 的倍数
    per = ((per + multiple - 1) // multiple) * multiple
    # 3. 返回填充后的总词表大小
    return per * tp_size


def _load_weight(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    *,
    padded_vocab: int,
    row_offset: int = 0,
) -> None:
    """
        从全量权重中加载当前 Rank 所需的分片。
        采用 In-place 切片与清零，避免带来额外拷贝开销。
    """
    # 获取了当前进程所属的 TP 组对象，核心有 TP 有几张 GPU，还有当前 GPU 的 rank 号
    tp = get_tp_group()
    # 计算每张 GPU 具体需要负责存储和处理多少个词
    per_partition = padded_vocab // tp.size
    # 计算当前 Rank 负责的词表起始索引
    start = per_partition * tp.group_rank
    
    # 计算当前 Rank 实际能切出来的有效行数 (处理词表末尾越界情况)
    seg_len = max(0, min(per_partition, loaded_weight.shape[0] - start))
    
    if seg_len > 0:
        # 原地拷贝有效权重到目标参数中 (支持 row_offset 偏移，用于 Embedding 哑元行)
        param.data[row_offset : row_offset + seg_len].copy_(
            loaded_weight[start : start + seg_len]
        )
        
    if seg_len < per_partition:
        # 如果切片越界，将剩余未填充的行原地清零
        param.data[row_offset + seg_len : row_offset + per_partition].zero_()


class VocabParallelEmbedding(nn.Module):
    """
        词表并行 Embedding 层。
        核心优化：引入 "+1 哑元行"，将越界 Token 映射到索引 0。
        查表操作直接输出全 0 向量，后续只需进行 AllReduce，省去对高维浮点 Tensor 
        进行 Mask 操作，将访存压力从 [B, S, H] 降维到了 [B, S]。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        tp = get_tp_group()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # 计算对齐后的总词表大小
        padded = _pad_vocab_size(num_embeddings, tp.size)
        
        # 计算当前 Rank 负责的词表 ID 范围 [start, end)
        self.vocab_start_index, self.vocab_end_index = self._vocab_range(
            padded // tp.size, tp
        )
        
        # 分配权重：大小比 per_partition 多 1 行。第 0 行是永远为 0 的哑元行 (Dummy Row)
        self.weight = nn.Parameter(
            torch.zeros(padded // tp.size + 1, embedding_dim)
        )
        # 绑定权重加载回调
        setattr(self.weight, "weight_loader", self.weight_loader)

    def _vocab_range(self, per_partition: int, tp) -> tuple[int, int]:
        """计算当前 Rank 负责的词表 ID 边界。"""
        return (
            per_partition * tp.group_rank,
            per_partition * (tp.group_rank + 1),
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int = 0
    ) -> None:
        """权重加载回调函数。"""
        _load_weight(
            param,
            loaded_weight,
            padded_vocab=_pad_vocab_size(self.num_embeddings, get_tp_group().size),
            row_offset=1,  # 【关键】跳过第 0 行的哑元占位行，从第 1 行开始写入真实权重
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        # 1. 找出不在当前 Rank 词表范围内的 Token (越界 Token)
        mask = (x < self.vocab_start_index) | (x >= self.vocab_end_index)
        
        # 2. 将 Token ID 转换为当前分片的局部索引，并加上 1 (避开第 0 行哑元)
        local_idx = x - self.vocab_start_index + 1
        
        # 3. 将越界 Token 的局部索引强制设为 0 (In-place 操作，节省显存)
        # 这样它们查表时就会读到第 0 行的全 0 向量
        local_idx.masked_fill_(mask, 0)
        
        # 4. 查表并聚合：真实 Token 得到正确向量，越界 Token 得到 0 向量
        # 经过 AllReduce 后：真向量 + 0 + 0 + 0 = 最终完整的 Embedding 向量
        out = torch.nn.functional.embedding(local_idx, self.weight)
        return tensor_model_parallel_all_reduce(out)


class ParallelLMHead(nn.Module):
    """
        并行 LM Head 层 (用于输出 Logits)。
        包含常规的 AllGather 前向传播，以及针对推理期 Greedy Decoding
        优化 argmax，大幅降低节点间的通信带宽压力。
    """

    def __init__(self, hidden_size: int, vocab_size: int, *, bias: bool = False):
        super().__init__()
        tp = get_tp_group()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # 计算对齐后的词表大小及当前 Rank 的分片大小
        self.padded_vocab_size = _pad_vocab_size(vocab_size, tp.size)
        self.output_size_per_partition = self.padded_vocab_size // tp.size
        
        # 当前 Rank 负责的词表范围
        self.vocab_start_index = self.output_size_per_partition * tp.group_rank
        self.vocab_end_index = self.output_size_per_partition * (tp.group_rank + 1)
        
        # 分配局部权重和可选的 Bias
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, hidden_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
        else:
            self.register_parameter("bias", None)
            
        setattr(self.weight, "weight_loader", self.weight_loader)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int = 0
    ) -> None:
        """权重加载回调函数。"""
        _load_weight(
            param,
            loaded_weight,
            padded_vocab=self.padded_vocab_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
            标准前向传播。
            局部线性变换 -> AllGather 拼接全局 Logits -> 截断掉 Padding 部分。
        """
        logits = torch.nn.functional.linear(hidden_states, self.weight, self.bias)
        logits = tensor_model_parallel_all_gather(logits, dim=-1)
        return logits[..., : self.vocab_size]

    def argmax(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
            【推理期优化】用于 Greedy Decoding 的分布式 Argmax。
            不传输完整的 Logits，而是传输每个 Rank 的局部最大值和索引，
            通信量降低 99% 以上，极大缓解多卡推理时的网络带宽瓶颈。
        """
        tp = get_tp_group()
        per = self.output_size_per_partition
        
        # 1. 本地计算局部 Logits
        local_logits = torch.nn.functional.linear(hidden_states, self.weight, self.bias)
        
        # 2. 处理词表 Padding 导致的无效 Logits：将其置为 -inf，防止干扰 Argmax
        pad_start = max(0, self.vocab_size - self.vocab_start_index)
        if pad_start < per:
            local_logits[..., pad_start:] = float("-inf")
            
        # 3. 在词表维度求局部的最大值及其对应的索引
        vals, idx = local_logits.max(dim=-1)
        
        # 4. 将局部索引还原为全局索引
        idx = idx + per * tp.group_rank
        
        # 5. 跨卡收集所有 Rank 的局部最大值 (vals) 和局部索引 (idx)
        vals = tensor_model_parallel_all_gather(vals, dim=-1)
        idx = tensor_model_parallel_all_gather(idx, dim=-1)
        
        # 6. 重组 Tensor 形状为 [Batch_Size, TP_Size]，以便在 TP 维度上进行对比
        bs = hidden_states.shape[0]
        vals = vals.reshape(tp.size, bs).t()
        idx = idx.reshape(tp.size, bs).t()
        
        # 7. 找出哪个 Rank 提供的 Value 最大，获取该 Rank 的序号 (best_rank)
        best = vals.argmax(dim=-1)
        
        # 8. 根据 best_rank，从对应的 idx 中提取出最终的全局最优 Token ID
        return idx.gather(-1, best.unsqueeze(-1)).squeeze(-1)