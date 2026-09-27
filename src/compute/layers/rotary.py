from __future__ import annotations

import torch
from torch import nn


def build_cos_sin_cache(
    head_dim: int,
    max_position_embeddings: int,
    rope_theta: float,
) -> torch.Tensor:
    """构建 [max_pos, head_dim] 查找表：前 head_dim/2 列为 cos、后为 sin。"""
    # 只取偶数下标 → head_dim/2 个逆频率（NeoX half-split 约定）
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
    )
    # 位置 × 频率 外积 → [max_pos, head_dim/2] 的相位角
    positions = torch.arange(max_position_embeddings, dtype=torch.float)
    angles = positions[:, None] * inv_freq[None, :]
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


class RotaryEmbedding(nn.Module):
    """持有 cos/sin 查找表，forward 转交 vllm::rotary_embedding 算子。"""

    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        # 仅支持全维旋转；部分旋转（rotary_dim < head_dim）需扩展此处
        assert rotary_dim == head_dim, "rotary_dim must equal head_dim (full rotation)"
        cache = build_cos_sin_cache(head_dim, max_position_embeddings, rope_theta)
        # persistent=False 表示该缓存实时计算，在保存权重时不会被保存
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _apply(self, fn, recurse: bool = True):
        """PyTorch在每次设备或者类型转换时自动调用的内部函数
        保证cos_sin_cache始终fp32，这是flashInfer的要求
        """

        # 父类照常执行转换操作
        out = super()._apply(fn, recurse) 
        # 如果缓存类型不是float32，则转换为float32
        if self.cos_sin_cache.dtype != torch.float32:
            self.cos_sin_cache = self.cos_sin_cache.to(torch.float32)
        return out

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 调用vllm的rotary_embedding算子
        return torch.ops.vllm.rotary_embedding(
            query, key, positions, self.cos_sin_cache, self.head_dim
        )
