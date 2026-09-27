"""Reusable nn modules (Attention, norms, RoPE, activations)."""

from __future__ import annotations

from . import ops  # noqa: F401  注册 vllm::silu_and_mul / rotary_embedding custom op
from .activation import SiluAndMul
from .attention import Attention
from .layernorm import RMSNorm
from .linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from .rotary import RotaryEmbedding
from .vocab_parallel import ParallelLMHead, VocabParallelEmbedding

__all__ = [
    "Attention",
    "RMSNorm",
    "RotaryEmbedding",
    "SiluAndMul",
    "ColumnParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "ReplicatedLinear",
    "RowParallelLinear",
    "ParallelLMHead",
    "VocabParallelEmbedding",
]
