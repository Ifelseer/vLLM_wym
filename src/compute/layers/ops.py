"""torch.compile 自定义算子注册。

该模块将 flashinfer 的底层 kernel 包装为 torch.library 自定义算子，
避免编译器尝试 trace 进这些 kernel 内部。

设计要点：
    - 每个算子通过 torch.library.custom_op 注册，编译时作为黑盒处理。
    - 每个算子注册 fake 实现，供 torch.compile 进行形状和 dtype 推导。
    - 底层 kernel 由 flashinfer 提供，本模块仅做形状适配与接口封装。
"""

from __future__ import annotations
import torch
from flashinfer import apply_rope_with_cos_sin_cache as _flashinfer_rope
from flashinfer.activation import silu_and_mul as _flashinfer_silu_and_mul
from flashinfer.norm import fused_add_rmsnorm as _flashinfer_fused_rmsnorm
from flashinfer.norm import rmsnorm as _flashinfer_rmsnorm


@torch.library.custom_op("vllm::silu_and_mul", mutates_args=())
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:

    return _flashinfer_silu_and_mul(x)


@silu_and_mul.register_fake
def _silu_and_mul_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty((x.shape[0], x.shape[1] // 2), dtype=x.dtype, device=x.device)


@torch.library.custom_op("vllm::rotary_embedding", mutates_args=())
def rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """执行 RoPE 旋转。"""
    n, n_qh = q.shape[:2]
    n_kh = k.shape[1]

    # flashinfer 要求输入为 [tokens, num_heads * head_size] 的连续张量。
    qf = q.reshape(n, n_qh * head_size).contiguous()
    kf = k.reshape(n, n_kh * head_size).contiguous()

    # 使用 neox 风格的 RoPE 实现。
    qo, ko = _flashinfer_rope(positions, qf, kf, head_size, cos_sin_cache, is_neox=True)

    # 恢复原始形状后返回。
    return qo.view_as(q), ko.view_as(k)


@rotary_embedding.register_fake
def _rotary_embedding_fake(q, k, positions, cos_sin_cache, head_size):
    return q, k


@torch.library.custom_op("vllm::rmsnorm", mutates_args=())
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig = x.shape
    x2 = x.reshape(-1, weight.shape[0])
    out = _flashinfer_rmsnorm(x2, weight, eps, enable_pdl=False)
    return out.view(orig)


@rmsnorm.register_fake
def _rmsnorm_fake(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("vllm::fused_add_rmsnorm", mutates_args=("x", "residual"))
def fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    # flashinfer fused 是 inplace：x 变为 norm 输出，residual 变为残差和。
    # 声明 mutates_args 让 torch.compile 正确做别名推断（否则 fake 认为输出是新的，
    # 与 CUDA Graph 捕获的实际写入冲突）。
    _flashinfer_fused_rmsnorm(x, residual, weight, eps, enable_pdl=False)


@fused_add_rmsnorm.register_fake
def _fused_add_rmsnorm_fake(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    pass