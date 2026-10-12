"""FP8 GEMM kernel：per-tensor（torch._scaled_mm）与 per-block 128×128（Triton）。

两条路径的 scale 语义（借鉴 vLLM ``w8a8_block_fp8`` 思路，独立实现）::

    per-tensor   W: [N, K] fp8 + 标量 w_scale；
                 激活动态量化：a_scale = amax(x, dim=-1) / 448（per-token）
                 → torch._scaled_mm(a_q, W.T, scale_a=a_scale, scale_b=w_scale)

    per-block    W: [N, K] fp8 + w_scale 网格 [ceil(N/128), ceil(K/128)]；
                 激活动态量化按 (token, 128 k-group)（kernel 内逐块归约），
                 T Triton ``_w8a8_block_fp8_kernel`` 按 k-block 用
                 a_s ⊗ w_scale 校正 fp32 累加值。

约定：
- 输出 dtype 与激活一致（bf16/fp16）；量化累加在 fp32。
- kernel 无逐帧分配（out 由调用方池化或一次分配），可进 CUDA Graph。
- ``tl.dot`` 的 FP8 输入需要 SM89+ tensor core；SM86 及以下不可用
  （QuantConfig.supports_device 已在配置层门控）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
SCALE_BLOCK = 128  # per-block scale 网格边长（weight 128x128 / act 1x128）


def quantize_fp8_per_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """整张量静态量化（测试 / cutlass MoE per-tensor 路径用）。"""
    amax = x.abs().amax().clamp(min=1e-12)
    scale = (amax / FP8_MAX).to(torch.float32)
    return (x / scale).to(FP8_DTYPE), scale


def quantize_fp8_per_block(
    x: torch.Tensor, block: int = SCALE_BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 [block, block] 二维网格量化 2D 张量（CPU/GPU 均可，加载期用）。

    Returns:
        (x_q fp8 [M, N], scale fp32 [ceil(M/block), ceil(N/block)])
    """
    assert x.dim() == 2, "per-block quantize expects a 2D tensor"
    m, n = x.shape
    mb, nb = (m + block - 1) // block, (n + block - 1) // block
    padded = torch.zeros(mb * block, nb * block, dtype=x.dtype, device=x.device)
    padded[:m, :n] = x
    blocks = padded.view(mb, block, nb, block)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale = (amax / FP8_MAX).to(torch.float32)
    q = (blocks / scale[:, None, :, None]).reshape(mb * block, nb * block)
    return q[:m, :n].to(FP8_DTYPE), scale


def dequantize_fp8_per_block(
    w_fp8: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """per-block fp8 → 高精度反量化（embedding/lm_head 等不量化模块用）。

    Args:
        w_fp8: [M, N] float8_e4m3fn。
        scale: [ceil(M/128), ceil(N/128)] fp32。
        out_dtype: 目标 dtype（通常 bf16/fp16）。
    """
    m, n = w_fp8.shape
    mb, nb = (m + 127) // 128, (n + 127) // 128
    s = scale.to(torch.float32)
    if s.shape != (mb, nb):
        s = s[:mb, :nb]
    expanded = s.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return (w_fp8.to(torch.float32) * expanded[:m, :n]).to(out_dtype)


# ------------------------------------------------------------------ #
# Triton per-block W8A8 kernel
# ------------------------------------------------------------------ #


def _block_configs():
    # (BM, BN, BK, GROUP_M, num_warps, num_stages)
    return [
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
            num_warps=8,
            num_stages=4,
        ),
    ]


@triton.autotune(configs=_block_configs(), key=["M", "N", "K"])
@triton.jit
def _w8a8_block_fp8_kernel(
    X,          # [M, K] bf16/fp16 激活（行主序）
    W,          # [N, K] fp8 权重（行主序）
    WS,         # [ceil(N/128), ceil(K/128)] fp32 权重 block scale
    Out,        # [M, N] 输出（激活 dtype）
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_wsn, stride_wsk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    n_start = pid_n * BLOCK_N
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)

    # BLOCK_N/BLOCK_K 均为 128 的因数 → 权重 scale 网格索引按 program 对齐
    ws_n = (n_start // SCALE_BLOCK) * stride_wsn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        if EVEN_K:
            x = tl.load(x_ptrs)
            w = tl.load(w_ptrs)
        else:
            k_mask = offs_k[None, :] + k < K
            x = tl.load(x_ptrs, mask=k_mask, other=0.0)
            w = tl.load(w_ptrs, mask=k_mask, other=0.0)
        w_scale = tl.load(WS + ws_n + (k // SCALE_BLOCK) * stride_wsk)

        # 激活动态量化：per-token × per-128-k-group（BLOCK_K == 128 对齐组宽）
        x_amax = tl.max(tl.abs(x), axis=1)
        x_q = (x / x_amax[:, None] * FP8_MAX).to(tl.float8e4nv)
        a_s = x_amax / FP8_MAX

        # W 已是 fp8；tl.dot 走 FP8 tensor core（SM89+），fp32 累加
        acc += tl.dot(x_q, tl.trans(w)) * (a_s[:, None] * w_scale)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    out = acc.to(Out.dtype.element_ty)
    o_ptrs = Out + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptrs, out, mask=mask)


def w8a8_block_fp8_matmul(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """per-block 128×128 FP8 GEMM：``x @ W^T``，激活动态 per-token 量化。

    Args:
        x: [M, K]，bf16/fp16，行主序连续。
        w_fp8: [N, K] float8_e4m3fn，行主序。
        w_scale: [ceil(N/128), ceil(K/128)] fp32。
        out: 可选输出缓冲（池化；CUDA Graph replay 时复用）。
    """
    M, K = x.shape
    N = w_fp8.shape[0]
    if out is None:
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if M == 0:
        return out
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    _w8a8_block_fp8_kernel[grid](
        x,
        w_fp8,
        w_scale,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w_fp8.stride(0),
        w_fp8.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        out.stride(0),
        out.stride(1),
        EVEN_K=(K % SCALE_BLOCK == 0),
    )
    return out


# ------------------------------------------------------------------ #
# per-tensor W8A8（torch._scaled_mm，SM89+ 原生）
# ------------------------------------------------------------------ #


def w8a8_tensor_fp8_matmul(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """per-tensor FP8 GEMM：激活 per-token 动态 × 权重标量 scale。

    ``w_scale`` 为形状 [1] / 标量的 fp32 张量。
    """
    a_scale = (x.abs().amax(dim=-1).clamp(min=1e-12) / FP8_MAX).to(torch.float32)
    a_q = (x / a_scale[:, None]).to(FP8_DTYPE)
    ws = w_scale.reshape(())
    if out is None:
        out = torch.empty((x.shape[0], w_fp8.shape[0]), dtype=x.dtype, device=x.device)
    torch._scaled_mm(
        a_q,
        w_fp8.t(),  # [K, N] column-major view（_scaled_mm 要求）
        scale_a=a_scale.view(-1, 1),
        scale_b=ws,
        out_dtype=out.dtype,
        out=out,
    )
    return out


def fp8_matmul(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    per_block: bool,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """按 scale 模式分派 FP8 GEMM。"""
    if per_block:
        return w8a8_block_fp8_matmul(x, w_fp8, w_scale, out=out)
    return w8a8_tensor_fp8_matmul(x, w_fp8, w_scale, out=out)
