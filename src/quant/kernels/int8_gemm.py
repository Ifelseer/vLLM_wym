"""INT8 W8A8 kernel：per-token 动态激活 × per-channel 静态权重。

设计（对齐 vLLM/SGLang 的 w8a8 int8 路径）：

- **激活**：per-token 动态对称量化（每行 ``amax/127``，运行时计算，无需校准）。
  per-token 而非 per-tensor：注意力/FFN 激活存在显著的逐 token 离群值，
  per-tensor 会压垮整行精度（与 FP8 路径同一动机）。
- **权重**：per-channel（输出通道）对称静态 int8，scale 为 ``[out]`` fp32。
  per-channel 而非 per-tensor：权重的离群值集中在少数输出通道上。
- **GEMM**：``torch._int_mm``（cuBLASLt int8 tensor core，SM61+ 可用，
  本地 3060 可端到端验证）→ int32 累加 → 乘 ``(a_scale ⊗ w_scale)`` 回 bf16。

量化是纯对称的（无零点），int8 范围 [-127, 127]（避开 -128 使量化
``round(x/s)`` 的表示域对称，反量化时无偏置项）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

INT8_DTYPE = torch.int8
_INT8_MAX = 127.0

# _int_mm 的输入约束（cuBLASLt int8）：K 与 N 需为 4 的倍数，且 M > 16
_ALIGN = 4
_INT_MM_MIN_M = 17


@triton.jit
def _per_token_quant_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    K,
    stride_xm,
    stride_xk,
    stride_qm,
    stride_qk,
    stride_sm,
    BLOCK_K: tl.constexpr,
):
    """融合 per-token 量化：一次 program 处理一行（token）。

    朴素实现（abs → amax → clamp → div → round → cast）是 6~9 个小 kernel、
    对 ``[M,K]`` 多次全量读写；实测 M=128/K=2048 时 165us，而 GEMM 本身
    只要 65us——量化是瓶颈。本内核两遍过 K（amax + quantize，同段命中
    L1/L2），把开销压到 ~20us。
    """
    pid = tl.program_id(0)
    amax = 0.0
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + pid * stride_xm + offs * stride_xk, mask=offs < K, other=0.0
        )
        amax = tl.maximum(amax, tl.max(tl.abs(x), axis=0))
    scale = tl.maximum(amax, 1e-12) / 127.0
    tl.store(s_ptr + pid * stride_sm, scale)
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(
            x_ptr + pid * stride_xm + offs * stride_xk, mask=mask, other=0.0
        )
        # Triton 的 tl 张量无 .round()；libdevice.round（round-half-away）
        # 与 torch.round（round-half-even）在 .5 边界可能差 1，对量化无实际影响
        q = libdevice.round(x / scale)
        tl.store(q_ptr + pid * stride_qm + offs * stride_qk, q.to(tl.int8), mask=mask)


def quantize_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``x[M,K]`` → (int8 [M,K], fp32 scale [M,1])（对称 per-token 动态）。

    CUDA 输入走融合 Triton 内核；CPU 输入（单测）走等价的 torch 实现。
    """
    if x.is_cuda:
        m, k = x.shape
        q = torch.empty(m, k, dtype=INT8_DTYPE, device=x.device)
        s = torch.empty(m, 1, dtype=torch.float32, device=x.device)
        block_k = min(triton.next_power_of_2(k), 2048)
        _per_token_quant_kernel[(m,)](
            x,
            q,
            s,
            k,
            x.stride(0),
            x.stride(1),
            q.stride(0),
            q.stride(1),
            s.stride(0),
            BLOCK_K=block_k,
            num_warps=4,
        )
        return q, s
    amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    # scale 在 fp32 下计算：bf16 amax 直接除会得到 bf16 scale（0.4% 系统误差）
    scale = amax.to(torch.float32) / _INT8_MAX
    q = (x / scale).round().to(INT8_DTYPE)
    return q, scale


def quantize_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``w[out,in]`` → (int8 [out,in], fp32 scale [out])（对称 per-channel 静态）。"""
    amax = w.abs().amax(dim=1).clamp(min=1e-12)
    scale = amax.to(torch.float32) / _INT8_MAX
    q = (w / scale[:, None]).to(INT8_DTYPE)
    return q, scale


def dequantize_per_channel(
    q: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """参考反量化（测试/调试用）：int8 [out,in] × fp32 [out] → fp32 [out,in]。"""
    return q.to(torch.float32) * scale[:, None].to(torch.float32)


@triton.jit
def _int8_gemm_small_m_kernel(
    a_q_ptr,
    a_s_ptr,
    q_ptr,
    s_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_asm,
    stride_qn,
    stride_qk,
    stride_sn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """小 M（decode）的 int8 GEMM：激活已量化，权重 int8 直接 tl.dot → int32 累加。

    ``torch._int_mm`` 硬性要求 M > 16（cuBLASLt int8 路径），decode 的
    M=1..16 走本内核；权重按 int8（1 字节/元素）访存，带宽最优。
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(
            a_q_ptr + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak,
            mask=m_mask[:, None],
            other=0,
        )
        w = tl.load(
            q_ptr + offs_n[None, :] * stride_qn + (k0 + offs_k)[:, None] * stride_qk,
            mask=n_mask[None, :],
            other=0,
        )
        acc += tl.dot(a, w, out_dtype=tl.int32)

    a_s = tl.load(a_s_ptr + offs_m * stride_asm, mask=m_mask, other=0.0)
    s = tl.load(s_ptr + offs_n * stride_sn, mask=n_mask, other=0.0)
    result = acc.to(tl.float32) * a_s[:, None] * s[None, :]
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        result.to(c_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def _int8_triton_matmul(
    a_q: torch.Tensor,
    a_s: torch.Tensor,
    w_int8: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """小 M（<=16）路径。

    ``BLOCK_M`` 按 M 精确取（1/2/4/8/16）而非固定 16：decode 的 M=1 是
    带宽瓶颈的 GEMV，用 BLOCK_M=1 让 Triton 走 FMA 路径（无需 tensor core），
    实测比 BLOCK_M=16+掩码 快 ~7x（267us → 36us，并比 bf16 cuBLAS 快 2x）。
    """
    m, k = a_q.shape
    n = w_int8.shape[0]
    block_m = max(1, triton.next_power_of_2(m))
    block_n, block_k = 128, 128
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _int8_gemm_small_m_kernel[grid](
        a_q,
        a_s,
        w_int8,
        w_scale,
        out,
        m,
        n,
        k,
        a_q.stride(0),
        a_q.stride(1),
        a_s.stride(0),
        w_int8.stride(0),
        w_int8.stride(1),
        w_scale.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )


def w8a8_int8_matmul(
    x: torch.Tensor,
    w_int8: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """W8A8 GEMM：``x[M,K] @ dequant(w_int8)[K,N]``（激活动态 per-token 量化）。

    两条路径（按 M 分派，cuBLASLt int8 要求 M > 16）：
    - ``M > 16``：``torch._int_mm``（int8 tensor core，int32 累加）。
    - ``M <= 16``：Triton 小 M 内核（decode 的 M=1..16；权重按 int8 访存）。

    Args:
        x: [M, K] bf16/fp16 激活。
        w_int8: [N, K] int8（N=out，与 ``dequantize_per_channel`` 的布局一致）。
        w_scale: [N] fp32 per-channel scale。
        out: 可选输出缓冲（CUDA Graph replay 语义：同形复用）。
    """
    m, k = x.shape
    n = w_int8.shape[0]
    if m == 0:
        if out is None:
            out = torch.empty(0, n, dtype=x.dtype, device=x.device)
        return out
    if k % _ALIGN != 0 or n % _ALIGN != 0:
        raise ValueError(
            f"int8 w8a8 requires K and N divisible by {_ALIGN}, got K={k} N={n}"
        )
    if w_int8.shape[1] != k:
        raise ValueError(f"weight K={w_int8.shape[1]} != activation K={k}")
    if w_int8.dtype != INT8_DTYPE or not w_int8.is_cuda:
        raise ValueError("w8a8_int8_matmul requires CUDA int8 weights")

    a_q, a_s = quantize_per_token(x)  # [M,K] int8, [M,1] fp32
    if out is None:
        out = torch.empty(m, n, dtype=x.dtype, device=x.device)
    if m < _INT_MM_MIN_M:
        _int8_triton_matmul(a_q, a_s, w_int8, w_scale, out)
        return out
    # mat2 需要列主序：w.t()（w 行主序 → 其转置是 [K,N] 列主序视图）
    acc = torch._int_mm(a_q, w_int8.t())  # int32 [M,N]
    out.copy_((acc.to(torch.float32) * a_s * w_scale.to(torch.float32)).to(out.dtype))
    return out
