"""Triton grouped GEMM for fused MoE (align / silu / sum via ``kernels.cuda``).

FP8 路径（``w13_scale`` / ``w2_scale`` 非 None）：
- 激活按 (token, 128 k-group) 动态量化（``_quantize_token_group``）；
- 权重为 fp8 + per-block 128×128 scale，kernel 内按 k-block 用
  ``a_s ⊗ w_s`` 校正 fp32 累加（与 dense 侧 ``fp8_gemm`` 语义一致）；
- 中间激活（gate_up / silu 输出）保持 bf16，仅 GEMM 输入瞬时量化。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .cuda import moe_align_block_size, moe_sum, silu_and_mul

_WS: dict[tuple, torch.Tensor] = {}

_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0
_GROUP = 128

# Triton 3.x 只允许内核引用 constexpr 全局；host 侧仍用上面的普通常量
_K_GROUP = tl.constexpr(_GROUP)


def clear_state() -> None:
    _WS.clear()
    from .cuda import clear_state as clear_cuda_state

    clear_cuda_state()


def _scratch(key: tuple, *, shape, dtype, device) -> torch.Tensor:
    buf = _WS.get(key)
    if buf is None or buf.shape != shape or buf.dtype != dtype or buf.device != device:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _WS[key] = buf
    return buf


def _quantize_token_group(
    x: torch.Tensor, group: int = _GROUP
) -> tuple[torch.Tensor, torch.Tensor]:
    """[M, K] bf16/fp16 → (fp8 [M, K], scale fp32 [M, K/group])。

    K 必须整除 group（hidden / moe_intermediate 均为 128 倍数）。
    """
    M, K = x.shape
    if K % group != 0:
        raise ValueError(f"fp8 MoE activation K={K} not divisible by {group}")
    xg = x.view(M, K // group, group)
    amax = xg.abs().amax(dim=-1).clamp(min=1e-12)
    s = (amax / _FP8_MAX).to(torch.float32)
    q = (xg / s[..., None]).reshape(M, K).to(_FP8)
    return q, s


def _blocks(M: int, E: int, top_k: int) -> tuple[int, int, int, int, int]:
    tpe = max((M * top_k) // max(E, 1), 1)
    block_m = 16 if tpe <= 16 else 32 if tpe <= 40 else 64 if tpe <= 80 else 128
    block_n = 64 if M <= 64 else 128
    block_k = 128 if M <= 64 else 64
    warps = 4 if M <= 128 else 8
    stages = 4 if M <= 32 else 3
    return block_m, block_n, block_k, warps, stages


@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sorted_ptr,
    expert_ptr,
    n_post_ptr,
    topk_w_ptr,
    a_s_ptr,
    b_s_ptr,
    num_valid,
    N,
    K,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsn,
    stride_bsk,
    row_div: tl.constexpr,
    mul_w: tl.constexpr,
    naive: tl.constexpr,
    use_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    n_post = tl.load(n_post_ptr)
    if pid_m * BLOCK_M >= n_post:
        return

    offs = tl.arange(0, BLOCK_M)
    if naive:
        token = tl.where(offs == 0, pid_m, num_valid)
    else:
        token = tl.load(sorted_ptr + pid_m * BLOCK_M + offs)

    expert = tl.load(expert_ptr + pid_m)
    if expert < 0:
        return

    a_row = tl.where(token < num_valid, token // row_div, 0)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        b_ptr
        + expert * stride_be
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )
    # per-block scale 索引：激活 [row, k_group]，权重 [expert, n_block, k_group]
    # n_blk 需 clamp：offs_n 可能越过 N（被 mask 的列），其输出本就被丢弃
    n_blk = tl.minimum(offs_n // _K_GROUP, (N // _K_GROUP) - 1)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _k in range(0, tl.cdiv(K, BLOCK_K)):
        k = _k * BLOCK_K
        a = tl.load(
            a_ptrs,
            mask=(token[:, None] < num_valid) & (offs_k[None, :] + k < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        if use_scale:
            k_blk = k // _K_GROUP
            a_s = tl.load(a_s_ptr + a_row * stride_asm + k_blk * stride_ask)
            w_s = tl.load(
                b_s_ptr
                + expert * stride_bse
                + n_blk * stride_bsn
                + k_blk * stride_bsk
            )
            acc += tl.dot(a, b) * (a_s[:, None] * w_s[None, :])
        else:
            acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if mul_w:
        w = tl.load(topk_w_ptr + token, mask=token < num_valid, other=0.0)
        acc = acc * w[:, None]

    tl.store(
        c_ptr + token[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=(token[:, None] < num_valid) & (offs_n[None, :] < N),
    )


def _gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    sorted_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_post: torch.Tensor,
    num_valid: int,
    row_div: int,
    block_m: int,
    block_n: int,
    block_k: int,
    warps: int,
    stages: int,
    grid_tokens: int,
    out_key: tuple,
    naive: bool,
    topk_w: torch.Tensor | None = None,
    a_scale: torch.Tensor | None = None,
    b_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    _, K = A.shape
    _, N, Kb = B.shape
    assert Kb == K
    use_scale = a_scale is not None
    assert use_scale == (b_scale is not None), "a_scale/b_scale must come in pairs"
    c_dtype = out_dtype or A.dtype
    C = _scratch(
        (out_key, c_dtype), shape=(max(num_valid, 1), N), dtype=c_dtype, device=A.device
    )
    C.zero_()
    if num_valid == 0 or grid_tokens == 0:
        return C
    sorted_ids = expert_ids if sorted_ids is None else sorted_ids
    tw = topk_w if topk_w is not None else A
    a_s = a_scale if a_scale is not None else A
    b_s = b_scale if b_scale is not None else B
    grid = (triton.cdiv(grid_tokens, block_m) * triton.cdiv(N, block_n),)
    _gemm_kernel[grid](
        A,
        B,
        C,
        sorted_ids,
        expert_ids,
        num_post,
        tw,
        a_s,
        b_s,
        num_valid,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        a_s.stride(0) if use_scale else 0,
        a_s.stride(1) if use_scale else 0,
        b_s.stride(0) if use_scale else 0,
        b_s.stride(1) if use_scale else 0,
        b_s.stride(2) if use_scale else 0,
        row_div=row_div,
        mul_w=topk_w is not None,
        naive=naive,
        use_scale=use_scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=warps,
        num_stages=stages,
    )
    return C


def forward(
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    w13 = w13_weight.contiguous()
    w2 = w2_weight.contiguous()
    M, top_k = hidden_states.shape[0], int(topk_ids.shape[1])
    E, num_valid = w13.shape[0], M * top_k
    dev = hidden_states.device.index or 0
    block_m, block_n, block_k, warps, stages = _blocks(M, E, top_k)
    quantized = w13_scale is not None
    act_dtype = torch.bfloat16 if quantized else hidden_states.dtype

    # Decode: skip align when extremely sparse (same heuristic as vLLM).
    if M * top_k * 4 <= E:
        expert_ids = topk_ids.reshape(-1).to(torch.int32)
        grid_tokens = expert_ids.numel() * block_m
        num_post = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
        num_post.fill_(grid_tokens)
        sorted_ids, naive = None, True
    else:
        sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, block_m, E)
        grid_tokens, naive = sorted_ids.numel(), False

    # FP8：激活按 (token, 128 group) 动态量化；权重已是 fp8 + block scale
    if quantized:
        a_q, a_s = _quantize_token_group(hidden_states)
        w13_s = w13_scale.contiguous()
    else:
        a_q, a_s = hidden_states, None
        w13_s = None

    gate_up = _gemm(
        a_q,
        w13,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_post=num_post,
        num_valid=num_valid,
        row_div=top_k,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        warps=warps,
        stages=stages,
        grid_tokens=grid_tokens,
        out_key=("gu", dev, num_valid, w13.shape[1], act_dtype),
        naive=naive,
        a_scale=a_s,
        b_scale=w13_s,
        out_dtype=act_dtype,
    )
    act2 = silu_and_mul(gate_up)
    if quantized:
        a2_q, a2_s = _quantize_token_group(act2)
        w2_s = w2_scale.contiguous()
    else:
        a2_q, a2_s = act2, None
        w2_s = None
    down = _gemm(
        a2_q,
        w2,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_post=num_post,
        num_valid=num_valid,
        row_div=1,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        warps=warps,
        stages=stages,
        grid_tokens=grid_tokens,
        out_key=("dn", dev, num_valid, w2.shape[1], act_dtype),
        naive=naive,
        topk_w=topk_weights.reshape(-1).contiguous(),
        a_scale=a2_s,
        b_scale=w2_s,
        out_dtype=act_dtype,
    )
    return moe_sum(down, M, top_k)
