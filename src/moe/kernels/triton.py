"""Triton grouped GEMM for fused MoE (align / silu / sum via ``kernels.cuda``).

FP8 路径（``w13_scale`` / ``w2_scale`` 非 None 且权重为 fp8）：
- 激活按 (token, 128 k-group) 动态量化（``_quantize_token_group``）；
- 权重为 fp8 + per-block 128×128 scale，kernel 内按 k-block 用
  ``a_s ⊗ w_s`` 校正 fp32 累加（与 dense 侧 ``fp8_gemm`` 语义一致）；
- 中间激活（gate_up / silu 输出）保持 bf16，仅 GEMM 输入瞬时量化。

INT4 weight-only 路径（``int4=True``，AWQ/GPTQ）：
- 激活保持 bf16（weight-only，不量化激活）；
- 权重为 int32 packed ``[E, K, N/8]`` + per-group scales/zeros
  （``[E, K/G, N]``），kernel 内按 k-block 解包 nibble 并做
  ``(q - z) * s`` 反量化，与 dense 侧 ``quant/kernels/int4_gemm.py`` 同语义；
- 要求 ``BLOCK_K == G``（一个 k-block 恰一个 scale 组）。
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
    b_z_ptr,
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
    stride_bze,
    stride_bzk,
    stride_bzn,
    row_div: tl.constexpr,
    mul_w: tl.constexpr,
    naive: tl.constexpr,
    use_scale: tl.constexpr,
    int4: tl.constexpr,
    has_zeros: tl.constexpr,
    zero_bias: tl.constexpr,
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
    # per-block scale 索引：激活 [row, k_group]，权重 [expert, k_group, n]
    # n_blk 需 clamp：offs_n 可能越过 N（被 mask 的列），其输出本就被丢弃
    n_blk = tl.minimum(offs_n // _K_GROUP, (N // _K_GROUP) - 1)

    if int4:
        # 权重 int32 packed [E, K, N/8]：打包列共 BLOCK_N/8 个，
        # nibble j 对应输出通道 8*n8+j（与 offs_n 的展开顺序一致）
        offs_n8 = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
        shifts = (4 * tl.arange(0, 8))[None, None, :]
        b_ptrs = (
            b_ptr
            + expert * stride_be
            + offs_k[:, None] * stride_bk
            + offs_n8[None, :] * stride_bn
        )
    else:
        b_ptrs = (
            b_ptr
            + expert * stride_be
            + offs_k[:, None] * stride_bk
            + offs_n[None, :] * stride_bn
        )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _k in range(0, tl.cdiv(K, BLOCK_K)):
        k = _k * BLOCK_K
        a = tl.load(
            a_ptrs,
            mask=(token[:, None] < num_valid) & (offs_k[None, :] + k < K),
            other=0.0,
        )
        if int4:
            # [BK, BN/8] int32 → [BK, BN] 解包（三维 load + reshape，无重复访存）
            w32 = tl.load(
                b_ptrs,
                mask=(offs_n8[None, :] < (N // 8)),
                other=0,
            )
            q = (w32[:, :, None] >> shifts) & 0xF
            q = tl.reshape(q, (BLOCK_K, BLOCK_N)).to(tl.float32)
            # scale/zeros 按 k-group 取一行（BLOCK_K == G）
            s = tl.load(
                b_s_ptr
                + expert * stride_bse
                + (k // _K_GROUP) * stride_bsk
                + offs_n * stride_bsn,
                mask=(offs_n < N),
                other=0.0,
            )
            if has_zeros:
                z = tl.load(
                    b_z_ptr
                    + expert * stride_bze
                    + (k // _K_GROUP) * stride_bzk
                    + offs_n * stride_bzn,
                    mask=(offs_n < N),
                    other=0.0,
                ).to(tl.float32)
            else:
                z = tl.full((BLOCK_N,), zero_bias, dtype=tl.float32)
            w = (q - z[None, :]) * s[None, :]
            acc += tl.dot(a, w.to(a.dtype))
        else:
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
    b_zeros: torch.Tensor | None = None,
    int4: bool = False,
    zero_bias: float = 0.0,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    _, K = A.shape
    if int4:
        # INT4 packed：B 为 [E, K, N/8] int32，scales/zeros 为 [E, K/G, N]
        _, n8 = B.shape[1], B.shape[2]
        N = n8 * 8
        assert b_scale is not None and b_scale.shape == (B.shape[0], K // _GROUP, N)
    else:
        _, N, Kb = B.shape
        assert Kb == K, f"weight K={Kb} != activation K={K}"
    use_scale = a_scale is not None and not int4
    assert (not int4) or (b_scale is not None), "int4 requires scales"
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
    b_z = b_zeros if b_zeros is not None else B
    # B 的 stride 映射：INT4 packed 为 [E, K, N/8]；FP8/BF16 为 [E, N, K]
    s_be, s_bk, s_bn = (
        (B.stride(0), B.stride(1), B.stride(2))
        if int4
        else (B.stride(0), B.stride(2), B.stride(1))
    )
    grid = (triton.cdiv(grid_tokens, block_m) * triton.cdiv(N, block_n),)
    # scale stride 按 kernel 形参语义 (bse, bsn, bsk) 传：
    #   FP8  scales [E, n_blk, k_blk] → n 维是 stride(1)、k 维是 stride(2)
    #   INT4 scales [E, k_group, n]   → k 维是 stride(1)、n 维是 stride(2)
    # （kernel 内 INT4 分支用 ``(k//G)*stride_bsk + offs_n*stride_bsn``）
    if int4:
        s_bse, s_bsn, s_bsk = b_s.stride(0), b_s.stride(2), b_s.stride(1)
    else:
        s_bse, s_bsn, s_bsk = b_s.stride(0), b_s.stride(1), b_s.stride(2)
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
        b_z,
        num_valid,
        N,
        K,
        A.stride(0),
        A.stride(1),
        s_be,
        s_bk,
        s_bn,
        C.stride(0),
        C.stride(1),
        a_s.stride(0) if use_scale else 0,
        a_s.stride(1) if use_scale else 0,
        s_bse if (use_scale or int4) else 0,
        s_bsn if (use_scale or int4) else 0,
        s_bsk if (use_scale or int4) else 0,
        b_z.stride(0) if (int4 and b_zeros is not None) else 0,
        b_z.stride(1) if (int4 and b_zeros is not None) else 0,
        b_z.stride(2) if (int4 and b_zeros is not None) else 0,
        row_div=row_div,
        mul_w=topk_w is not None,
        naive=naive,
        use_scale=use_scale,
        int4=int4,
        has_zeros=(int4 and b_zeros is not None),
        zero_bias=float(zero_bias),
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
    w13_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    int4: bool = False,
    zero_bias: float = 0.0,
) -> torch.Tensor:
    w13 = w13_weight.contiguous()
    w2 = w2_weight.contiguous()
    M, top_k = hidden_states.shape[0], int(topk_ids.shape[1])
    E, num_valid = w13.shape[0], M * top_k
    dev = hidden_states.device.index or 0
    block_m, block_n, block_k, warps, stages = _blocks(M, E, top_k)
    if int4:
        # INT4：一个 k-block 恰一个 scale 组 → BLOCK_K 固定为 group_size
        block_k = _GROUP
        block_n = max(block_n, 64)  # 解包后有 8× 的 N 宽，N tile 不宜过小
    quantized = (w13_scale is not None) and not int4
    act_dtype = torch.bfloat16 if (quantized or int4) else hidden_states.dtype

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
    # INT4：激活保持 bf16（weight-only），权重为 packed int32 + per-group scale/zeros
    if int4:
        a_q, a_s = hidden_states, None
        w13_s = w13_scale.contiguous()
        w13_z = w13_zeros.contiguous() if w13_zeros is not None else None
        n13 = w13.shape[2] * 8  # packed [E, K, N/8] → N
    elif quantized:
        a_q, a_s = _quantize_token_group(hidden_states)
        w13_s = w13_scale.contiguous()
        w13_z = None
        n13 = w13.shape[1]
    else:
        a_q, a_s = hidden_states, None
        w13_s = None
        w13_z = None
        n13 = w13.shape[1]

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
        out_key=("gu", dev, num_valid, n13, act_dtype),
        naive=naive,
        a_scale=a_s,
        b_scale=w13_s,
        b_zeros=w13_z,
        int4=int4,
        zero_bias=zero_bias,
        out_dtype=act_dtype,
    )
    act2 = silu_and_mul(gate_up)
    if int4:
        a2_q, a2_s = act2, None
        w2_s = w2_scale.contiguous()
        w2_z = w2_zeros.contiguous() if w2_zeros is not None else None
        n2 = w2.shape[2] * 8  # packed [E, K, N/8] → N
    elif quantized:
        a2_q, a2_s = _quantize_token_group(act2)
        w2_s = w2_scale.contiguous()
        w2_z = None
        n2 = w2.shape[1]
    else:
        a2_q, a2_s = act2, None
        w2_s = None
        w2_z = None
        n2 = w2.shape[1]
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
        out_key=("dn", dev, num_valid, n2, act_dtype),
        naive=naive,
        topk_w=topk_weights.reshape(-1).contiguous(),
        a_scale=a2_s,
        b_scale=w2_s,
        b_zeros=w2_z,
        int4=int4,
        zero_bias=zero_bias,
        out_dtype=act_dtype,
    )
    return moe_sum(down, M, top_k)
