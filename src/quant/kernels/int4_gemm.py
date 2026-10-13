"""INT4 weight-only（AWQ / GPTQ）的打包工具与 Triton fused dequant-GEMM。

**内部统一布局（in-major）**——两种 checkpoint 格式在加载期归一化到它：

    qweight : int32 [in, out/8]      nibble j of qweight[k, n8] = 输出通道 8*n8+j
    zeros   : uint8 [in/G, out]      （加载期解包，热路径不再做位运算）
    scales  : fp16  [in/G, out]

**两种 checkpoint 格式（检索日 2026-10-13，源于 vLLM 内核与 pack helper）**

GPTQ（AutoGPTQ，uint4b8 约定）::

    qweight [in/8, out]    nibble j = 输入通道 8*k8+j；存值 = 量化值 + 8（偏置）
    qzeros  [in/G, out/8]  nibble j = 输出通道 8*n8+j
    scales  [in/G, out]
    dequant: w = (q - 8 - z) * s        # sym 时 z=0；asym 时 z 来自 qzeros

AWQ（AutoAWQ，非对称）::

    qweight [in, out/8]（vLLM 侧）或 [out, in/8]（AutoAWQ 原始，形状可判）
    qzeros  [in/G, out/8]  （out-major 时为 [out, in/G/8]）
    scales  [in/G, out]    （out-major 时为 [out, in/G]）
    dequant: w = (q - z) * s
    注意：in-major 布局下输出通道在 int32 内按 [0,2,4,6,1,3,5,7] 交织，
    加载期需还原（见 ``normalize_int4_to_in_major``）。

**为什么加载期归一化而不是内核里适配**：交织/转置是"每个权重一次性"的
成本（CPU numpy 级），而热路径每步都跑；用一次性成本换内核与测试面的
统一（单一布局 + 单一内核，AWQ/GPTQ 只在偏置与零点语义上分叉）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# AWQ 在 in-major 布局下输出通道的交织顺序（vLLM awq_triton 的逆序表）
AWQ_INTERLEAVE = (0, 2, 4, 6, 1, 3, 5, 7)
# uint4b8 的对称中心（GPTQ sym=true 时代替 zeros）
INT4_SYM_BIAS = 8.0

_WS: dict[tuple, torch.Tensor] = {}


def clear_state() -> None:
    _WS.clear()


def _scratch(key: tuple, *, shape, dtype, device) -> torch.Tensor:
    buf = _WS.get(key)
    if buf is None or buf.shape != shape or buf.dtype != dtype or buf.device != device:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _WS[key] = buf
    return buf


# ====================================================================== #
# 加载期：打包 / 解包 / 归一化（CPU 友好）
# ====================================================================== #


def pack_int4_rows(w_uint8: torch.Tensor, *, interleave: bool = False) -> torch.Tensor:
    """[K, N] uint4 值 → int32 [K, N/8]（nibble j = N 通道 8*n8+j）。

    Args:
        interleave: 是否施加 AWQ 的 N 交织（模拟 AWQ checkpoint 用）。
    """
    k, n = w_uint8.shape
    assert n % 8 == 0, f"N={n} must be divisible by 8"
    w = w_uint8.to(torch.int64)
    if interleave:
        # 每个 8 通道组按 AWQ_INTERLEAVE 重排后再打包
        w = w.view(k, n // 8, 8)[:, :, list(AWQ_INTERLEAVE)].reshape(k, n)
    w = w.view(k, n // 8, 8)
    out = torch.zeros(k, n // 8, dtype=torch.int32, device=w_uint8.device)
    for j in range(8):
        out |= (w[:, :, j] & 0xF).to(torch.int32) << (4 * j)
    return out


def pack_int4_cols(w_uint8: torch.Tensor) -> torch.Tensor:
    """[K, N] uint4 值 → int32 [K/8, N]（nibble j = K 通道 8*k8+j；GPTQ 用）。"""
    k, n = w_uint8.shape
    assert k % 8 == 0, f"K={k} must be divisible by 8"
    w = w_uint8.to(torch.int64).view(k // 8, 8, n)
    out = torch.zeros(k // 8, n, dtype=torch.int32, device=w_uint8.device)
    for j in range(8):
        out |= (w[:, j, :] & 0xF).to(torch.int32) << (4 * j)
    return out


def unpack_int4_rows(q: torch.Tensor, n: int) -> torch.Tensor:
    """int32 [K, N/8] → uint8 [K, N]（pack_int4_rows 的逆）。"""
    k = q.shape[0]
    ql = q.to(torch.int64)
    out = torch.zeros(k, n, dtype=torch.uint8, device=q.device)
    v = out.view(k, n // 8, 8)
    for j in range(8):
        v[:, :, j] = ((ql >> (4 * j)) & 0xF).to(torch.uint8)
    return out


def unpack_int4_cols(q: torch.Tensor, k: int) -> torch.Tensor:
    """int32 [K/8, N] → uint8 [K, N]（pack_int4_cols 的逆）。"""
    n = q.shape[1]
    ql = q.to(torch.int64)
    out = torch.zeros(k, n, dtype=torch.uint8, device=q.device)
    v = out.view(k // 8, 8, n)
    for j in range(8):
        v[:, j, :] = ((ql >> (4 * j)) & 0xF).to(torch.uint8)
    return out


def unpack_int4_packed_cols(q: torch.Tensor, n: int) -> torch.Tensor:
    """int32 [*, N/8] → uint8 [*, N]（按最后一维解包；qzeros 的 GPTQ/AWQ 通用式）。"""
    shape = q.shape[:-1]
    ql = q.to(torch.int64)
    out = torch.zeros(*shape, n, dtype=torch.uint8, device=q.device)
    v = out.view(*shape, n // 8, 8)
    for j in range(8):
        v[..., j] = ((ql >> (4 * j)) & 0xF).to(torch.uint8)
    return out


@torch.no_grad()
def normalize_int4_to_in_major(
    qweight: torch.Tensor,
    qzeros: torch.Tensor | None,
    scales: torch.Tensor,
    *,
    group_size: int,
    method: str,
    awq_interleave: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """把 AWQ/GPTQ checkpoint 张量归一化到内部 in-major 布局。

    Args:
        qweight: [in, out/8]（in-major）或 [out, in/8]（AWQ 原始 out-major）。
        qzeros: [in/G, out/8] / [out, in/G/8] / [in/G, out/8]；None 表示对称。
        scales: [in/G, out] 或 [out, in/G]。
        group_size: 分组大小（128）。
        method: "awq" | "gptq"。
        awq_interleave: in-major 布局下是否按 AWQ 交织还原输出通道顺序。

    Returns:
        (qweight [in, out/8] int32, zeros uint8 [in/G, out] 或 None, scales [in/G, out])

    Raises:
        ValueError: 形状不匹配任一已知布局。
    """
    qw = qweight.to(torch.int32)
    sc = scales.float()
    zz = None if qzeros is None else qzeros.to(torch.int32)
    if sc.ndim != 2:
        raise ValueError(f"scales must be 2D, got {tuple(sc.shape)}")

    # 布局判定（两种布局形状互斥，可无歧义识别）
    if sc.shape[0] == qw.shape[0] and sc.shape[1] == qw.shape[1] * 8:
        # [in/G, out] + [in, out/8] → in-major（vLLM 内核约定）
        out_f = sc.shape[1]
        if method == "awq" and awq_interleave:
            in_f = qw.shape[0]
            w_u8 = unpack_int4_rows(qw, out_f)
            # 还原 AWQ 的 N 交织：nibble i 对应输出通道 8*n8 + AWQ_INTERLEAVE[i]
            w_u8 = w_u8.view(in_f, -1, 8)[:, :, _inverse_perm()].reshape(in_f, out_f)
            qw = pack_int4_rows(w_u8)
    elif sc.shape[0] == qw.shape[1] * 8 and sc.shape[1] == qw.shape[0]:
        # [out, in/G] + [out, in/8] → out-major（AutoAWQ 原始布局）
        out_f, in_packed = qw.shape
        in_f = in_packed * 8
        w_u8 = unpack_int4_cols(qw, in_f)           # [in, out]
        qw = pack_int4_rows(w_u8.t().contiguous())  # [in, out/8]
        sc = sc.t().contiguous()                    # [in/G, out]
    else:
        raise ValueError(
            "unsupported INT4 checkpoint layout: "
            f"qweight={tuple(qw.shape)} scales={tuple(sc.shape)}; "
            "expected [in, out/8]+[in/G, out] or [out, in/8]+[out, in/G]"
        )

    zeros_u8 = (
        None if zz is None else unpack_int4_packed_cols(zz, sc.shape[1])
    )  # [in/G, out]
    return qw.contiguous(), zeros_u8, sc.contiguous()


def _inverse_perm() -> list[int]:
    """AWQ_INTERLEAVE 的逆置换：把交织后的位置还原为原始输出通道序。"""
    inv = [0] * 8
    for i, p in enumerate(AWQ_INTERLEAVE):
        inv[p] = i
    return inv


def qweight_to_in_major(
    loaded: torch.Tensor,
    *,
    in_full: int,
    out_full: int,
    method: str,
    awq_interleave: bool = True,
) -> torch.Tensor:
    """checkpoint qweight → 内部 [in, out/8]。

    三种已知布局（形状唯一确定，按 method 优先匹配）：

    1. ``[in, out/8]``   AWQ in-major（输出通道在 int32 内交织，需还原）
    2. ``[out, in/8]``   AWQ out-major（AutoAWQ 原始，末维按输入打包）
    3. ``[in/8, out]``   GPTQ（沿输入 K 打包）
    """
    qw = loaded.to(torch.int32)
    cands: list[tuple[str, tuple[int, ...]]] = [
        ("in_major", (in_full, out_full // 8)),
        ("out_major", (out_full, in_full // 8)),
        ("gptq_rows", (in_full // 8, out_full)),
    ]
    if method == "gptq":
        cands = [cands[2], cands[0], cands[1]]
    for layout, shape in cands:
        if qw.shape != shape:
            continue
        if layout == "in_major":
            if method == "awq" and awq_interleave:
                w = unpack_int4_rows(qw, out_full)
                w = w.view(in_full, -1, 8)[:, :, _inverse_perm()].reshape(
                    in_full, out_full
                )
                qw = pack_int4_rows(w)
            return qw.contiguous()
        if layout == "out_major":
            # 末维按输入打包 → 沿末维解包得 [out, in]，转置后重打包
            w = unpack_int4_packed_cols(qw, in_full)
            return pack_int4_rows(w.t().contiguous())
        # GPTQ：沿首维（K）打包 → unpack 得 [in, out]
        return pack_int4_rows(unpack_int4_cols(qw, in_full))
    raise ValueError(
        f"unsupported AWQ/GPTQ qweight shape {tuple(qw.shape)} (method={method}); "
        f"expected {sorted({s for _, s in cands})}"
    )


def scales_to_in_major(
    loaded: torch.Tensor, *, in_full: int, out_full: int, group_size: int
) -> torch.Tensor:
    """checkpoint scales → 内部 [in/G, out]。"""
    sc = loaded.float()
    g = _cdiv_(in_full, group_size)
    if sc.shape == (g, out_full):
        return sc.contiguous()
    if sc.shape == (out_full, g):
        return sc.t().contiguous()
    raise ValueError(
        f"unsupported scales shape {tuple(sc.shape)}; expected "
        f"[{g}, {out_full}] or [{out_full}, {g}]"
    )


def qzeros_to_in_major(
    loaded: torch.Tensor,
    *,
    in_full: int,
    out_full: int,
    group_size: int,
    method: str = "awq",
    awq_interleave: bool = True,
) -> torch.Tensor:
    """checkpoint qzeros（int32 打包）→ 内部 [in/G, out/8]。

    两种布局的**打包轴不同**（in-major 打包输出通道，out-major 打包组），
    因此 out-major 需"解包 → 转置 → 重打包"，不能只靠转置。

    AWQ 的 qzeros 与 qweight 用同一组 nibble 顺序（vLLM 的 awq_gemm 内核
    对两者施加相同的 shifts），因此 in-major AWQ 同样要还原交织。
    """
    z = loaded.to(torch.int32)
    g = _cdiv_(in_full, group_size)
    if z.shape == (g, out_full // 8):
        if method == "awq" and awq_interleave:
            logical = unpack_int4_packed_cols(z, out_full)  # [g, out]
            logical = logical.view(g, -1, 8)[:, :, _inverse_perm()].reshape(
                g, out_full
            )
            return pack_int4_rows(logical)
        return z.contiguous()
    if g % 8 == 0 and z.shape == (out_full, g // 8):
        logical = unpack_int4_packed_cols(z, g)      # [out, g]（沿组维解包）
        return pack_int4_rows(logical.t().contiguous())  # [g, out/8]
    raise ValueError(
        f"unsupported qzeros shape {tuple(z.shape)}; expected "
        f"[{g}, {out_full // 8}] (in-major) or [{out_full}, {g // 8}] "
        f"(out-major，要求 in/G={g} 是 8 的倍数)"
    )


def _cdiv_(a: int, b: int) -> int:
    return (a + b - 1) // b


def dequantize_int4_reference(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    *,
    group_size: int,
    zero_bias: float = 0.0,
) -> torch.Tensor:
    """in-major 布局的参考反量化（fp32 [in, out]），供测试与调试对照。

    w = (q - zero_bias - z) * s
    """
    in_f = qweight.shape[0]
    out_f = scales.shape[1]
    q = unpack_int4_rows(qweight, out_f).to(torch.float32)  # [in, out]
    g = torch.arange(in_f) // group_size
    s = scales.float()[g]                                   # [in, out]
    if zeros is None:
        z = zero_bias
    else:
        z = zeros.float()[g]                                # [in, out]
        z = z + zero_bias
    return (q - z) * s


# ====================================================================== #
# 热路径：Triton fused dequant-GEMM（weight-only, W4A16）
# ====================================================================== #


@triton.jit
def _int4_gemm_kernel(
    a_ptr,
    q_ptr,
    s_ptr,
    z_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_qk,
    stride_qn,
    stride_sg,
    stride_sn,
    stride_zg,
    stride_zn,
    stride_cm,
    stride_cn,
    ZERO_BIAS: tl.constexpr,
    HAS_ZEROS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C[M,N] = A[M,K] @ dequant(qweight)[K,N]^T（激活保持 bf16/fp16）。

    要求 K 整除 BLOCK_K 且 BLOCK_K == group_size（一个 k-block 恰一个 scale 组）。
    权重以 int32 [K, N/8] 存储：每个 int32 的 8 个 nibble 是 8 个连续输出通道，
    用 ``[BK, BN8, 8]`` 三维 load + reshape 得到 [BK, BN]，无重复访存。
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)      # 输出通道（解包后）
    offs_n8 = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)  # 打包列
    m_mask = offs_m < M
    n_mask = offs_n < N
    n8_mask = offs_n8 < (N // 8)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    j = tl.arange(0, 8)
    shifts = (4 * j)[None, None, :]

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=m_mask[:, None],
            other=0.0,
        )
        # 权重：[BLOCK_K, BLOCK_N/8] int32 → [BLOCK_K, BLOCK_N] 解包
        w32 = tl.load(
            q_ptr + offs_k[:, None] * stride_qk + offs_n8[None, :] * stride_qn,
            mask=n8_mask[None, :],
            other=0,
        )
        q = (w32[:, :, None] >> shifts) & 0xF
        q = tl.reshape(q, (BLOCK_K, BLOCK_N)).to(tl.float32)

        g = k0 // BLOCK_K  # BLOCK_K == group_size ⇒ 一个 k-block 一个组
        s = tl.load(s_ptr + g * stride_sg + offs_n * stride_sn, mask=n_mask, other=0.0)
        if HAS_ZEROS:
            z = tl.load(
                z_ptr + g * stride_zg + offs_n * stride_zn, mask=n_mask, other=0.0
            ).to(tl.float32)
        else:
            z = tl.full((BLOCK_N,), ZERO_BIAS, dtype=tl.float32)
        w = (q - z[None, :]) * s[None, :]
        acc += tl.dot(a, w.to(a.dtype), out_dtype=tl.float32)

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


# 固定 tile 配置（有意不用 @triton.autotune）：
#   1) M 变化频繁（decode 每步 batch 不同），autotune 按 M 重调会累积编译与
#      基准测试开销，且在 CUDA Graph capture 期间做 do_bench 属非法同步；
#   2) 固定配置 + 掩码覆盖 M=1..prefill 全区间，行为确定、首次编译即稳定。
# 后续若要压榨吞吐，可在此处按 M 分档手工选择 tile（而非 autotune）。
BLOCK = 128  # 必须等于 group_size（一个 k-block 恰一个 scale 组）
BLOCK_M = 32
BLOCK_N = 64
WARPS = 4
STAGES = 3


def int4_weight_only_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    *,
    group_size: int = 128,
    zero_bias: float = 0.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """W4A16 GEMM：``x[M,K] @ dequant(qweight)[in=K, out=N]``（输出 [M, N]）。

    Args:
        x: [M, K] bf16/fp16 激活。
        qweight: int32 [K, N/8]（in-major，nibble j = 输出通道 8*n8+j）。
        scales: [K/G, N]。
        zeros: uint8/fp16 [K/G, N] 或 None（对称量化）。
        group_size: 必须等于 128（kernel 的 BLOCK_K）。
        zero_bias: 额外零点偏置（GPTQ sym 用 8）。
    """
    m, k = x.shape
    k_q, n8 = qweight.shape
    n = n8 * 8
    assert k_q == k, f"qweight K={k_q} != activation K={k}"
    assert group_size == BLOCK, f"only group_size={BLOCK} supported, got {group_size}"
    assert k % group_size == 0, f"K={k} must be divisible by {group_size}"
    assert scales.shape == (k // group_size, n), (
        f"scales {tuple(scales.shape)} != ({k // group_size}, {n})"
    )
    if zeros is not None:
        assert zeros.shape == (k // group_size, n), (
            f"zeros {tuple(zeros.shape)} != ({k // group_size}, {n})"
        )
    if out is None:
        out = torch.empty(m, n, dtype=x.dtype, device=x.device)
    if m == 0:
        return out
    num_pid_m = triton.cdiv(m, BLOCK_M)
    num_pid_n = triton.cdiv(n, BLOCK_N)
    grid = (num_pid_m * num_pid_n,)
    z = zeros if zeros is not None else scales  # 占位指针（HAS_ZEROS=False 时不读）
    _int4_gemm_kernel[grid](
        x,
        qweight,
        scales.float(),
        z.float() if zeros is not None else z,
        out,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        (zeros.stride(0) if zeros is not None else 0),
        (zeros.stride(1) if zeros is not None else 0),
        out.stride(0),
        out.stride(1),
        ZERO_BIAS=float(zero_bias),
        HAS_ZEROS=zeros is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK,
        num_warps=WARPS,
        num_stages=STAGES,
    )
    return out
