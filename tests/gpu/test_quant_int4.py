"""INT4 weight-only kernel 数值对齐（任意 SM，含本地 3060）。

对照方式：Triton fused dequant-GEMM vs「参考反量化 → bf16 matmul」。
参考侧用 dequantize_int4_reference（与 kernel 同布局语义，fp32 计算），
因此差值只反映 kernel 实现，不掺入量化误差。

验收：dequant-GEMM 与参考的相对误差 < 2%（同为 fp32 累加，差的主要是
输出 bf16 舍入与 tile 边界处理）。
"""

from __future__ import annotations

import pytest

from tests.conftest import needs_cuda
from tests.helpers.devices import pick_gpus, run_isolated

pytestmark = [pytest.mark.gpu, needs_cuda]

_WORKER = r"""
import sys

import torch

from src.quant.config import QuantConfig
from src.quant.kernels.int4_gemm import (
    dequantize_int4_reference,
    int4_weight_only_matmul,
    pack_int4_rows,
)

torch.manual_seed(0)
dev = torch.device("cuda:0")
DTYPE = torch.bfloat16
GROUP = 128


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-6)).item()


# --- 自建量化器（对齐 vLLM 参考语义） ---
def quantize_asym(w, group=GROUP):
    k, n = w.shape
    wg = w.view(k // group, group, n)
    wmax, wmin = wg.amax(dim=1), wg.amin(dim=1)
    s = (wmax - wmin).clamp(min=1e-5) / 15.0
    z = (wmin.abs() / s).round().clamp(0, 15)
    q = ((wg / s[:, None, :]).round() + z[:, None, :]).clamp(0, 15)
    return q.to(torch.uint8).view(k, n), s.float(), z.to(torch.uint8)


def quantize_sym(w, group=GROUP):
    k, n = w.shape
    wg = w.view(k // group, group, n)
    wmax, wmin = wg.amax(dim=1), wg.amin(dim=1)
    s = torch.maximum(wmax.abs() / 7.0, wmin.abs() / 8.0).clamp(min=1e-5)
    q = ((wg / s[:, None, :]).round() + 8).clamp(0, 15)
    return q.to(torch.uint8).view(k, n), s.float()


# ---------------------------------------------------------------- #
# 1. AWQ（非对称，带 zeros）
# ---------------------------------------------------------------- #
for M in (1, 7, 33, 128, 512):
    K, N = 1024, 512
    x = torch.randn(M, K, device=dev, dtype=DTYPE) * 0.5
    w = torch.randn(K, N, device=dev, dtype=torch.float32) * 0.05  # [in, out]
    q, s, z = quantize_asym(w)
    qw = pack_int4_rows(q).to(dev)
    s = s.to(dev)
    z = z.to(dev)

    ref_w = dequantize_int4_reference(qw.cpu(), s.cpu(), z.cpu(), group_size=GROUP).to(dev)
    ref = (x.float() @ ref_w).to(DTYPE)
    got = int4_weight_only_matmul(x, qw, s, z, group_size=GROUP)

    assert got.shape == (M, N) and got.dtype == DTYPE, (got.shape, got.dtype)
    e = rel(got, ref)
    assert e < 0.02, f"AWQ asym M={M} rel={e:.4f}"
    print(f"int4 awq   M={M:<4d} rel={e:.5f}")

# ---------------------------------------------------------------- #
# 2. GPTQ（对称 uint4b8：zero_bias=8，无 zeros）
# ---------------------------------------------------------------- #
for M in (1, 16, 128):
    K, N = 512, 1024
    x = torch.randn(M, K, device=dev, dtype=DTYPE) * 0.5
    w = torch.randn(K, N, device=dev, dtype=torch.float32) * 0.05
    q, s = quantize_sym(w)
    qw = pack_int4_rows(q).to(dev)
    s = s.to(dev)

    ref_w = dequantize_int4_reference(
        qw.cpu(), s.cpu(), None, group_size=GROUP, zero_bias=8.0
    ).to(dev)
    ref = (x.float() @ ref_w).to(DTYPE)
    got = int4_weight_only_matmul(
        x, qw, s, None, group_size=GROUP, zero_bias=8.0
    )
    e = rel(got, ref)
    assert e < 0.02, f"GPTQ sym M={M} rel={e:.4f}"
    print(f"int4 gptq  M={M:<4d} rel={e:.5f}")

# ---------------------------------------------------------------- #
# 3. 输出缓冲池化（CUDA Graph replay 语义：同形复用）
# ---------------------------------------------------------------- #
K, N, M = 1024, 512, 8
x = torch.randn(M, K, device=dev, dtype=DTYPE)
w = torch.randn(K, N, device=dev, dtype=torch.float32) * 0.05
q, s, z = quantize_asym(w)
qw = pack_int4_rows(q).to(dev)
s = s.to(dev)
z = z.to(dev)
buf = torch.empty(M, N, device=dev, dtype=DTYPE)
o1 = int4_weight_only_matmul(x, qw, s, z, group_size=GROUP, out=buf)
o2 = int4_weight_only_matmul(x, qw, s, z, group_size=GROUP, out=buf)
assert torch.equal(o1, o2), "pooled output buffer mismatch"

# M=0 早退
empty = int4_weight_only_matmul(
    x[:0], qw, s, z, group_size=GROUP, out=torch.empty(0, N, device=dev, dtype=DTYPE)
)
assert empty.shape == (0, N)

# ---------------------------------------------------------------- #
# 4. MoE grouped GEMM（INT4 vs 同管线的 BF16 权重参考）
#    本地 torch 2.11 与 moe_align_block_size 不兼容时自动跳过
# ---------------------------------------------------------------- #
try:
    from src.moe import clear_fused_state, route_experts
    from src.moe.experts import Experts
    from src.moe.fused import fused_moe_forward
    from src.moe.kernels.cuda import moe_align_block_size
    from src.moe.parallel import MoEParallelConfig

    probe = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=dev)
    moe_align_block_size(probe, 16, 4)
    torch.cuda.synchronize()
    MOE_OK = True
except Exception as exc:  # noqa: BLE001 - 环境探测，任何异常都跳过
    print(f"SKIP MoE INT4 tests: {type(exc).__name__}: {exc}")
    MOE_OK = False

if MOE_OK:
    E, h, i = 4, 256, 512
    torch.manual_seed(7)
    w13 = torch.randn(E, 2 * i, h, device=dev, dtype=DTYPE) * 0.05
    w2 = torch.randn(E, h, i, device=dev, dtype=DTYPE) * 0.05
    x = torch.randn(16, h, device=dev, dtype=DTYPE) * 0.5
    logits = torch.randn(16, E, device=dev, dtype=torch.float32)
    tw, tid = route_experts(logits, 2, norm_topk_prob=True)
    tw = tw.to(DTYPE)

    clear_fused_state()
    ref = fused_moe_forward(
        backend="triton",
        hidden_states=x,
        topk_ids=tid,
        topk_weights=tw,
        w13_weight=w13,
        w2_weight=w2,
    ).clone()  # 必须 clone：返回的是池化缓冲，下一次调用会原地覆写

    def load_int4_experts(method: str, sym: bool):
        par = MoEParallelConfig.single("triton")
        cfg = QuantConfig(
            method=method, bits=4, weight_scale="group", act_scale=None,
            group_size=GROUP, zero_point=not sym, sym=sym,
        )
        exp = Experts(E, h, i, par, quant=cfg).to(dev)
        up, gate = w13[:, :i, :], w13[:, i:, :]
        for e in range(E):
            for name, w in (("up", up[e]), ("gate", gate[e]), ("down", w2[e])):
                proj = f"{name}_proj.weight"
                if sym:
                    # GPTQ（uint4b8）：无 AWQ 交织，zeros 由 zero_bias=8 表达
                    q, s = quantize_sym(w.t().contiguous())
                    qw = pack_int4_rows(q)
                    zz = None
                else:
                    q, s, z = quantize_asym(w.t().contiguous())
                    qw = pack_int4_rows(q, interleave=True)
                    zz = pack_int4_rows(z, interleave=True)
                exp.weight_loader_int4(qw, proj, e, "qweight")
                exp.weight_loader_int4(s, proj, e, "scales")
                if zz is not None:
                    exp.weight_loader_int4(zz, proj, e, "qzeros")
        return exp

    for method, sym in (("awq", False), ("gptq", True)):
        exp = load_int4_experts(method, sym)
        got = exp(x, tid, tw)
        e = rel(got, ref)
        assert got.shape == ref.shape and got.dtype == DTYPE, (got.shape, ref.shape)
        # 界 0.25 的依据：INT4 group=128 每元素相对误差 ~6%（4 位 / 16 级），
        # 经 gate_up + silu + down 两级后 max 口径放大到 ~15~20%；实测 0.16。
        # 走错 scale 分支 / 张量布局错会给出 >0.5 的误差，仍能被拦住。
        assert e < 0.25, f"MoE int4 ({method}) rel={e:.4f}"
        print(f"int4 moe ({method:<4s})   rel={e:.5f}")

print("ALL INT4 KERNEL TESTS PASSED")
"""


def test_int4_kernel_numerics():
    """任意支持 Triton 的 GPU 均可运行 dense 段（含 3060/SM86）。

    MoE 段依赖 ``moe_align_block_size``（自定义 CUDA 扩展），worker 内探测，
    不可用时打印 SKIP 并跳过（本地 torch 2.11 已知不兼容）。
    """
    gpus = pick_gpus(1, min_free_mib=1500)
    out = run_isolated(gpus, _WORKER, timeout=900)
    if "SKIP" in out:
        import pytest

        skipped = [ln for ln in out.strip().splitlines() if "SKIP" in ln]
        if "ALL INT4 KERNEL TESTS PASSED" not in out:
            pytest.skip(skipped[-1] if skipped else out)
    assert "ALL INT4 KERNEL TESTS PASSED" in out
