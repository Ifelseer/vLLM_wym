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

print("ALL INT4 KERNEL TESTS PASSED")
"""


def test_int4_kernel_numerics():
    """任意支持 Triton 的 GPU 均可运行（含 3060/SM86）。"""
    gpus = pick_gpus(1, min_free_mib=1500)
    out = run_isolated(gpus, _WORKER, timeout=900)
    assert "ALL INT4 KERNEL TESTS PASSED" in out
