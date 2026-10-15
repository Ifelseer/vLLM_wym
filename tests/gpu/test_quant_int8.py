"""INT8 W8A8 kernel 数值对齐（任意 SM61+ GPU，含本地 3060）。

对照方式：``w8a8_int8_matmul`` vs「per-token 量化 → per-channel 反量化 →
bf16 matmul」的参考实现。差值反映 kernel 实现（int32 累加 + scale 融合），
不掺入量化误差——量化误差由 unit 测试的往返覆盖。

验收：rel < 2%（int8 每元素 ~0.8% + bf16 输出舍入）。
"""

from __future__ import annotations

import pytest

from tests.conftest import needs_cuda
from tests.helpers.devices import pick_gpus, run_isolated

pytestmark = [pytest.mark.gpu, needs_cuda]

_WORKER = r"""
import sys

import torch

from src.quant.kernels.int8_gemm import (
    dequantize_per_channel,
    quantize_per_channel,
    quantize_per_token,
    w8a8_int8_matmul,
)

torch.manual_seed(0)
dev = torch.device("cuda:0")
DTYPE = torch.bfloat16


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-6)).item()


# ---------------------------------------------------------------- #
# 1. W8A8 GEMM vs 参考（含 decode M=1 与 prefill 大 M）
# ---------------------------------------------------------------- #
for M in (1, 2, 7, 33, 128, 512, 2049):
    K, N = 1024, 512
    x = torch.randn(M, K, device=dev, dtype=DTYPE) * 0.5
    w = torch.randn(N, K, device=dev, dtype=torch.float32) * 0.05  # [out, in]
    wq, ws = quantize_per_channel(w)

    # 参考：激活 per-token 量化 + 权重反量化 → matmul（注意 W 是 [out, in]，需转置）
    aq, asc = quantize_per_token(x)
    ref = (aq.float() * asc) @ dequantize_per_channel(wq, ws).t()
    ref = ref.to(DTYPE)
    got = w8a8_int8_matmul(x, wq, ws)

    assert got.shape == (M, N) and got.dtype == DTYPE, (got.shape, got.dtype)
    e = rel(got, ref)
    assert e < 0.02, f"int8 w8a8 M={M} rel={e:.4f}"
    print(f"int8 w8a8  M={M:<5d} rel={e:.5f}")

# ---------------------------------------------------------------- #
# 2. 输出缓冲池化（CUDA Graph replay 语义：同形复用）
# ---------------------------------------------------------------- #
K, N, M = 1024, 512, 8
x = torch.randn(M, K, device=dev, dtype=DTYPE)
w = torch.randn(N, K, device=dev, dtype=torch.float32) * 0.05
wq, ws = quantize_per_channel(w)
buf = torch.empty(M, N, device=dev, dtype=DTYPE)
o1 = w8a8_int8_matmul(x, wq, ws, out=buf)
o2 = w8a8_int8_matmul(x, wq, ws, out=buf)
assert torch.equal(o1, o2), "pooled output buffer mismatch"

# M=0 早退
empty = w8a8_int8_matmul(
    x[:0], wq, ws, out=torch.empty(0, N, device=dev, dtype=DTYPE)
)
assert empty.shape == (0, N)

# ---------------------------------------------------------------- #
# 3. 约束报错：K 不满足 _int_mm 的 4 对齐
# ---------------------------------------------------------------- #
try:
    bad_x = torch.randn(4, 1022, device=dev, dtype=DTYPE)
    bad_w = torch.randn(512, 1022, device=dev, dtype=DTYPE).to(torch.int8)
    bad_s = torch.ones(512, device=dev)
    w8a8_int8_matmul(bad_x, bad_w, bad_s)
    raise AssertionError("expected ValueError for unaligned K")
except ValueError as exc:
    assert "divisible by 4" in str(exc), exc
    print("int8 align  check ok")

# ---------------------------------------------------------------- #
# 4. 逐通道离群值：per-channel 权重显著优于 per-tensor
#    （验证量化粒度选择的有效性，而非 kernel 正确性）
# ---------------------------------------------------------------- #
K, N = 1024, 512
x = torch.randn(64, K, device=dev, dtype=DTYPE) * 0.5
w = torch.randn(N, K, device=dev, dtype=torch.float32) * 0.05
w[7] *= 40.0  # 离群输出通道
wq, ws = quantize_per_channel(w)
s_ts = w.abs().max().clamp(min=1e-12) / 127.0
q_ts = (w / s_ts).round().to(torch.int8)

got_pc = w8a8_int8_matmul(x, wq, ws)
ref_ts = (quantize_per_token(x)[0].float() * quantize_per_token(x)[1]) @ (
    q_ts.float() * s_ts
).t()
e_pc = rel(got_pc, x.float() @ dequantize_per_channel(wq, ws).t())
e_ts = rel(ref_ts, x.float() @ (q_ts.float() * s_ts).t())
assert e_pc < e_ts, f"per_channel={e_pc:.5f} should beat per_tensor={e_ts:.5f}"
print(f"int8 outlier per_channel={e_pc:.5f} < per_tensor={e_ts:.5f}")

print("ALL INT8 KERNEL TESTS PASSED")

# ---------------------------------------------------------------- #
# 5. 吞吐微基准（本地 3060）：bf16 GEMM vs int8 W8A8，双口径
#    - eager：含 kernel launch 开销（int8 路径 3 个 kernel vs cuBLAS 1 个）
#    - CG replay：引擎 decode 的真实形态（launch 开销被摊掉）
#    口径：单层 [N=4096, K=2048]（Qwen3-0.6B 的 o_proj/qkv 量级），
#    warmup 后 200 次平均。decode（M 小）看带宽，prefill（M 大）看算力。
# ---------------------------------------------------------------- #
import time

import torch.nn.functional as F

K, N = 2048, 4096
w = torch.randn(N, K, device=dev, dtype=torch.float32) * 0.05
wq, ws = quantize_per_channel(w)
w_bf16 = w.to(DTYPE)


def bench_eager(fn, iters: int = 200) -> float:
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def bench_cg(fn, iters: int = 200) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


print(f"{'M':>5} {'bf16 eager':>11} {'int8 eager':>11} {'bf16 CG':>9} {'int8 CG':>9} {'CG speedup':>11}")
for M in (1, 8, 128, 512):
    x = torch.randn(M, K, device=dev, dtype=DTYPE) * 0.5
    xc = x.clone()
    be = bench_eager(lambda: F.linear(x, w_bf16))
    ie = bench_eager(lambda: w8a8_int8_matmul(x, wq, ws))
    bc = bench_cg(lambda: F.linear(xc, w_bf16))
    ic = bench_cg(lambda: w8a8_int8_matmul(xc, wq, ws))
    print(
        f"{M:>5} {be:>9.1f}us {ie:>9.1f}us {bc:>7.1f}us {ic:>7.1f}us {bc / ic:>9.2f}x"
    )
print("ALL INT8 BENCH DONE")
"""


def test_int8_kernel_numerics():
    """任意支持 cuBLASLt int8 的 GPU 均可运行（含 3060/SM86）。

    末尾附带吞吐微基准（bf16 vs int8，多 M 形状），数据打印到 stdout。
    """
    gpus = pick_gpus(1, min_free_mib=1500)
    out = run_isolated(gpus, _WORKER, timeout=900)
    assert "ALL INT8 KERNEL TESTS PASSED" in out
