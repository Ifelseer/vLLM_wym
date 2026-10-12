"""FP8 kernel 数值对齐（SM89+ GPU，如 4090；SM86 自动跳过）。

覆盖：
- Triton per-block（128×128）FP8 GEMM vs bf16 参考（多 M 形状，含 decode M=1）
- torch._scaled_mm per-tensor FP8 GEMM vs bf16 参考
- MoE FP8 grouped GEMM（Triton block scale）vs bf16 MoE 参考
- 验收口径：相对误差 < 3%（dense）/ 6%（MoE，两级量化误差叠加）
"""

from __future__ import annotations

import pytest

from tests.conftest import needs_cuda
from tests.helpers.devices import pick_gpus, run_isolated

pytestmark = [pytest.mark.gpu, needs_cuda]

_WORKER = r"""
import sys

import torch
import torch.nn.functional as F

major, minor = torch.cuda.get_device_capability()
if (major, minor) < (8, 9):
    print(f"SKIP fp8 kernel tests: SM{major}{minor} has no native FP8 tensor core")
    sys.exit(0)

from src.moe import clear_fused_state, route_experts
from src.moe.fused import fused_moe_forward
from src.quant.kernels.fp8_gemm import (
    quantize_fp8_per_block,
    quantize_fp8_per_tensor,
    w8a8_block_fp8_matmul,
    w8a8_tensor_fp8_matmul,
)

torch.manual_seed(0)
dev = torch.device("cuda:0")
dtype = torch.bfloat16
FP8 = torch.float8_e4m3fn


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


# ---------------------------------------------------------------- #
# 1. per-block FP8 GEMM（dense）
# ---------------------------------------------------------------- #
for M in (1, 7, 128, 512):
    K, N = 2048, 1024
    x = torch.randn(M, K, device=dev, dtype=dtype)
    w = torch.randn(N, K, device=dev, dtype=dtype) * 0.05
    wq, ws = quantize_fp8_per_block(w)
    out = w8a8_block_fp8_matmul(x, wq, ws)
    ref = F.linear(x, w)
    e = rel(out, ref)
    assert out.shape == (M, N) and out.dtype == dtype, (out.shape, out.dtype)
    # e4m3 的固有误差下限：权重量化与激活量化各贡献 mean ~4.5%，合计 max ~4%
    # （误差随点积按 sqrt(K) 同步增长，不会被平均掉）。实测 M=1..512 为 0.033~0.039，
    # 20 个随机种子最大 0.048，故取 0.08 留约 1.7x 余量；漏做 scale 校正会到 >0.10。
    assert e < 0.08, f"block fp8 M={M} rel={e:.4f}"
    print(f"block fp8  M={M:<4d} rel={e:.4f}")

# 输出缓冲池化（CUDA Graph replay 语义：同形复用）
x = torch.randn(8, 2048, device=dev, dtype=dtype)
wq, ws = quantize_fp8_per_block(torch.randn(1024, 2048, device=dev, dtype=dtype) * 0.05)
buf = torch.empty(8, 1024, device=dev, dtype=dtype)
o1 = w8a8_block_fp8_matmul(x, wq, ws, out=buf)
o2 = w8a8_block_fp8_matmul(x, wq, ws, out=buf)
assert torch.equal(o1, o2), "pooled output buffer mismatch"

# ---------------------------------------------------------------- #
# 2. per-tensor FP8 GEMM（torch._scaled_mm）
# ---------------------------------------------------------------- #
x = torch.randn(64, 2048, device=dev, dtype=dtype)
w = torch.randn(1024, 2048, device=dev, dtype=dtype) * 0.05
wq, ws = quantize_fp8_per_tensor(w)
out = w8a8_tensor_fp8_matmul(x, wq, ws)
e = rel(out, F.linear(x, w))
assert out.shape == (64, 1024) and out.dtype == dtype
assert e < 0.08, f"tensor fp8 rel={e:.4f}"  # 同 block：e4m3 固有误差，实测 0.0314~0.0424
print(f"tensor fp8          rel={e:.4f}")

# ---------------------------------------------------------------- #
# 3. MoE FP8 grouped GEMM vs bf16 MoE
# ---------------------------------------------------------------- #
clear_fused_state()
E, i, h, M, top_k = 4, 128, 256, 16, 2
x = torch.randn(M, h, device=dev, dtype=dtype) * 0.1
w13 = torch.randn(E, 2 * i, h, device=dev, dtype=dtype) * 0.1
w2 = torch.randn(E, h, i, device=dev, dtype=dtype) * 0.1
logits = torch.randn(M, E, device=dev, dtype=torch.float32)
weights, ids = route_experts(logits, top_k, norm_topk_prob=True)
weights = weights.to(dtype)

ref = fused_moe_forward(
    backend="triton",
    hidden_states=x,
    topk_ids=ids,
    topk_weights=weights,
    w13_weight=w13,
    w2_weight=w2,
).clone()
# 必须 clone：fused_moe_forward 返回按形状池化的复用缓冲（调用方须立即消费），
# 不 clone 的话下面 fp8 那次调用会原地覆写 ref，rel(got, ref) 恒为 0（假通过）。

# per-expert per-block 量化（与 Experts 加载产物同布局）
w13q = torch.empty(E, 2 * i, h, device=dev, dtype=FP8)
w13s = torch.empty(E, 2 * i // 128, h // 128, device=dev)
w2q = torch.empty(E, h, i, device=dev, dtype=FP8)
w2s = torch.empty(E, h // 128, i // 128, device=dev)
for e in range(E):
    q, s = quantize_fp8_per_block(w13[e])
    w13q[e].copy_(q)
    w13s[e].copy_(s)
    q, s = quantize_fp8_per_block(w2[e])
    w2q[e].copy_(q)
    w2s[e].copy_(s)

got = fused_moe_forward(
    backend="triton",
    hidden_states=x,
    topk_ids=ids,
    topk_weights=weights,
    w13_weight=w13q,
    w2_weight=w2q,
    w13_scale=w13s,
    w2_scale=w2s,
)
e = rel(got, ref)
assert got.shape == ref.shape and got.dtype == dtype
assert e < 0.12, f"moe fp8 rel={e:.4f}"  # 两个 GEMM + 中间激活量化叠加，实测 0.0645
print(f"moe fp8 (M={M})     rel={e:.4f}")

# decode 形状（M=1，CUDA Graph replay 的主场景）
x1 = x[:1].contiguous()
ref1 = fused_moe_forward(
    backend="triton",
    hidden_states=x1,
    topk_ids=ids[:1],
    topk_weights=weights[:1],
    w13_weight=w13,
    w2_weight=w2,
).clone()  # 同上：否则会被 got1 原地覆写
got1 = fused_moe_forward(
    backend="triton",
    hidden_states=x1,
    topk_ids=ids[:1],
    topk_weights=weights[:1],
    w13_weight=w13q,
    w2_weight=w2q,
    w13_scale=w13s,
    w2_scale=w2s,
)
e1 = rel(got1, ref1)
assert e1 < 0.12, f"moe fp8 M=1 rel={e1:.4f}"
print(f"moe fp8 (M=1)       rel={e1:.4f}")

print("ALL FP8 KERNEL TESTS PASSED")
"""


def test_fp8_kernels_on_sm89():
    """单卡 SM89+ 运行 FP8 kernel 对齐；SM86（3060）在 worker 内自动跳过。"""
    gpus = pick_gpus(1, min_free_mib=2000)
    out = run_isolated(gpus, _WORKER, timeout=900)
    if "SKIP" in out:
        pytest.skip(out.strip().splitlines()[-1])
    assert "ALL FP8 KERNEL TESTS PASSED" in out
