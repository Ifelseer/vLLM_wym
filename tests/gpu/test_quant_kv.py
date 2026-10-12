"""FP8 KV cache 数值对齐（SM89+ GPU，如 4090；SM86 自动跳过）。

三条对照（全部与 FA2 + BF16 KV 的参考实现比）：
1. decode：fp8 cache + FlashInfer paged decode vs FA2 flash_attn_with_kvcache
2. prefill（有前缀缓存）：fp8 cache + FlashInfer paged prefill vs FA2 varlen paged
3. 量化往返：store_kvcache_fp8 写入后反量化 ≈ 原始 bf16（scale 校准语义）

验收口径：rel = max|Δ|/max|ref|；e4m3 KV 的物理下限约 4~6%
（与 dense FP8 同级，见 docs/fp8-debug-report.md §4）。
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
    print(f"SKIP fp8 kv tests: SM{major}{minor} < SM89 (no FlashInfer fp8 KV)")
    sys.exit(0)

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)

from src.compute.fp8_kv import block_tables_to_csr
from src.compute.kv_ops import store_kvcache, store_kvcache_fp8

torch.manual_seed(0)
dev = torch.device("cuda:0")
dtype = torch.bfloat16
PAGE = 256
FP8 = torch.float8_e4m3fn
WS = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


def make_pool(num_blocks, nkv, hd, dt):
    return torch.empty(2, 1, num_blocks, PAGE, nkv, hd, dtype=dt, device=dev)


# ---------------------------------------------------------------- #
# 1. decode：fp8 cache + FlashInfer vs FA2 bf16 cache
# ---------------------------------------------------------------- #
bs, L, nkv, hd = 2, 300, 2, 128
H = 8
nblocks = 8
# 每序列 2 页（300 tok）；第二序列从 page 2 开始
block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=dev)
cache_seqlens = torch.tensor([L, L], dtype=torch.int32, device=dev)

k_full = torch.randn(bs, L, nkv, hd, device=dev, dtype=dtype) * 0.5
v_full = torch.randn(bs, L, nkv, hd, device=dev, dtype=dtype) * 0.5
q = torch.randn(bs, H, hd, device=dev, dtype=dtype)

# 参考：bf16 cache + FA2
pool_bf16 = make_pool(nblocks, nkv, hd, dtype)
slots = []
for i in range(bs):
    for t in range(L):
        b, off = t // PAGE, t % PAGE
        slots.append((i, block_tables[i, b].item() * PAGE + off))
slot_mapping = torch.tensor([s for _, s in slots], dtype=torch.int32, device=dev)
k_flat = torch.cat([k_full[i] for i in range(bs)], dim=0)
v_flat = torch.cat([v_full[i] for i in range(bs)], dim=0)
store_kvcache(
    k_flat, v_flat,
    pool_bf16[0, 0].reshape(-1, nkv, hd), pool_bf16[1, 0].reshape(-1, nkv, hd),
    slot_mapping,
)
ref = flash_attn_with_kvcache(
    q.unsqueeze(1), pool_bf16[0, 0], pool_bf16[1, 0],
    cache_seqlens=cache_seqlens, block_table=block_tables, causal=True,
).squeeze(1)

# fp8 cache + FlashInfer（固定 scale：amax/448*2，对齐 Fp8KVManager 语义）
pool_fp8 = make_pool(nblocks, nkv, hd, FP8)
amax = max(k_flat.abs().amax(), v_flat.abs().amax())
scale = (amax / 448.0 * 2.0).clamp(min=1e-4)
k_scale = scale.reshape(())
v_scale = scale.reshape(())
store_kvcache_fp8(
    k_flat, v_flat,
    pool_fp8[0, 0].reshape(-1, nkv, hd), pool_fp8[1, 0].reshape(-1, nkv, hd),
    slot_mapping, k_scale, v_scale,
)

indptr = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
indices = torch.zeros(bs * 4 + 1, dtype=torch.int32, device=dev)
last_len = torch.zeros(bs, dtype=torch.int32, device=dev)
block_tables_to_csr(block_tables, cache_seqlens, PAGE, indptr, indices, last_len)

dec = BatchDecodeWithPagedKVCacheWrapper(WS, kv_layout="NHD")
dec.plan(
    indptr, indices, last_len, H, nkv, hd, PAGE,
    q_data_type=dtype, kv_data_type=FP8, o_data_type=dtype,
)
got = dec.run(q, (pool_fp8[0, 0], pool_fp8[1, 0]), k_scale=k_scale, v_scale=v_scale)
e = rel(got, ref)
assert got.shape == ref.shape, (got.shape, ref.shape)
assert e < 0.08, f"fp8 kv decode rel={e:.4f}"
print(f"fp8 kv decode      rel={e:.4f}  (scale={scale.item():.5f})")

# ---------------------------------------------------------------- #
# 2. prefill（有前缀缓存）：fp8 cache + FlashInfer paged vs FA2 varlen paged
# ---------------------------------------------------------------- #
bs = 2
q_lens = [32, 16]
kv_lens = [160, 100]  # 前缀 + 本 chunk
cu_q = torch.tensor([0, 32, 48], dtype=torch.int32, device=dev)
cu_k = torch.tensor([0, 160, 260], dtype=torch.int32, device=dev)
block_tables_p = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=dev)
total_q = 48

q_p = torch.randn(total_q, H, hd, device=dev, dtype=dtype)
k_p = torch.randn(260, nkv, hd, device=dev, dtype=dtype) * 0.5
v_p = torch.randn(260, nkv, hd, device=dev, dtype=dtype) * 0.5

pool_bf16_p = make_pool(8, nkv, hd, dtype)
slots = []
for i in range(bs):
    for t in range(kv_lens[i]):
        b, off = t // PAGE, t % PAGE
        slots.append(block_tables_p[i, b].item() * PAGE + off)
store_kvcache(
    k_p, v_p,
    pool_bf16_p[0, 0].reshape(-1, nkv, hd), pool_bf16_p[1, 0].reshape(-1, nkv, hd),
    torch.tensor(slots, dtype=torch.int32, device=dev),
)
ref_p = flash_attn_varlen_func(
    q_p,
    pool_bf16_p[0, 0],
    pool_bf16_p[1, 0],
    cu_seqlens_q=cu_q,
    cu_seqlens_k=cu_k,
    max_seqlen_q=32,
    max_seqlen_k=160,
    causal=True,
    block_table=block_tables_p,
)
# 注意：FA2 传 block_table 时从 paged cache（pool_bf16_p）读，
# 与引擎 attention.py 的用法一致；cache 内容由上面的 store_kvcache 写入。

pool_fp8_p = make_pool(8, nkv, hd, FP8)
amax_p = max(k_p.abs().amax(), v_p.abs().amax())
scale_p = (amax_p / 448.0 * 2.0).clamp(min=1e-4)
store_kvcache_fp8(
    k_p, v_p,
    pool_fp8_p[0, 0].reshape(-1, nkv, hd), pool_fp8_p[1, 0].reshape(-1, nkv, hd),
    torch.tensor(slots, dtype=torch.int32, device=dev),
    scale_p.reshape(()), scale_p.reshape(()),
)
kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32, device=dev)
indptr_p = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
indices_p = torch.zeros(bs * 4 + 1, dtype=torch.int32, device=dev)
last_len_p = torch.zeros(bs, dtype=torch.int32, device=dev)
block_tables_to_csr(block_tables_p, kv_lens_t, PAGE, indptr_p, indices_p, last_len_p)

pre = BatchPrefillWithPagedKVCacheWrapper(WS, kv_layout="NHD")
pre.plan(
    cu_q, indptr_p, indices_p, last_len_p, H, nkv, hd, PAGE,
    causal=True, q_data_type=dtype, kv_data_type=FP8, o_data_type=dtype,
)
got_p = pre.run(
    q_p, (pool_fp8_p[0, 0], pool_fp8_p[1, 0]),
    k_scale=scale_p.reshape(()), v_scale=scale_p.reshape(()),
)
e_p = rel(got_p, ref_p)
assert got_p.shape == ref_p.shape, (got_p.shape, ref_p.shape)
assert e_p < 0.08, f"fp8 kv prefill rel={e_p:.4f}"
print(f"fp8 kv prefill     rel={e_p:.4f}")

# ---------------------------------------------------------------- #
# 3. 量化往返：store_fp8 → 反量化 ≈ 原始 bf16
# ---------------------------------------------------------------- #
k_rt = torch.randn(512, nkv, hd, device=dev, dtype=dtype) * 0.5
v_rt = torch.randn(512, nkv, hd, device=dev, dtype=dtype) * 0.5
pool_rt = make_pool(4, nkv, hd, FP8)
slot_rt = torch.arange(512, dtype=torch.int32, device=dev)
amax_rt = max(k_rt.abs().amax(), v_rt.abs().amax())
scale_rt = (amax_rt / 448.0 * 2.0).clamp(min=1e-4)
store_kvcache_fp8(
    k_rt, v_rt,
    pool_rt[0, 0].reshape(-1, nkv, hd), pool_rt[1, 0].reshape(-1, nkv, hd),
    slot_rt, scale_rt.reshape(()), scale_rt.reshape(()),
)
k_back = pool_rt[0, 0].reshape(-1, nkv, hd).float() * scale_rt
e_rt = rel(k_back, k_rt.float())
assert e_rt < 0.08, f"fp8 kv roundtrip rel={e_rt:.4f}"
print(f"fp8 kv roundtrip   rel={e_rt:.4f}")

print("ALL FP8 KV TESTS PASSED")
"""


def test_fp8_kv_cache_on_sm89():
    """单卡 SM89+ 运行 FP8 KV cache 对齐；SM86 在 worker 内自动跳过。"""
    gpus = pick_gpus(1, min_free_mib=2000)
    out = run_isolated(gpus, _WORKER, timeout=900)
    if "SKIP" in out:
        pytest.skip(out.strip().splitlines()[-1])
    assert "ALL FP8 KV TESTS PASSED" in out
