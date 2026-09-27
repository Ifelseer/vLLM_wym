"""MoE Triton fused path under CUDA Graph capture / replay."""

from __future__ import annotations

import pytest

from tests.conftest import needs_cuda
from tests.helpers import pick_gpus
from tests.helpers.devices import run_isolated

pytestmark = [pytest.mark.gpu, needs_cuda]

_WORKER = r"""
import torch
from src.moe import clear_fused_state, route_experts
from src.moe.fused import fused_moe_forward

clear_fused_state()
torch.manual_seed(0)
device = torch.device("cuda:0")
m, hidden, inter, n_exp, top_k = 8, 64, 128, 8, 2
dtype = torch.float16
x = torch.randn(m, hidden, device=device, dtype=dtype) * 0.05
w13 = torch.randn(n_exp, 2 * inter, hidden, device=device, dtype=dtype) * 0.05
w2 = torch.randn(n_exp, hidden, inter, device=device, dtype=dtype) * 0.05

def run(seed):
    torch.manual_seed(seed)
    logits = torch.randn(m, n_exp, device=device, dtype=torch.float32)
    weights, ids = route_experts(logits, top_k, norm_topk_prob=True)
    return fused_moe_forward(
        backend="triton",
        hidden_states=x,
        topk_ids=ids,
        topk_weights=weights.to(dtype),
        w13_weight=w13,
        w2_weight=w2,
    )

# Warmup fills workspace pools (align / silu / sum / gemm scratch).
eager0 = run(0).clone()
eager1 = run(1).clone()

g = torch.cuda.CUDAGraph()
# Capture with seed-0 routing; buffers already pooled.
torch.manual_seed(0)
logits = torch.randn(m, n_exp, device=device, dtype=torch.float32)
weights, ids = route_experts(logits, top_k, norm_topk_prob=True)
weights = weights.to(dtype)
# Static holders mutated before replay (same addresses as capture inputs).
x_cap = x.clone()
ids_cap = ids.clone()
w_cap = weights.clone()
with torch.cuda.graph(g):
    out_cap = fused_moe_forward(
        backend="triton",
        hidden_states=x_cap,
        topk_ids=ids_cap,
        topk_weights=w_cap,
        w13_weight=w13,
        w2_weight=w2,
    )
torch.testing.assert_close(out_cap, eager0, atol=5e-2, rtol=5e-2)

# Replay with a different routing (same shapes).
torch.manual_seed(1)
logits = torch.randn(m, n_exp, device=device, dtype=torch.float32)
weights, ids = route_experts(logits, top_k, norm_topk_prob=True)
ids_cap.copy_(ids)
w_cap.copy_(weights.to(dtype))
g.replay()
torch.testing.assert_close(out_cap, eager1, atol=5e-2, rtol=5e-2)
print("moe_cudagraph_ok", float(out_cap.abs().mean()))
"""


def test_moe_triton_cudagraph_replay():
    phys = pick_gpus(1, prefer="3090", min_free_mib=2000)
    out = run_isolated(phys, _WORKER, timeout=300)
    assert "moe_cudagraph_ok" in out
