"""Fused MoE kernel vs naive + EP map (isolated non-Blackwell GPU)."""

from __future__ import annotations

import pytest

from tests.conftest import needs_cuda
from tests.helpers import pick_gpus
from tests.helpers.devices import run_isolated

pytestmark = [pytest.mark.gpu, needs_cuda]

_WORKER = r"""
import torch
import torch.nn.functional as F
from src.moe import clear_fused_state, make_expert_map, route_experts
from src.moe.fused import fused_moe_forward

def naive(hidden, topk_ids, topk_weights, w13, w2):
    m, h = hidden.shape
    top_k = topk_ids.shape[1]
    out = hidden.new_zeros(m, h)
    for t in range(m):
        acc = hidden.new_zeros(h, dtype=torch.float32)
        for k in range(top_k):
            eid = int(topk_ids[t, k].item())
            if eid < 0:
                continue
            x = hidden[t].float()
            gu = torch.nn.functional.linear(x, w13[eid].float())
            up, gate = gu.chunk(2, dim=-1)
            inter = up * F.silu(gate)
            y = torch.nn.functional.linear(inter, w2[eid].float())
            acc = acc + y * float(topk_weights[t, k])
        out[t] = acc.to(hidden.dtype)
    return out

clear_fused_state()
torch.manual_seed(0)
device = torch.device("cuda:0")
m, hidden, inter, n_exp, top_k = 8, 32, 64, 4, 2
dtype = torch.float16
x = torch.randn(m, hidden, device=device, dtype=dtype) * 0.1
w13 = torch.randn(n_exp, 2 * inter, hidden, device=device, dtype=dtype) * 0.1
w2 = torch.randn(n_exp, hidden, inter, device=device, dtype=dtype) * 0.1
logits = torch.randn(m, n_exp, device=device, dtype=torch.float32)
weights, ids = route_experts(logits, top_k, norm_topk_prob=True)
weights = weights.to(dtype)
ref = naive(x, ids, weights, w13, w2)
got = fused_moe_forward(
    backend="triton",
    hidden_states=x,
    topk_ids=ids,
    topk_weights=weights,
    w13_weight=w13,
    w2_weight=w2,
)
torch.testing.assert_close(got, ref, atol=5e-2, rtol=5e-2)

torch.manual_seed(1)
m, hidden, inter, n_exp, top_k, ep = 6, 32, 64, 8, 2, 2
x = torch.randn(m, hidden, device=device, dtype=dtype) * 0.1
w13 = torch.randn(n_exp, 2 * inter, hidden, device=device, dtype=dtype) * 0.1
w2 = torch.randn(n_exp, hidden, inter, device=device, dtype=dtype) * 0.1
logits = torch.randn(m, n_exp, device=device, dtype=torch.float32)
weights, ids = route_experts(logits, top_k, norm_topk_prob=True)
weights = weights.to(dtype)
full = fused_moe_forward(
    backend="triton",
    hidden_states=x,
    topk_ids=ids,
    topk_weights=weights,
    w13_weight=w13,
    w2_weight=w2,
)
acc = torch.zeros_like(full)
for rank in range(ep):
    mapping = make_expert_map(n_exp, ep, rank, device=device)
    local_ids = mapping[ids.long()]
    start, end = rank * (n_exp // ep), (rank + 1) * (n_exp // ep)
    acc = acc + fused_moe_forward(
        backend="triton",
        hidden_states=x,
        topk_ids=local_ids,
        topk_weights=weights,
        w13_weight=w13[start:end],
        w2_weight=w2[start:end],
    )
torch.testing.assert_close(acc, full, atol=5e-2, rtol=5e-2)
print("OK")
"""


def test_triton_fused_and_ep_map():
    phys = pick_gpus(1, prefer="3090", min_free_mib=2000)
    out = run_isolated(phys, _WORKER, timeout=300)
    assert "OK" in out
