"""MoE TP / EP: greedy tokens must match (one layout per subprocess)."""

from __future__ import annotations

import json

import pytest

from tests.conftest import needs_2gpu, needs_cuda
from tests.helpers import MOE_MODEL, pick_gpus, require_model
from tests.helpers.devices import run_isolated

pytestmark = [
    pytest.mark.distributed,
    needs_cuda,
    needs_2gpu,
    require_model(MOE_MODEL),
]

_WORKER = r"""
import json, sys
from src.moe import resolve_moe_backend
from tests.helpers.engine import cfg, greedy_tokens
from tests.helpers.parallel import layout_ok
from tests.helpers.paths import MOE_MODEL

prompt, n, tp_s, ep_s = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
tp, enable_ep = int(tp_s), ep_s == "1"
ok, reason = layout_ok(MOE_MODEL, tp=tp, enable_ep=enable_ep)
if not ok:
    raise SystemExit(f"bad layout: {reason}")
backend = resolve_moe_backend("triton")
tokens = greedy_tokens(
    cfg(
        MOE_MODEL,
        device_ids=list(range(tp)),
        tp_size=tp,
        enable_ep=enable_ep,
        max_num_seqs=1,
        context_len=512,
        num_kvcache_blocks=8,
        gpu_memory_utilization=0.70,
        moe_backend=backend,
        nccl_port=29620 + tp * 2 + int(enable_ep),
    ),
    prompt,
    n=n,
)
print(json.dumps({"tokens": tokens}))
"""


def _tokens(phys: list[int], *, tp: int, enable_ep: bool) -> list[int]:
    out = run_isolated(
        phys[:tp],
        _WORKER,
        "The capital of France is",
        "8",
        str(tp),
        "1" if enable_ep else "0",
        timeout=1200,
    )
    lines = [ln for ln in out.splitlines() if ln.strip().startswith("{")]
    assert lines, out
    return json.loads(lines[-1])["tokens"]


def test_moe_tp1_tp2_ep_match():
    # A2.7B ≈22GB：单卡 tp1 优先 A100；tp2/ep 用两张同型号 3090。
    try:
        tp1_devs = pick_gpus(1, prefer="A100", min_free_mib=30000)
    except RuntimeError:
        tp1_devs = pick_gpus(1, prefer="3090", min_free_mib=20000)
    pair = pick_gpus(2, prefer="3090", min_free_mib=18000)
    ref = _tokens(tp1_devs, tp=1, enable_ep=False)
    tp2 = _tokens(pair, tp=2, enable_ep=False)
    ep2 = _tokens(pair, tp=2, enable_ep=True)
    assert ref == tp2 == ep2, f"tp1={ref} tp2={tp2} ep2={ep2}"
