"""Real MoE weights: greedy smoke in an isolated GPU process."""

from __future__ import annotations

import pytest

from tests.conftest import needs_cuda
from tests.helpers import MOE_30B, MOE_MODEL, pick_gpus, require_model
from tests.helpers.devices import run_isolated

pytestmark = [pytest.mark.gpu, needs_cuda]

PROMPT = "The capital of France is"

_WORKER = r"""
import sys
from src.moe import resolve_moe_backend
from tests.helpers.engine import cfg, greedy_tokens
from tests.helpers.paths import MOE_MODEL, MOE_30B

kind, prompt, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
model = MOE_MODEL if kind == "a27" else MOE_30B
backend = resolve_moe_backend("triton")
tokens = greedy_tokens(
    cfg(
        model,
        device_ids=[0],
        max_num_seqs=1,
        context_len=512,
        num_kvcache_blocks=16,
        gpu_memory_utilization=0.85,
        moe_backend=backend,
    ),
    prompt,
    n=n,
)
assert len(tokens) == n, tokens
print("OK", tokens)
"""


@require_model(MOE_MODEL)
def test_a27_greedy_smoke():
    try:
        phys = pick_gpus(1, prefer="A100", min_free_mib=16000)
    except RuntimeError:
        phys = pick_gpus(1, prefer="3090", min_free_mib=16000)
    out = run_isolated(phys, _WORKER, "a27", PROMPT, "8", timeout=900)
    assert "OK" in out


@pytest.mark.slow
@require_model(MOE_30B)
def test_30b_a3b_greedy_smoke():
    try:
        phys = pick_gpus(1, prefer="A100", min_free_mib=40000)
    except RuntimeError:
        pytest.skip("need ~40GB free GPU for 30B-A3B")
    out = run_isolated(phys, _WORKER, "30b", PROMPT, "4", timeout=1200)
    assert "OK" in out
