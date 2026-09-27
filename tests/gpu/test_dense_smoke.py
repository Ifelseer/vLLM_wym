"""Dense Qwen3-0.6B smoke (isolated CUDA_VISIBLE_DEVICES)."""

from __future__ import annotations

import pytest

from tests.conftest import needs_cuda
from tests.helpers import DENSE_MODEL, pick_gpus, require_model
from tests.helpers.devices import run_isolated

pytestmark = [pytest.mark.gpu, needs_cuda, require_model(DENSE_MODEL)]

_WORKER = r"""
import sys
from tests.helpers.engine import cfg, greedy_tokens
from tests.helpers.paths import DENSE_MODEL
prompt = sys.argv[1]
n = int(sys.argv[2])
tokens = greedy_tokens(
    cfg(
        DENSE_MODEL,
        device_ids=[0],
        max_num_seqs=1,
        context_len=512,
        num_kvcache_blocks=16,
    ),
    prompt,
    n=n,
)
assert len(tokens) == n, tokens
print("OK", tokens)
"""


def test_dense_greedy_smoke():
    phys = pick_gpus(1, prefer="3090", min_free_mib=4000)
    out = run_isolated(phys, _WORKER, "The capital of France is", "8")
    assert "OK" in out
