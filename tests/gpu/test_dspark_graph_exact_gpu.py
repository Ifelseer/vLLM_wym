"""DSpark 最严格的一条 gate：**verify 进图 + bit-exact** 下 greedy 逐位一致。

为什么单独测这个组合：

- `spec_verify_graph` 把 verify 录进 CUDA Graph，可能引入静态缓冲残留；
- `spec_exact_kv` 在 verify 之后逐位置用 decode 内核重写 KV（图外），消除
  "verify 写进 cache 的 KV 与 baseline 有 ~1ulp 差异"导致的并列翻转。

两者正交，叠加后是**目前唯一能声称"逐位无损"的配置**。如果这里逐位一致，
就说明：图里的静态缓冲干净、接受/回滚账本正确、且 KV 的数值路径与 baseline
一致——三条同时成立。

需要：Qwen3-8B + dspark_qwen3_8b_block7。用 FP8 目标把显存压到 ~16G。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.conftest import needs_cuda
from tests.helpers import DENSE_8B, require_model
from tests.helpers.engine import cfg, engine_session
from src import SamplingParams

DRAFT = Path(os.path.expanduser("~/huggingface/dspark_qwen3_8b_block7"))

pytestmark = [
    pytest.mark.gpu,
    needs_cuda,
    require_model(DENSE_8B),
    pytest.mark.skipif(
        not (DRAFT.is_dir() and any(DRAFT.glob("*.safetensors"))),
        reason=f"missing dspark draft: {DRAFT}",
    ),
]

_N = 96
_K = 7
_PROMPT = "List the first twenty prime numbers, separated by commas: 2, 3, 5, 7, 11,"


def _run(*, spec: int):
    """spec=0 → baseline；spec=7 → dspark + verify 图 + bit-exact。"""
    kw = dict(
        max_num_seqs=1,
        quantization="fp8",
        context_len=1024,
        num_kvcache_blocks=32,
    )
    if spec:
        kw.update(
            spec_tokens=spec,
            spec_draft="dspark",
            spec_draft_model=str(DRAFT),
            spec_verify_graph=True,
            spec_exact_kv=True,
        )
    with engine_session(cfg(DENSE_8B, **kw)) as eng:
        seq = eng.add_request(
            _PROMPT, SamplingParams(temperature=0.0, max_tokens=_N, ignore_eos=True)
        )
        while not seq.is_finished:
            eng.step()
        out = list(seq.token_ids[seq.num_prompt_tokens :])
        stats = dict(eng.scheduler.spec_stats)
    return out, stats


def test_graph_plus_exact_kv_is_bit_exact():
    base, _ = _run(spec=0)
    assert len(base) == _N, f"baseline produced {len(base)} tokens"

    spec_tokens, stats = _run(spec=_K)
    assert stats["steps"] > 0, "spec path was never taken"
    assert stats["accepted"] > 0, "no draft accepted — the prompt is not drafting"
    assert spec_tokens == base, (
        "verify-graph + exact-kv 下 greedy 仍分叉（应逐位一致）\n"
        f"first diff at {next((i for i, (a, b) in enumerate(zip(base, spec_tokens)) if a != b), None)}\n"
        f"base={base[:40]}\nspec={spec_tokens[:40]}"
    )
