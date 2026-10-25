"""batch>1 下的无损性 gate：多条序列同时投机解码，逐条都必须与 baseline 一致。

为什么单独测：单序列时"位置偏移"只会影响自己；多序列时 verify 把每条序列的
`[anchor + drafts]` 按 `cu_seqlens` **拼平**做一次前向，偏移会串到相邻序列——
症状是"某条序列从第 N 个 token 起整体错位"，而单序列测试永远发现不了。

prompt 故意选两条不同内容，避免"两条正好同分布"掩盖问题。
"""

from __future__ import annotations

import pytest

from tests.conftest import needs_cuda
from tests.helpers import DENSE_MODEL, require_model
from tests.helpers.engine import cfg, engine_session
from src import SamplingParams

pytestmark = [pytest.mark.gpu, needs_cuda, require_model(DENSE_MODEL)]

_N = 24
_K = 4
_PROMPTS = [
    "List the first ten prime numbers, separated by commas: 2, 3, 5, 7,",
    "Count down from 100 by sevens: 100, 93, 86, 79,",
]


def _run(*, spec: int) -> tuple[list[list[int]], dict]:
    """两条序列同时跑；返回各自的生成结果与投机统计。"""
    with engine_session(
        cfg(DENSE_MODEL, max_num_seqs=len(_PROMPTS), spec_tokens=spec)
    ) as eng:
        seqs = [
            eng.add_request(
                p, SamplingParams(temperature=0.0, max_tokens=_N, ignore_eos=True)
            )
            for p in _PROMPTS
        ]
        while not all(s.is_finished for s in seqs):
            eng.step()
        outs = [
            list(s.token_ids[s.num_prompt_tokens :]) for s in seqs
        ]
        stats = dict(eng.scheduler.spec_stats)
    return outs, stats


def test_multi_sequence_greedy_matches_baseline():
    base, _ = _run(spec=0)
    assert all(len(o) == _N for o in base), f"baseline lengths: {[len(o) for o in base]}"

    spec_outs, stats = _run(spec=_K)
    assert stats["steps"] > 0, "spec path was never taken"
    assert stats["accepted"] > 0, "no draft accepted — prompt not repeating"

    for i, (b, s) in enumerate(zip(base, spec_outs)):
        assert s == b, (
            f"sequence {i} diverged under multi-seq spec decode\n"
            f"base={b}\nspec={s}"
        )
