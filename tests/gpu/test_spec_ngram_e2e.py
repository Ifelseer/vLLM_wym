"""投机解码 Phase 1 的端到端 gate：greedy 下必须与 baseline **逐 token 一致**。

为什么这是唯一的硬指标：投机解码是**无损加速**——接受/拒绝只影响"一步算了
几个 token"，不允许改变输出。所以任何一处实现错（草稿位置偏移一位、KV 回滚
少/多一个、bonus token 取错位置），都会表现为 token 序列分叉。这比"吞吐变快"
更能证明正确性。

测试还要证明**不是空跑**：断言草稿被真正提出且被接受过（``spec_stats``），
否则一个永远返回空草稿的实现也能"逐位一致"地通过。

为了让 ngram 有命中机会，prompt 用「模型自己的输出」构造成重复文本：
先贪心生成 N 个 token，再把这段贴回 prompt 尾部，于是后缀匹配能取到
真正会出现的续写。
"""

from __future__ import annotations

import pytest

from tests.conftest import needs_cuda
from tests.helpers import DENSE_MODEL, require_model
from tests.helpers.engine import cfg, engine_session
from src import SamplingParams

pytestmark = [pytest.mark.gpu, needs_cuda, require_model(DENSE_MODEL)]

_N = 24
_SPEC_K = 4
_PROMPT = "List the first ten prime numbers, separated by commas: 2, 3, 5, 7,"


def _run(prompt: str | list[int], *, spec: int, n: int = _N):
    """返回 (prompt_ids, 生成的 token, 调度器投机统计)。"""
    with engine_session(cfg(DENSE_MODEL, max_num_seqs=1, spec_tokens=spec)) as eng:
        seq = eng.add_request(
            prompt,
            SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True),
        )
        while not seq.is_finished:
            eng.step()
        prompt_ids = list(seq.token_ids[: seq.num_prompt_tokens])
        out = list(seq.token_ids[seq.num_prompt_tokens :])
        stats = dict(eng.scheduler.spec_stats)
    return prompt_ids, out, stats


def test_spec_greedy_matches_baseline():
    prompt_ids, base, base_stats = _run(_PROMPT, spec=0)
    assert len(base) == _N
    assert base_stats["steps"] == 0, "spec_tokens=0 时不应走投机路径"

    # 用 baseline 的输出构造重复文本：后缀匹配能取到真正会出现的续写
    rep_prompt = prompt_ids + base + base[: _N // 2]

    _, spec_tokens, stats = _run(rep_prompt, spec=_SPEC_K, n=_N)
    assert stats["proposed"] > 0, "drafter never proposed — the test is vacuous"
    assert stats["accepted"] > 0, (
        "no draft was accepted; the repeated prompt is not reproducing the "
        "model's own continuation, so this run proves nothing about verify"
    )
    assert spec_tokens == base, (
        "speculative decode changed the output (must be lossless)\n"
        f"base={base}\nspec={spec_tokens}"
    )


def test_spec_off_is_deterministic_baseline():
    """spec_tokens=0 必须完全走原路径，且同输入两次结果一致。"""
    _, a, stats_a = _run(_PROMPT, spec=0)
    _, b, _ = _run(_PROMPT, spec=0)
    assert a == b
    assert stats_a == {"steps": 0, "proposed": 0, "accepted": 0}
