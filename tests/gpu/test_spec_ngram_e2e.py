"""投机解码端到端的 gate：greedy 下必须与 baseline **逐 token 一致**。

为什么这是唯一的硬指标：投机解码是**无损加速**——接受/拒绝只影响"一步算了
几个 token"，不允许改变输出。所以任何一处实现错（草稿位置偏移一位、KV 回滚
少/多一个、bonus token 取错位置），都会表现为 token 序列分叉。

测试还要证明**不是空跑**：断言草稿被真正提出且被接受过（``spec_stats``），
否则一个永远返回空草稿的实现也能"逐位一致"地通过。

为了让 ngram 有命中机会，prompt 用「模型自己的输出」构造成重复文本：
先贪心生成 N 个 token，再把这段贴回 prompt 尾部，于是后缀匹配能取到
真正会出现的续写。

⚠️ 断言必须拿**同一个 rep_prompt 的 baseline** 做参照。早期版本错用了原始
prompt 的 baseline：rep_prompt 末尾重复的是 ``base[:N/2]``，模型自然会从
``base[N/2:]`` 继续写，所以 spec 与"原始 baseline"本来就不该相等。
"""

from __future__ import annotations

import pytest

from tests.conftest import needs_cuda
from tests.helpers import DENSE_8B, DENSE_MODEL, require_model
from tests.helpers.engine import cfg, engine_session
from src import SamplingParams

pytestmark = [pytest.mark.gpu, needs_cuda, require_model(DENSE_MODEL)]

_N = 24
_SPEC_K = 4
_PROMPT = "List the first ten prime numbers, separated by commas: 2, 3, 5, 7,"


def _run(
    prompt: str | list[int], *, spec: int, n: int = _N, tie_eps: float = 0.0,
    model=DENSE_MODEL,
):
    """返回 (prompt_ids, 生成的 token, 调度器投机统计)。"""
    with engine_session(
        cfg(model, max_num_seqs=1, spec_tokens=spec, spec_tie_eps=tie_eps)
    ) as eng:
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


def _rep_prompt(model=DENSE_MODEL) -> tuple[list[int], list[int]]:
    """构造重复 prompt：原 prompt + baseline 输出 + 其前一半。

    返回 (rep_prompt_ids, 该 prompt 上的 baseline 输出)。
    """
    prompt_ids, base, _ = _run(_PROMPT, spec=0, model=model)
    assert len(base) == _N, f"baseline produced {len(base)} tokens"
    rep = prompt_ids + base + base[: _N // 2]
    _, base_rep, _ = _run(rep, spec=0, model=model)
    return rep, base_rep


def test_spec_greedy_matches_baseline():
    rep, base_rep = _rep_prompt()

    _, spec_tokens, stats = _run(rep, spec=_SPEC_K, n=_N)
    assert stats["proposed"] > 0, "drafter never proposed — the test is vacuous"
    assert stats["accepted"] > 0, (
        "no draft was accepted; the repeated prompt is not reproducing the "
        "model's own continuation, so this run proves nothing about verify"
    )
    assert spec_tokens == base_rep, (
        "speculative decode changed the output (must be lossless)\n"
        f"base={base_rep}\nspec={spec_tokens}"
    )


@require_model(DENSE_8B)
def test_spec_greedy_matches_baseline_8b_tie_guard():
    """8B 上必须开 tie 守卫才能逐位一致（0.6B 不需要，但开着也无害）。

    8B 实测：某位置 top1 == top2（28.5 / 28.5），verify 的 paged-varlen 与
    baseline 的 ``flash_attn_with_kvcache`` 归约顺序不同，~1ulp 差异把逗号
    翻成句号——开 ``spec_tie_eps`` 后由 decode 内核重算该位置即可对齐。
    """
    rep, base_rep = _rep_prompt(model=DENSE_8B)

    _, spec_tokens, stats = _run(
        rep, spec=_SPEC_K, n=_N, tie_eps=0.01, model=DENSE_8B
    )
    assert stats["proposed"] > 0 and stats["accepted"] > 0
    assert spec_tokens == base_rep, (
        "8B greedy mismatch even with tie guard "
        f"(eps=0.01)\nbase={base_rep}\nspec={spec_tokens}"
    )


def test_spec_off_is_deterministic_baseline():
    """spec_tokens=0 必须完全走原路径，且同输入两次结果一致。"""
    _, a, stats_a = _run(_PROMPT, spec=0)
    _, b, _ = _run(_PROMPT, spec=0)
    assert a == b
    assert stats_a == {"steps": 0, "proposed": 0, "accepted": 0}


def _run_temp(prompt: str | list[int], *, spec: int, temperature: float, n: int = 16):
    with engine_session(cfg(DENSE_MODEL, max_num_seqs=1, spec_tokens=spec)) as eng:
        seq = eng.add_request(
            prompt,
            SamplingParams(temperature=temperature, max_tokens=n, ignore_eos=True),
        )
        while not seq.is_finished:
            eng.step()
        out = list(seq.token_ids[seq.num_prompt_tokens :])
        stats = dict(eng.scheduler.spec_stats)
    return out, stats


def test_spec_temperature_path_runs_and_keeps_invariants():
    """温度采样下的 smoke gate。

    分布等价性由 CPU 单测（``tests/unit/test_spec_rejection.py``）负责——
    那里直接验证"输出频次 == 目标分布"，且用的是 GPU 上同一个
    ``accept_rejection`` / ``Sampler.final_probs``，没有 GPU 专属分支。
    这里只保证：拒绝采样路径能跑通、token 数正确、账本不变量不破。
    """
    prompt_ids, base, _ = _run(_PROMPT, spec=0)
    rep = prompt_ids + base + base[:8]
    n = 16
    out, stats = _run_temp(rep, spec=_SPEC_K, temperature=0.8, n=n)
    assert len(out) == n
    assert stats["steps"] > 0, "spec path was never taken"
    assert stats["proposed"] > 0
