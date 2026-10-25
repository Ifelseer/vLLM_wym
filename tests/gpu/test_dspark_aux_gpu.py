"""aux hidden 的**逐位置稳定性**（不需要草稿模型，也不需要 vLLM 对照）。

要验证的性质：

> aux hidden 在位置 i 上只取决于 tokens[0..i] 与权重，**与批组成无关**。

于是可以构造一个不依赖任何外部参考的对照：同一段 token，

- A 路：一次 prefill 吃完 6 个 token；
- B 路：`max_num_batched_tokens=4` 迫使 chunked prefill，拆成 4 + 2 两次前向
  （第二段必然要读第一段写入的 KV——正是 verify 通路的行为）。

两条路的 aux 必须逐位置一致。**结构错（槽位错、行错位、`h+residual` 写成 `h`）
会产生 O(1) 的差异，一测就露**；而两条路的 attention kernel 调用形状不同，
数值上仍会有 ~1ulp 的抖动，所以用宽松容差（atol=0.1）——只抓结构错，不抓浮点。

为什么必须单独测这一条：如果只用"输出 token 对不对"来验，`h` 少加一次 residual
可能仍然生成通顺文本，接受率却会莫名偏低，几乎无法定位。
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import needs_cuda
from tests.helpers import DENSE_MODEL, require_model
from tests.helpers.engine import cfg, engine_session
from src import SamplingParams

pytestmark = [pytest.mark.gpu, needs_cuda, require_model(DENSE_MODEL)]

# 语义无所谓：aux 的稳定性与 token 内容无关，用任意合法 id 即可
_TOKENS = [100, 200, 300, 400, 500, 600]
_N = len(_TOKENS)
_ATOL = 0.1  # bf16 + 两种 attention 形状，抓结构错足够


def _runner(eng):
    """取到 in-process 的 ModelRunner（测试用白盒访问）。"""
    return eng._client.engine_core.executor.worker.model_runner


def _run(*, budget: int, snapshots: list[int]) -> list[torch.Tensor]:
    """跑一条 prompt，按快照点抓 aux（每次 eng.step() 后缓冲就是该步的 aux）。"""
    config = cfg(
        DENSE_MODEL,
        max_num_seqs=1,
        max_num_batched_tokens=budget,
        context_len=64,
        spec_tokens=1,
        spec_aux_hidden=True,
    )
    out: list[torch.Tensor] = []
    with engine_session(config) as eng:
        mr = _runner(eng)
        seq = eng.add_request(
            _TOKENS, SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        )
        for n_rows in snapshots:
            assert not seq.is_finished, "prompt 还没吃完就结束了"
            eng.step()
            buf = mr._aux_buf
            assert buf is not None, "aux 缓冲未分配（spec_aux_hidden 没生效？）"
            out.append(buf[:n_rows].clone())
    return out


def test_aux_is_position_stable_across_chunking():
    # A 路：一口气吃完（6 行）
    (full,) = _run(budget=64, snapshots=[_N])
    assert full.shape[0] == _N
    assert full.shape[1] % 5 == 0, "宽度必须是 5 层 hidden 的整数倍"
    assert torch.isfinite(full).all(), "aux 出现 NaN/Inf"
    assert full.abs().sum().item() > 0, "aux 全零 → 没采到"

    # B 路：chunk 4 + chunk 2
    a1, a2 = _run(budget=4, snapshots=[4, 2])

    assert torch.allclose(a1, full[:4], atol=_ATOL), (
        "第一段 chunk 与整段前向的 aux 不一致："
        f"max|diff|={(a1 - full[:4]).abs().max().item():.4f}"
    )
    assert torch.allclose(a2, full[4:], atol=_ATOL), (
        "第二段 chunk（读前段 KV）与整段前向的 aux 不一致 —— 高概率是"
        "`h + residual` 口径写错，或槽位/行错位："
        f"max|diff|={(a2 - full[4:]).abs().max().item():.4f}"
    )


def test_aux_slots_are_distinct():
    """不同层的槽位必须真的不同（否则说明 `_aux_slots` 只写了同一个槽）。"""
    (full,) = _run(budget=64, snapshots=[_N])
    hidden = full.shape[1] // 5
    layers = [full[:, i * hidden : (i + 1) * hidden] for i in range(5)]
    for i in range(5):
        for j in range(i + 1, 5):
            assert not torch.allclose(layers[i], layers[j], atol=1e-3), (
                f"第 {i} 个与第 {j} 个 aux 槽位完全相同 —— 槽位索引写错了"
            )

    # 同一槽位内不同位置也必须不同（行错位/广播错会满足前一条但违反这条）
    assert not torch.allclose(full[0], full[1], atol=1e-3)
