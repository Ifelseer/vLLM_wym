"""ZOS 的硬 gate：开/关两组的 greedy 输出必须**逐 token 相同**。

ZOS 只改变"什么时候提交"，不改变算出来的东西，所以输出必须完全一致。任何差异都说明
`advance_decode` 推进错了（位置 / slot / cache_seqlens 之一），而这类错误**不会报错**
——只会静默写到错的 KV 槽上。

除了逐 token 比对，还断言 `lookahead_submitted > 0`：否则等于 ZOS 从没真正提前提交
（比如每拍都被 drain 掉），测试会因为"没测到东西"而假通过。
"""

from __future__ import annotations

import pytest

from tests.conftest import needs_cuda
from tests.helpers import DENSE_MODEL, require_model
from tests.helpers.engine import cfg, engine_session
from src import SamplingParams

pytestmark = [pytest.mark.gpu, needs_cuda, require_model(DENSE_MODEL)]

_N = 48
_PROMPT = "Count down from 200 by threes, separated by commas: 200, 197, 194,"


def _run(*, zos: bool, n: int = _N):
    with engine_session(
        cfg(DENSE_MODEL, max_num_seqs=1, enable_zos=zos, enforce_eager=False)
    ) as eng:
        seq = eng.add_request(
            _PROMPT, SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        )
        while not seq.is_finished:
            eng.step()
        out = list(seq.token_ids[seq.num_prompt_tokens :])
        stats = eng.zos_stats()
    return out, stats


def test_zos_matches_sync_greedy():
    base, _ = _run(zos=False)
    assert len(base) == _N, f"baseline produced {len(base)} tokens"

    got, stats = _run(zos=True)
    assert stats, "enable_zos=True 时没有 AsyncDecode 实例（config.zos_enabled 没生效？）"
    assert stats["lookahead_submitted"] > 0, (
        "一次都没提前提交，说明流水线没跑起来（全被 drain 了）\n"
        f"stats={stats}"
    )
    assert got == base, (
        "ZOS 下 greedy 输出与同步路径不一致\n"
        f"first diff at "
        f"{next((i for i, (a, b) in enumerate(zip(base, got)) if a != b), None)}\n"
        f"base={base[:32]}\nzos ={got[:32]}\nstats={stats}"
    )
