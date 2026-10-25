"""verify 图的桶划分（纯逻辑，无需 GPU）。

`max_seqlen_k` 是 attention kernel 的标量参数，录图后不可变，所以只能按长度分桶。
这里的规则是：取 256/512/1024/2048 中**不超过**配置上界的那几档，再补上配置值
本身——这样既不会出现"桶比上界还大"（浪费 tile），也不会漏掉非标准上界。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.compute.model_runner import ModelRunner


def _bounds(cap: int) -> list[int]:
    fake = SimpleNamespace(config=SimpleNamespace(spec_verify_graph_max_k=cap))
    return ModelRunner._verify_graph_bounds(fake)


class TestVerifyGraphBounds:
    @pytest.mark.parametrize(
        "cap, want",
        [
            (256, [256]),
            (512, [256, 512]),
            (1024, [256, 512, 1024]),
            (2048, [256, 512, 1024, 2048]),
            (300, [256, 300]),  # 非标准上界也要成为一档
            (100, [100]),  # 比最小档还小 → 只留它自己
        ],
    )
    def test_bounds(self, cap, want):
        assert _bounds(cap) == want

    def test_sorted_and_unique(self):
        b = _bounds(2048)
        assert b == sorted(set(b))
        assert b[-1] == 2048, "最大一档必须等于配置上界，否则超出部分白回退"
