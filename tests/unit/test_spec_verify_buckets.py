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

    def test_graph_plan_bounds_combinations(self):
        """分档计划必须只随 max_num_seqs 线性增长，不能爆炸。

        bs=1 铺全部 m（草稿数可变）；bs>1 只录 m=k+1（每条序列提满 k 个）。
        """

        def plan(spec_tokens: int, max_num_seqs: int) -> list[tuple[int, int]]:
            fake = SimpleNamespace(
                config=SimpleNamespace(
                    spec_tokens=spec_tokens, max_num_seqs=max_num_seqs
                )
            )
            return ModelRunner._verify_graph_plan(fake)

        # k=7：bs=1 有 m=2..8 共 7 个；bs=2..4 各 1 个
        p = plan(7, 4)
        assert len(p) == 7 + 3
        assert (1, 8) in p and (4, 8) in p
        assert (2, 3) not in p, "bs>1 只录 m=k+1"

        # 组合数 = (k) + (max_num_seqs-1)，与桶数相乘后仍然可控
        assert len(plan(7, 1)) == 7
        assert len(plan(4, 8)) == 4 + 7

    def test_sorted_and_unique(self):
        b = _bounds(2048)
        assert b == sorted(set(b))
        assert b[-1] == 2048, "最大一档必须等于配置上界，否则超出部分白回退"
