"""设备侧接受裁决 `accept_greedy_device` 与 CPU 版 `accept_greedy` 的逐位等价。

这是 Z2（投机进 ZOS）的正确性 gate。两边只要有一处 off-by-one，症状就是**接受长度
差一个**：多接受会污染 KV 并在后续分叉，少接受只是白丢收益——**都不会报错**。
所以这里对 k ≤ 4 做全枚举，再补随机大 k。
"""

from __future__ import annotations

import itertools

import pytest
import torch

from src.spec.verify import accept_greedy, accept_greedy_device


def _compare(drafts_2d: torch.Tensor, preds_2d: torch.Tensor) -> None:
    next_tok, advances, tokens = accept_greedy_device(drafts_2d, preds_2d)
    b, k = drafts_2d.shape
    assert next_tok.shape == (b,)
    assert advances.shape == (b,) and advances.dtype == torch.int32
    assert tokens.shape == (b, k + 1)

    for r in range(b):
        d = drafts_2d[r].tolist()
        p = preds_2d[r].tolist()
        want_toks, want_acc = accept_greedy(d, p)
        j = want_acc
        assert int(advances[r]) == 1 + j, f"row{r} advances"
        assert int(next_tok[r]) == p[j], f"row{r} next_token 必须是 preds[j]"
        got = tokens[r, : j + 1].tolist()
        assert got == want_toks, f"row{r} tokens: got={got} want={want_toks}"
        assert int(tokens[r, j]) == p[j]


class TestAcceptGreedyDevice:
    @pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
    def test_exhaustive_binary(self, k):
        """k 个草稿 × 二值词表全枚举（2^k × 2^(k+1) 组）。"""
        drafts = torch.tensor(
            list(itertools.product([0, 1], repeat=k)) or [()], dtype=torch.int64
        )
        preds = torch.tensor(
            list(itertools.product([0, 1], repeat=k + 1)), dtype=torch.int64
        )
        # 笛卡尔积：每行是一个 (drafts, preds) 组合
        d = drafts.repeat_interleave(preds.shape[0], dim=0)
        p = preds.repeat(drafts.shape[0], 1)
        _compare(d, p)

    def test_all_accepted_takes_bonus(self):
        _compare(
            torch.tensor([[5, 6, 7]]),
            torch.tensor([[5, 6, 7, 99]]),
        )

    def test_first_rejected(self):
        _compare(torch.tensor([[5, 6, 7]]), torch.tensor([[0, 6, 7, 99]]))

    def test_middle_rejected(self):
        _compare(torch.tensor([[5, 6, 7]]), torch.tensor([[5, 0, 7, 99]]))

    def test_random_larger_k(self):
        gen = torch.Generator().manual_seed(0)
        b, k, vocab = 64, 7, 5
        drafts = torch.randint(0, vocab, (b, k), generator=gen, dtype=torch.int64)
        # 让一半的行在某个随机位置"故意不匹配"，覆盖不同前缀长度
        preds = torch.randint(0, vocab, (b, k + 1), generator=gen, dtype=torch.int64)
        preds[:, :k] = drafts.clone()
        break_pos = torch.randint(0, k, (b,), generator=gen)
        for r in range(b):
            preds[r, break_pos[r]] = (drafts[r, break_pos[r]] + 1) % vocab
        _compare(drafts, preds)

    def test_shape_validation(self):
        with pytest.raises(ValueError, match=r"\[B, k\+1\]"):
            accept_greedy_device(
                torch.zeros(2, 3, dtype=torch.int64),
                torch.zeros(2, 3, dtype=torch.int64),
            )
