"""`advance_reference` 的算术测试（纯 CPU，不需要 Triton / GPU）。

测的是**"设备侧推进"与"CPU 重建"逐位等价**这条不变量：`Batch.build_decode` 每轮用
`pos = num_cached_tokens` 重算全部字段；ZOS 改成在设备缓冲上 `pos += advances`。
两者只要有一处算错（尤其跨页时读错 page），KV 就会写到错的 slot 上，而**不会报错**
——只会静默污染别的序列的缓存。所以这里把边界情形逐个算死。
"""

from __future__ import annotations

import torch

from src.compute.decode_inputs import advance_reference

B = 4  # 每页 4 个 slot，便于制造跨页
ROWS = 3


def _state(pages: int = 8):
    """构造一套最小状态：ids/positions/slots/lengths/tables。"""
    ids = torch.zeros(ROWS, dtype=torch.int64)
    positions = torch.zeros(ROWS, dtype=torch.int64)
    slots = torch.full((ROWS,), -1, dtype=torch.int32)
    lengths = torch.zeros(ROWS, dtype=torch.int32)
    # 每行一个独立的连续页表（0,1,2,... 行间错开，便于发现串行）
    tables = torch.full((ROWS, pages), -1, dtype=torch.int32)
    for r in range(ROWS):
        tables[r, :] = torch.arange(pages, dtype=torch.int32) + r * pages
    return ids, positions, slots, lengths, tables


def _step(state, tokens, advances, bs=ROWS):
    ids, positions, slots, lengths, tables = state
    return advance_reference(
        tokens=tokens,
        advances=advances,
        ids=ids,
        positions=positions,
        slots=slots,
        lengths=lengths,
        tables=tables,
        bs=bs,
        block_size=B,
    )


class TestAdvanceReference:
    def test_three_steps_match_cpu_rebuild(self):
        """连续 3 步、每步 +1，与 CPU 侧 `pos += 1` 重建的结果完全一致。"""
        state = _state()
        expect_pos = [0, 0, 0]
        for step in range(3):
            toks = torch.tensor([10 + step, 20 + step, 30 + step], dtype=torch.int64)
            state = _step(state, toks, torch.ones(ROWS, dtype=torch.int32))
            for r in range(ROWS):
                expect_pos[r] += 1
                pos = expect_pos[r]
                page = state[4][r, pos // B].item()
                assert int(state[1][r]) == pos, f"row{r} step{step} position"
                assert int(state[2][r]) == page * B + pos % B, f"row{r} slot"
                assert int(state[3][r]) == pos + 1, f"row{r} cache_seqlens"
                assert int(state[0][r]) == 10 + r * 10 + step, "input_ids 应为该步 token"

    def test_cross_page_reads_new_page(self):
        """跨页那一步必须读**新页**的页号，而不是旧页。"""
        state = _state()
        # 先推到 pos=3（页 0 的最后一格）
        for _ in range(3):
            state = _step(state, torch.zeros(ROWS, dtype=torch.int64),
                          torch.ones(ROWS, dtype=torch.int32))
        assert [int(p) for p in state[1]] == [3, 3, 3]
        # 再推一步 → pos=4，落在页 1 的第 0 格
        state = _step(state, torch.zeros(ROWS, dtype=torch.int64),
                      torch.ones(ROWS, dtype=torch.int32))
        for r in range(ROWS):
            page = int(state[4][r, 1])
            assert int(state[1][r]) == 4
            assert int(state[2][r]) == page * B, "跨页后 slot 必须指向新页首格"

    def test_variable_advance_per_row(self):
        """投机：每行推进 1+j，各行可以不同，且互不干扰。"""
        state = _state()
        advances = torch.tensor([1, 3, 2], dtype=torch.int32)  # j = 0 / 2 / 1
        toks = torch.tensor([100, 200, 300], dtype=torch.int64)
        state = _step(state, toks, advances)
        assert [int(p) for p in state[1]] == [1, 3, 2]
        assert [int(p) for p in state[3]] == [2, 4, 3]  # pos + 1
        for r in range(ROWS):
            pos = [1, 3, 2][r]
            page = int(state[4][r, pos // B])
            assert int(state[2][r]) == page * B + pos % B
        # 第二拍继续往后推，验证状态是累加的
        state = _step(state, torch.zeros(ROWS, dtype=torch.int64),
                      torch.tensor([1, 1, 1], dtype=torch.int32))
        assert [int(p) for p in state[1]] == [2, 4, 3]

    def test_bs_smaller_than_capacity(self):
        """只推进前 bs 行时，其余行必须原封不动。"""
        state = _state()
        state[1][2] = 7  # 第三行预置一个值
        state = _step(
            state,
            torch.tensor([1, 2, 999], dtype=torch.int64),
            torch.ones(ROWS, dtype=torch.int32),
            bs=2,
        )
        assert [int(p) for p in state[1]] == [1, 1, 7], "第 3 行不该被推进"
        assert int(state[0][2]) == 0, "第 3 行的 input_ids 不该被写"

    def test_negative_advance_is_not_allowed_in_practice(self):
        """advance 必须 >= 1（0 会让位置原地不动，等于重复写同一格 KV）。

        这里只固定语义：参考实现在 advance=0 时位置不变、其余字段同步不变。
        """
        state = _state()
        state = _step(state, torch.tensor([5, 5, 5], dtype=torch.int64),
                      torch.zeros(ROWS, dtype=torch.int32))
        assert [int(p) for p in state[1]] == [0, 0, 0]
        assert [int(p) for p in state[3]] == [1, 1, 1]
