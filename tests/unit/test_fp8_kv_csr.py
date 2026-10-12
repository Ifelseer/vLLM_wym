"""FP8 KV cache 的 block_tables → FlashInfer CSR 转换（CPU 单测）。

转换是零同步的纯 GPU 张量运算，数学语义可在 CPU 上完整验证。
"""

from __future__ import annotations

import pytest
import torch

from src.compute.fp8_kv import block_tables_to_csr

PAGE = 256


def _run(block_tables: list[list[int]], seqlens: list[int]):
    bs, mb = len(block_tables), len(block_tables[0])
    bt = torch.tensor(block_tables, dtype=torch.int32)
    cs = torch.tensor(seqlens, dtype=torch.int32)
    indptr = torch.full((bs + 1,), -1, dtype=torch.int32)
    indices = torch.full((bs * mb + 1,), -1, dtype=torch.int32)  # +1 trash 槽
    last_len = torch.full((bs,), -1, dtype=torch.int32)
    block_tables_to_csr(bt, cs, PAGE, indptr, indices, last_len)
    total = int(indptr[-1])
    return indptr.tolist(), indices[:total].tolist(), last_len.tolist()


class TestBlockTablesToCsr:
    def test_multi_seq(self):
        # seq0: 300 tok → 2 pages, last=44；seq1: 100 tok → 1 page, last=100
        indptr, indices, last = _run(
            [[5, 6], [7, 0]], [300, 100]
        )
        assert indptr == [0, 2, 3]
        assert indices == [5, 6, 7]
        assert last == [44, 100]

    def test_exact_page_multiple(self):
        indptr, indices, last = _run([[9]], [256])
        assert indptr == [0, 1]
        assert indices == [9]
        assert last == [256]  # 满页：last_page_len == page_size（合法）

    def test_one_over_page(self):
        indptr, indices, last = _run([[3, 4]], [257])
        assert indptr == [0, 2]
        assert indices == [3, 4]
        assert last == [1]

    def test_single_token(self):
        indptr, indices, last = _run([[0]], [1])
        assert indptr == [0, 1]
        assert indices == [0]
        assert last == [1]

    def test_long_sequence(self):
        # 4096 tok → 16 pages
        bt = [list(range(100, 116))]
        indptr, indices, last = _run(bt, [4096])
        assert indptr == [0, 16]
        assert indices == list(range(100, 116))
        assert last == [256]

    def test_indices_beyond_indptr_untouched(self):
        """预分配缓冲中 indptr 之外的条目不得被依赖（可含陈旧值）。"""
        bs, mb = 2, 4
        bt = torch.tensor([[5, 6, 99, 99], [7, 99, 99, 99]], dtype=torch.int32)
        cs = torch.tensor([300, 100], dtype=torch.int32)
        indptr = torch.zeros(bs + 1, dtype=torch.int32)
        indices = torch.full((bs * mb + 1,), 12345, dtype=torch.int32)  # 陈旧值
        last_len = torch.zeros(bs, dtype=torch.int32)
        block_tables_to_csr(bt, cs, PAGE, indptr, indices, last_len)
        total = int(indptr[-1])
        assert total == 3
        # 紧凑 packing：前 total 项为两序列的页，无行间空洞
        assert indices[:total].tolist() == [5, 6, 7]
        # 缓冲尾部保持陈旧值（不被读取；trash 槽被清 0）
        assert indices[total:-1].tolist() == [12345] * (bs * mb - total)
        assert indices[-1].item() == 0

    def test_padded_rows_seqlens_one(self):
        """CG 补齐行（seqlens=1, page 0）产生合法 CSR。"""
        indptr, indices, last = _run([[0], [0]], [1, 1])
        assert indptr == [0, 1, 2]
        assert indices == [0, 0]
        assert last == [1, 1]

    def test_cumsum_dtype_int32(self):
        """indptr 必须为 int32（FlashInfer 要求）。"""
        bt = torch.tensor([[1, 2, 3]], dtype=torch.int32)  # 600 tok 需 3 页
        cs = torch.tensor([600], dtype=torch.int32)
        indptr = torch.zeros(2, dtype=torch.int32)
        indices = torch.zeros(4, dtype=torch.int32)  # 3 pages + trash
        last_len = torch.zeros(1, dtype=torch.int32)
        block_tables_to_csr(bt, cs, PAGE, indptr, indices, last_len)
        assert indptr.dtype == torch.int32
        assert indptr.tolist() == [0, 3]
        assert last_len.tolist() == [88]

    @pytest.mark.parametrize("seqlens", [[1], [128], [255], [256], [257], [512], [1024]])
    def test_invariants(self, seqlens):
        """不变量：1 <= last_page_len <= page_size；n_blocks == ceil(L/page)。"""
        mb = 8
        bt = [list(range(1, 1 + mb))]
        indptr, indices, last = _run(bt, seqlens)
        L = seqlens[0]
        expect_blocks = (L + PAGE - 1) // PAGE
        assert indptr == [0, expect_blocks]
        assert 1 <= last[0] <= PAGE
        assert last[0] == L - (expect_blocks - 1) * PAGE
