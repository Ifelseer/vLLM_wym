"""页式草稿 KV 的不变量测试（纯 CPU，不需要 Triton / GPU）。

三条不变量各对应一条测试，另加页号编码与「页池不足」的降级行为：

1. **页表覆盖的页数必须够放下逻辑长度** —— 不够时读到的就是空页；
2. **crop 必须退还尾部整页** —— 只做逻辑缩短会让页数按 `m` 增长，浪费照旧；
3. **未参与本步的行绝不能被推进或改写** —— 行池长期持有，每步参与者不同。

写错这三条中的任何一条，症状都是「accept_len 悄悄下滑」而不是报错 —— 所以必须有
CPU 侧的确定性断言。
"""

from __future__ import annotations

import torch

from src.spec.dspark.paged_cache import (
    TIER_CPU,
    decode_page,
    encode_page,
    PagedDraftKV,
)

HEADS = 1
DIM = 2
PAGE = 4


def _kv(n_rows: int, length: int, base: float, heads: int = HEADS) -> torch.Tensor:
    """造一个 ``[B, heads, L, dim]``，值可辨认（base + 位置）。"""
    t = torch.zeros(n_rows, heads, length, DIM)
    for r in range(n_rows):
        for i in range(length):
            t[r, :, i, :] = base + r * 100 + i
    return t


def _cache(*, rows: int = 2, pages: int = 8, layers: int = 1) -> PagedDraftKV:
    return PagedDraftKV(
        num_layers=layers,
        num_kv_heads=HEADS,
        head_dim=DIM,
        num_rows=rows,
        num_pages=pages,
        page_size=PAGE,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


class TestPageEncoding:
    def test_roundtrip_and_high_bits(self):
        assert encode_page(0) == 0
        assert encode_page(5) == 5
        v = encode_page(5, tier=TIER_CPU, device=3)
        assert decode_page(v) == (5, TIER_CPU, 3)
        assert decode_page(-1) == (-1, -1, -1)

    def test_out_of_range_rejected(self):
        import pytest

        with pytest.raises(AssertionError):
            encode_page(1 << 36)
        with pytest.raises(AssertionError):
            encode_page(1, device=256)


class TestSeedAndAlloc:
    def test_seed_allocates_ceil_pages(self):
        c = _cache()
        assert c.num_free == 8
        assert c.seed_row(0, 5) is True          # ceil(5/4) = 2 页
        assert c.num_free == 6
        assert c.row_len(0) == 5
        c.seed_row(0, 3)                          # 缩到 3 → 1 页
        assert c.num_free == 7

    def test_reset_returns_all_pages(self):
        c = _cache()
        c.seed_row(0, 8)
        assert c.num_free == 6
        c.reset_row(0)
        assert c.num_free == 8
        assert c.row_len(0) == 0

    def test_pool_exhausted_returns_false(self):
        c = _cache(pages=1)
        assert c.seed_row(0, 5) is False          # 需要 2 页，池只有 1 页
        assert c.row_len(0) == 0


class TestWriteAndRead:
    def test_ctx_and_block_land_at_logical_positions(self):
        c = _cache()
        c.seed_row(0, 3)                          # 已有 3 个位置
        assert c.begin_step([0], [2], seg_max=2, block=2) == 3 + 2 + 2
        k = _kv(1, 4, base=10.0, heads=HEADS)     # [1,1,4,2]：ctx 0..1 + block 2..3
        v = k + 0.5
        dk, dv = c.update(k, v, 0)
        assert dk.shape == (1, HEADS, 7, DIM)
        # 逻辑 3..4 = ctx 源 0..1；逻辑 5..6 = block 源 2..3
        assert torch.equal(dk[0, :, 3:5], k[0, :, 0:2])
        assert torch.equal(dk[0, :, 5:7], k[0, :, 2:4])
        assert torch.equal(dv[0, :, 3:5], v[0, :, 0:2])

    def test_pad_rows_are_not_written(self):
        """seg=1 但 seg_max=3：源 1..2 是 pad，绝不能落到逻辑位置上。"""
        c = _cache()
        c.begin_step([0], [1], seg_max=3, block=2)
        k = _kv(1, 5, base=0.0)
        k[0, :, 1:3, :] = -999.0                  # 标记 pad 段
        dk, _ = c.update(k, _kv(1, 5, base=0.0), 0)
        assert not bool((dk == -999.0).any()), "pad 段被写进了逻辑位置"

    def test_rows_isolated(self):
        """两行各写各的，不能串位。"""
        c = _cache()
        c.begin_step([0, 1], [1, 1], seg_max=1, block=1)
        k = _kv(2, 2, base=0.0)
        k[0] = 1.0
        k[1] = 2.0
        dk, _ = c.update(k, k, 0)
        assert torch.all(dk[0] == 1.0)
        assert torch.all(dk[1] == 2.0)


class TestMask:
    def test_pad_region_masked(self):
        """矩形读会带出短行的 pad 区，掩码必须把 `[lens+seg+block, kv_len)` 遮掉。"""
        c = _cache()
        c.begin_step([0, 1], [1, 1], seg_max=3, block=2)   # kv_len = 3+2 = 5
        m = c.mask(q_len=2)
        assert m.shape == (2, 1, 2, 5)
        for r in range(2):
            assert not torch.isneginf(m[r, 0, 0, :3]).any(), "有效范围不能被遮"
            assert torch.isneginf(m[r, 0, 0, 3:]).all(), "pad 区必须遮掉"

    def test_mask_follows_per_row_length(self):
        c = _cache()
        c.seed_row(0, 2)                                   # 行 0 已有 2 个位置
        c.begin_step([0, 1], [1, 1], seg_max=2, block=1)   # kv_len = 2+2+1 = 5
        m = c.mask(q_len=1)
        assert not torch.isneginf(m[0, 0, 0, : 2 + 1 + 1]).any(), "行 0 有效到 4"
        assert torch.isneginf(m[0, 0, 0, 4]), "行 0 的第 5 个位置无效"
        assert not torch.isneginf(m[1, 0, 0, : 0 + 1 + 1]).any(), "行 1 有效到 2"
        assert torch.isneginf(m[1, 0, 0, 2:]).all()


class TestCrop:
    def test_crop_releases_tail_pages(self):
        """block 段用完即弃：3 个 ctx 只要 1 页，crop 后必须退还另一页。"""
        c = _cache()
        before = c.num_free
        c.begin_step([0], [3], seg_max=3, block=2)   # want=5 → 2 页
        assert c.num_free == before - 2
        c.update(_kv(1, 5, 0.0), _kv(1, 5, 0.0), 0)
        c.crop()
        assert c.row_len(0) == 3, "长度应推进到 已有+seg"
        assert c.num_free == before - 1, "尾部整页必须退还"

    def test_crop_keeps_partial_page(self):
        """跨页那一页即使只用到一部分也必须保留（尾部浪费 ≤1 页）。"""
        c = _cache()
        c.begin_step([0], [5], seg_max=5, block=1)   # want=6 → 2 页
        c.update(_kv(1, 6, 0.0), _kv(1, 6, 0.0), 0)
        c.crop()
        assert c.row_len(0) == 5
        assert c.num_free == 8 - 2, "5 个 token 需要 2 页，不能退"

    def test_second_step_continues(self):
        c = _cache()
        c.begin_step([0], [3], seg_max=3, block=2)
        c.update(_kv(1, 5, 0.0), _kv(1, 5, 0.0), 0)
        c.crop()
        # 第二步：从 3 继续，seg=2、block=2
        assert c.begin_step([0], [2], seg_max=2, block=2) == 3 + 2 + 2
        k = _kv(1, 4, base=10.0)
        dk, _ = c.update(k, k, 0)
        assert torch.equal(dk[0, :, 3:5], k[0, :, 0:2]), "第二步 ctx 应从 3 开始"
        assert torch.equal(dk[0, :, 5:7], k[0, :, 2:4])


class TestRowIsolation:
    def test_unparticipating_row_untouched(self):
        c = _cache(rows=3)
        c.seed_row(1, 4)
        c.begin_step([0], [2], seg_max=2, block=1)
        c.update(_kv(1, 3, 0.0), _kv(1, 3, 0.0), 0)
        c.crop()
        assert c.row_len(1) == 4, "未参与的行不能被 crop 推进"
        assert c.row_len(0) == 2

    def test_begin_step_drops_row_when_pool_short(self):
        c = _cache(pages=1)
        kv_len = c.begin_step([0], [8], seg_max=8, block=1)
        assert kv_len == 0
        assert c._rows == [], "页不够应把该行移出本步，而不是抛错"
