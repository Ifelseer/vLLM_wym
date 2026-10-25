"""草稿批量化的 KV 组件（纯 CPU，随机张量）。

只测四条**不变量**——它们写错时不会报错，只会让草稿看到错的历史/噪声，
表现为接受率无声下滑：

1. 每行的新 K/V 落在自己的偏移上，不串行；
2. **未参与本步的行不推进**（否则它的位置整体错一格）；
3. padding 在 mask 里是 -inf（不是 0）；
4. crop 之后每行只保留 `已有 + seg`，块内候选不残留。

不测"数值对不对"——那是 GPU 上接进模型后的事，这里测的是账本。
"""

from __future__ import annotations

import torch

from src.spec.dspark.padded_cache import PaddedKVCache

H = 2
D = 4
LAYERS = 2
MAX_LEN = 32
ROWS = 4


def _cache() -> PaddedKVCache:
    return PaddedKVCache(
        num_layers=LAYERS,
        num_kv_heads=H,
        head_dim=D,
        num_rows=ROWS,
        max_len=MAX_LEN,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _kv(n: int, length: int, fill: float) -> torch.Tensor:
    return torch.full((n, H, length, D), fill)


class TestPaddedKVCache:
    def test_per_row_offsets(self):
        """三行历史长度不同：写入后每行只能在自己的区间里看到自己的值。"""
        c = _cache()
        for r, n in enumerate((3, 5, 0)):
            c.seed_row(r, n)
        rows, segs = [0, 1, 2], [2, 1, 2]
        c.begin_step(rows, segs, seg_max=2, block=3)
        k = torch.stack([_kv(1, 5, float(r + 1))[0] for r in rows])
        kk, _ = c.update(k, k.clone(), 0, rows)
        assert torch.all(kk[0, :, :3] == 0) and torch.all(kk[0, :, 3:8] == 1.0)
        assert torch.all(kk[1, :, :5] == 0) and torch.all(kk[1, :, 5:10] == 2.0)
        assert torch.all(kk[2, :, :5] == 3.0)

    def test_non_participating_rows_do_not_advance(self):
        """只推进参与行：行 3 不参与，lens 必须保持原值，且有效位不被改动。"""
        c = _cache()
        c.seed_row(3, 7)
        rows = [0, 1]
        c.begin_step(rows, [1, 1], seg_max=1, block=2)
        c.update(_kv(2, 3, 1.0), _kv(2, 3, 1.0), 0, rows)
        c.crop(rows, [1, 1])
        assert c.row_len(0) == 0 + 1 and c.row_len(1) == 1
        assert c.row_len(3) == 7, "未参与的行不能被推进"
        assert bool(c.valid[3, :7].all()) and not bool(c.valid[3, 7:].any())

    def test_mask_marks_padding_neg_inf(self):
        c = _cache()
        rows = [0, 1]
        # seg_max=3：行 0 有 2 个真实 ctx（1 pad），行 1 只有 1 个（2 pad）
        c.begin_step(rows, [2, 1], seg_max=3, block=2)
        c.update(_kv(2, 5, 1.0), _kv(2, 5, 1.0), 0, rows)
        m = c.mask(rows, kv_len=5, q_len=2)
        assert m.shape == (2, 1, 2, 5)
        assert not torch.isneginf(m[0, 0, 0, 1])
        assert torch.isneginf(m[0, 0, 0, 2]), "seg_max 尾部的 pad 必须被 mask"
        assert not torch.isneginf(m[0, 0, 0, 4])
        assert torch.isneginf(m[1, 0, 0, 1]) and torch.isneginf(m[1, 0, 0, 2])
        assert not torch.isneginf(m[1, 0, 0, 3])

    def test_crop_keeps_only_context(self):
        c = _cache()
        c.seed_row(0, 4)
        c.seed_row(1, 2)
        rows = [0, 1]
        c.begin_step(rows, [2, 2], seg_max=2, block=3)
        c.update(_kv(2, 5, 1.0), _kv(2, 5, 1.0), 0, rows)
        c.crop(rows, [2, 2])
        assert [c.row_len(0), c.row_len(1)] == [6, 4], "应保留 已有长度 + seg"
        assert bool(c.valid[0, :6].all()) and not bool(c.valid[0, 6:].any())
        assert bool(c.valid[1, :4].all()) and not bool(c.valid[1, 4:].any())

    def test_second_step_continues_from_crop(self):
        c = _cache()
        rows = [0, 1]
        c.seed_row(0, 4)
        c.seed_row(1, 4)
        c.begin_step(rows, [1, 1], seg_max=1, block=2)
        c.update(_kv(2, 3, 1.0), _kv(2, 3, 1.0), 0, rows)
        c.crop(rows, [1, 1])
        kv_len = c.begin_step(rows, [1, 1], seg_max=1, block=2)
        assert kv_len == 5 + 3
        kk, _ = c.update(_kv(2, 3, 2.0), _kv(2, 3, 2.0), 0, rows)
        assert torch.all(kk[:, :, 5:8] == 2.0)
        assert torch.all(kk[:, :, :5] != 2.0)
