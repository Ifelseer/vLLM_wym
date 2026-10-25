"""草稿批量化的 KV 组件（纯 CPU，随机张量）。

只测三条**不变量**——它们写错时不会报错，只会让草稿看到错的历史/噪声，
表现为接受率无声下滑：

1. 每行的新 K/V 落在自己的偏移上，不串行；
2. padding 位置在 mask 里是 -inf（不是 0）；
3. crop 之后每行只保留 `已有 + seg_row`，块内候选不残留。

注意不测"数值对不对"——那是 GPU 上接进模型后的事，这里测的是账本。
"""

from __future__ import annotations

import torch

from src.spec.dspark.padded_cache import PaddedKVCache

H = 2
D = 4
LAYERS = 2
MAX_LEN = 32


def _cache() -> PaddedKVCache:
    return PaddedKVCache(
        num_layers=LAYERS,
        num_kv_heads=H,
        head_dim=D,
        max_len=MAX_LEN,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _kv(batch: int, length: int, fill: float) -> torch.Tensor:
    return torch.full((batch, H, length, D), fill)


class TestPaddedKVCache:
    def test_per_row_offsets(self):
        """三段长度不同的历史：写入后每行只能在自己的区间里看到自己的值。"""
        c = _cache()
        c.reset([3, 5, 0])
        seg_rows, seg_max, block = [2, 1, 2], 2, 3
        c.begin_step(seg_rows, seg_max, block)
        L = seg_max + block
        k = torch.stack([_kv(1, L, float(r + 1))[0] for r in range(3)])
        v = k.clone()
        kk, vv = c.update(k, v, 0)
        # 行 0：历史 0..2 是零，新增 3..7 是 1.0
        assert torch.all(kk[0, :, :3] == 0)
        assert torch.all(kk[0, :, 3:8] == 1.0)
        # 行 1：历史 0..4 是零，新增 5..9 是 2.0
        assert torch.all(kk[1, :, :5] == 0)
        assert torch.all(kk[1, :, 5:10] == 2.0)
        # 行 2：从 0 开始
        assert torch.all(kk[2, :, :5] == 3.0)

    def test_mask_marks_padding_neg_inf(self):
        c = _cache()
        c.reset([0, 0])
        # seg_max=3：行 0 有 2 个真实 ctx（1 个 pad），行 1 只有 1 个（2 个 pad）
        c.begin_step([2, 1], seg_max=3, block=2)
        c.update(_kv(2, 5, 1.0), _kv(2, 5, 1.0), 0)
        m = c.mask(kv_len=5, q_len=2)
        assert m.shape == (2, 1, 2, 5)
        # 行 0：ctx 0..1 有效，位置 2 是 pad，3..4 是 block（有效）
        assert not torch.isneginf(m[0, 0, 0, 1])
        assert torch.isneginf(m[0, 0, 0, 2]), "seg_max 尾部的 pad 必须被 mask"
        assert not torch.isneginf(m[0, 0, 0, 4])
        # 行 1：ctx 0 有效，位置 1..2 是 pad，3..4 是 block
        assert torch.isneginf(m[1, 0, 0, 1])
        assert torch.isneginf(m[1, 0, 0, 2])
        assert not torch.isneginf(m[1, 0, 0, 3])

    def test_crop_keeps_only_context(self):
        c = _cache()
        c.reset([4, 2])
        c.begin_step([2, 2], seg_max=2, block=3)
        c.update(_kv(2, 5, 1.0), _kv(2, 5, 1.0), 0)
        c.crop([2, 2])
        assert c.lens.tolist() == [6, 4], "应保留 已有长度 + seg"
        # 保留段内全有效，之后全无效
        assert bool(c.valid[0, :6].all()) and not bool(c.valid[0, 6:].any())
        assert bool(c.valid[1, :4].all()) and not bool(c.valid[1, 4:].any())

    def test_second_step_continues_from_crop(self):
        c = _cache()
        c.reset([4, 4])
        c.begin_step([1, 1], seg_max=1, block=2)
        c.update(_kv(2, 3, 1.0), _kv(2, 3, 1.0), 0)
        c.crop([1, 1])
        # 第二步：从 5 开始，再追加 1 个 ctx + 2 个候选
        kv_len = c.begin_step([1, 1], seg_max=1, block=2)
        assert kv_len == 8
        kk, _ = c.update(_kv(2, 3, 2.0), _kv(2, 3, 2.0), 0)
        assert torch.all(kk[:, :, 5:8] == 2.0)
        assert torch.all(kk[:, :, :5] != 2.0)
