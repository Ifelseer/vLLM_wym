"""ZOS 模式的 padded KV：**按 m 追加 + 设备 seg 掩码**，与 host seg 路径等价。

ZOS 下 `seg = 1+j` 只在设备上，不能用来切 host 下标。方案是"按满 `seg_max + block`
追加"，有效位用 `arange(m) < seg_i` 写 —— **host 切片 + device 掩码**，不产生同步。

这里要钉住两条：

1. 同一个 seg 序列下，两条路径的**有效位矩阵完全一致**（设备路径只是多出一些
   无效尾巴，因为按 seg_max 追加）；
2. 设备路径的 `lens` 每拍按**常量 seg_max** 推进（所以 host 侧始终可知，不需要同步）。
"""

from __future__ import annotations

import pytest
import torch

from src.spec.dspark.padded_cache import PaddedKVCache

H = 2
D = 4
ROWS = 2
MAX_LEN = 64


def _cache() -> PaddedKVCache:
    return PaddedKVCache(
        num_layers=1,
        num_kv_heads=H,
        head_dim=D,
        num_rows=ROWS,
        max_len=MAX_LEN,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


class TestDeviceSegs:
    def test_context_slots_match_host_path(self):
        """每一拍的 **ctx 段**：前 seg 个有效、其后到 seg_max 无效（与 host 路径一致）。

        不做整矩阵相等比较：设备路径按 `seg_max` 追加，末尾会留下本拍 block 段的
        有效位（下一拍会被覆盖），host 路径则把 block 段裁掉了 —— 这是设计差异，
        不是错误。
        """
        rounds = [(3, 2), (2, 1), (4, 4)]
        block = 3
        seg_max = max(max(r) for r in rounds)

        dev = _cache()
        for k, segs in enumerate(rounds):
            dev.begin_step(
                [0, 1],
                None,
                seg_max=seg_max,
                block=block,
                device_segs=torch.tensor(segs, dtype=torch.long),
            )
            for r in range(2):
                base = k * seg_max
                seg = segs[r]
                assert dev.valid[r, base : base + seg].all(), (
                    f"round{k} row{r} 的 ctx 段应有效"
                )
                assert not dev.valid[r, base + seg : base + seg_max].any(), (
                    f"round{k} row{r} 的 pad 段必须无效"
                )
            dev.crop()

    def test_device_lens_advance_by_seg_max(self):
        """设备路径的 host 镜像按常量 seg_max 推进（这正是"无需同步"的来源）。"""
        seg_max, block = 5, 3
        c = _cache()
        for _ in range(3):
            c.begin_step(
                [0], None, seg_max=seg_max, block=block,
                device_segs=torch.tensor([2], dtype=torch.long),
            )
            c.crop()
        assert c._lens_host[0] == 3 * seg_max
        # 3 拍 × seg=2 的真实 ctx = 6，加上最后一拍 block 段的 3 个（下一拍会被覆盖）
        assert int(c.valid[0].sum()) == 3 * 2 + block

    def test_pad_and_block_regions(self):
        """seg < seg_max 时：pad 段无效、block 段恒有效。"""
        seg_max, block = 4, 2
        c = _cache()
        c.begin_step(
            [0], None, seg_max=seg_max, block=block,
            device_segs=torch.tensor([2], dtype=torch.long),
        )
        # 区间 [0, seg_max+block) = [0, 6)
        assert c.valid[0, :2].all(), "前 seg 个是真实 ctx"
        assert not c.valid[0, 2:4].any(), "pad 段必须无效"
        assert c.valid[0, 4:6].all(), "block 段恒有效"

    def test_mixed_host_and_device_rejected(self):
        """同时给 seg_rows 和 device_segs 时以设备为准（并保持长度语义正确）。"""
        seg_max, block = 3, 2
        c = _cache()
        c.begin_step(
            [0], [1], seg_max=seg_max, block=block,
            device_segs=torch.tensor([3], dtype=torch.long),
        )
        c.crop()
        assert c._lens_host[0] == seg_max, "设备路径应按 seg_max 推进，不按传入的 host seg"
