"""ZOS 的设备侧输入推进：把下一轮 decode 的元数据在当前 GPU 缓冲里就地算出。

## 为什么需要它

普通 decode 每轮的输入由 CPU 填好再 H2D：位置 = `num_cached_tokens`、slot =
`block_table[pos // B] * B + pos % B`、token = 上一步采样结果。这条路有个硬约束——
**CPU 必须先把上一个 token 读回来**，才能算下一轮的位置和 slot。于是每轮都变成：

```
GPU 前向 → 同步回读 token → CPU 更新 → H2D 新输入 → GPU 前向
```

ZOS 要把中间那段摘掉：**GPU 用自己上一轮的输出当下一轮的输入**，CPU 在后面追。
前提就是这里这个 kernel —— 位置、长度、slot 全部在设备侧由"上一轮的值 + 1"推出来，
不经过主机。

## 为什么是「每行一个推进量」而不是固定的 +1

非投机时每轮推进 1 个位置；投机时一轮写入 `1 + j` 个位置（j = 被接受的草稿数）。
两者只差一个数，所以 kernel 从第一天就收 `advances[row]`：

```
advances 全 1        → 普通 decode
advances = 1 + j     → 投机（j 由设备侧接受裁决给出）
```

这样不必为投机再写第二个 kernel。

## 一个容易忽略的点：`tokens` 必须是"本行最后一个被接受的 token"

投机时一次产出 `1+j` 个 token，下一轮要喂的是**最后一个**（新的 anchor）。把它 gather
成 `[bs]` 是调用方的事（设备侧索引，仍不需要同步），kernel 只负责"取一个 token 写进
input_ids"。

## 与 `Batch.build_decode` 的对应关系

`build_decode` 里逐行的三行算术：

```python
pos  = s.num_cached_tokens
ids  = s.token_ids[-1]
slot = bt[pos // B] * B + pos % B
csl  = pos + 1
```

这里的差别只有一处：`pos` 不是从 CPU 状态读，而是**从设备缓冲里读上一轮的值再加
`advances`**。其余完全一致——这是"设备侧推进"与"CPU 重建"必须逐位等价的依据。
"""

from __future__ import annotations

import torch

__all__ = ["advance_decode", "advance_reference"]

try:  # 允许在没有 Triton 的环境（CPU 单测）import 本模块
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _advance_kernel(
        tokens,  # [bs] int64：本行下一个输入 token
        advances,  # [bs] int32：本行推进几个位置（1 或 1+j）
        ids,  # [max_tokens] int64
        positions,  # [max_tokens] int64
        slots,  # [max_tokens] int32
        lengths,  # [max_bs] int32
        tables,  # [max_bs, WIDTH] int32
        BS: tl.constexpr,
        WIDTH: tl.constexpr,
        PAGE: tl.constexpr,
        TILE: tl.constexpr,
    ):
        row = tl.arange(0, TILE)
        valid = row < BS

        step = tl.load(advances + row, mask=valid, other=1).to(tl.int64)
        pos = tl.load(positions + row, mask=valid, other=0) + step
        page = tl.load(
            tables + row * WIDTH + pos // PAGE, mask=valid, other=0
        ).to(tl.int64)
        token = tl.load(tokens + row, mask=valid, other=0)

        tl.store(ids + row, token, mask=valid)
        tl.store(positions + row, pos, mask=valid)
        tl.store(slots + row, (page * PAGE + pos % PAGE).to(tl.int32), mask=valid)
        tl.store(lengths + row, (pos + 1).to(tl.int32), mask=valid)


def advance_decode(
    data,
    tokens: torch.Tensor,
    advances: torch.Tensor,
    bs: int,
    block_size: int,
) -> None:
    """就地推进 ``data``（:class:`~src.compute.batch.Batch`）的前 ``bs`` 行。

    Args:
        data: ``Batch``，用到 ``input_ids / positions / slot_mapping /
            cache_seqlens / block_tables`` 五个缓冲（都是常驻的）。
        tokens: ``[bs]`` int64，每行的下一个输入 token（已 gather，仍在 device 上）。
        advances: ``[bs]`` int32，每行推进几个位置（普通 decode 全 1）。
        bs: 本步行数（可能小于缓冲容量）。
        block_size: KV 页大小。
    """
    assert _HAS_TRITON, "advance_decode requires triton"
    _advance_kernel[(1,)](
        tokens,
        advances,
        data.input_ids,
        data.positions,
        data.slot_mapping,
        data.cache_seqlens,
        data.block_tables,
        bs,
        data.block_tables.shape[1],
        block_size,
        triton.next_power_of_2(bs),
        num_warps=4,
    )


def advance_reference(
    *,
    tokens: torch.Tensor,
    advances: torch.Tensor,
    ids: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    lengths: torch.Tensor,
    tables: torch.Tensor,
    bs: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """纯 torch 版，逐步对应 ``_advance_kernel``（CPU 可测）。

    返回**新张量**而不是原地改，方便测试同时持有 before/after。
    """
    ids = ids.clone()
    positions = positions.clone()
    slots = slots.clone()
    lengths = lengths.clone()

    rows = torch.arange(bs, device=positions.device)
    pos = positions[rows] + advances[:bs].to(torch.int64)
    page = tables[rows, pos // block_size].to(torch.int64)

    ids[rows] = tokens[:bs]
    positions[rows] = pos
    slots[rows] = (page * block_size + pos % block_size).to(torch.int32)
    lengths[rows] = (pos + 1).to(torch.int32)
    return ids, positions, slots, lengths, tables
