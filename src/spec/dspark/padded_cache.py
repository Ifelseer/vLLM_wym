"""草稿批量化用的 KV 缓存：**每行长度不同，物理上 padded 到同一长度**。

## 为什么不能直接用 `DynamicCache`

草稿批量化要把 B 条序列的块前向合到一次调用里（否则权重被读 B 遍：4.7GB × B）。
但 `DynamicCache` 的布局是 `[B, heads, seq, dim]`，要求**所有行等长**；而各序列的
`num_cached_tokens` 天生不同（prompt 长度不同、每步接受的草稿数不同）。所以必须换
一种存储：每行写到自己的偏移，读的时候按 `max(长度)` 取前缀，再用**逐行有效位掩码**
把 padding 藏掉。

## 三条不变量（写错任何一条都会静默算错，不会报错）

1. **写入偏移按行**：第 r 行的新 K/V 落在 `[lens[r], lens[r]+L)`，且所有行的 `L`
   必须相同（调用方负责把 ctx 段 pad 到 `seg_max`）；
2. **padding 不参与 attention**：有效位由 `begin_step(valid_new)` 给出，第 r 行
   在新增区段里"真"的只有前 `seg_row[r]` 个 + 末尾 `block` 个，中间是 pad；
3. **`crop` 的目标也是按行**：块内那几个候选位置的 KV 用完即弃（上游语义），
   每行保留 `lens_before + seg_row`。

## 与上游协议的对应

上游用 `DynamicCache.crop(start)` 做"只留 context"；这里等价于
`keep = [len_before[r] + seg_row[r]]`。区别只是它一行一值、我们一次给一批。
"""

from __future__ import annotations

import torch

__all__ = ["PaddedKVCache"]


class PaddedKVCache:
    """按 batch 行存储、逐行不等长的草稿 KV（padded 到同一前缀长度）。"""

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.max_len = int(max_len)
        self.device = device
        self.dtype = dtype

        self.batch = 0
        # 每层一个 [B, H, max_len, D] 缓冲；首次用到该层时才分配（省显存）
        self.k: list[torch.Tensor | None] = [None] * self.num_layers
        self.v: list[torch.Tensor | None] = [None] * self.num_layers
        #: 每行**已有**的有效长度（不含本步新增）
        self.lens = torch.zeros(0, dtype=torch.long, device=device)
        #: [B, max_len] bool：该位置是否参与 attention
        self.valid = torch.zeros(0, 0, dtype=torch.bool, device=device)
        self._new_len = 0

    # ---------------------------------------------------------------- #
    # 生命周期
    # ---------------------------------------------------------------- #

    def reset(self, lengths: list[int]) -> None:
        """开始新的一批：``lengths[r]`` 是第 r 行已有的有效长度。"""
        self.batch = len(lengths)
        assert self.batch > 0
        self.lens = torch.tensor(lengths, dtype=torch.long, device=self.device)
        assert int(self.lens.max()) <= self.max_len, "初始长度超出缓冲上限"
        self.valid = torch.zeros(
            self.batch, self.max_len, dtype=torch.bool, device=self.device
        )
        for r, n in enumerate(lengths):
            self.valid[r, :n] = True
        for buf in (self.k, self.v):
            for i, t in enumerate(buf):
                if t is not None and (t.shape[0] != self.batch or t.shape[2] != self.max_len):
                    buf[i] = None  # 批大小变了就重分配
        self._new_len = 0

    def begin_step(self, seg_rows: list[int], seg_max: int, block: int) -> int:
        """登记本步新增区段的有效位，返回本次 attention 的 KV 长度。

        新增区段布局（每行 L = seg_max + block 个位置）：

            [ 真实 ctx: seg_rows[r] ][ pad: seg_max - seg_rows[r] ][ block 个候选 ]

        返回 ``max(lens) + L``，即本步 attention 的 kv 长度上界。
        """
        assert len(seg_rows) == self.batch
        new_len = seg_max + block
        start = self.lens
        end = start + new_len
        assert int(end.max()) <= self.max_len, (
            f"KV 缓冲不够：需要 {int(end.max())} > max_len={self.max_len}"
        )
        for r in range(self.batch):
            s = int(start[r])
            valid_r = self.valid[r]
            valid_r[s : s + seg_rows[r]] = True          # 真实 ctx 段
            valid_r[s + seg_max : s + new_len] = True     # block 个候选
        self._new_len = new_len
        return int(end.max())

    # ---------------------------------------------------------------- #
    # 供 attention 使用
    # ---------------------------------------------------------------- #

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, cache_kwargs=None):
        """写入本步 K/V 并返回**整段**（含历史）的 padded K/V。

        ``k``/``v``: ``[B, H, L, D]``，``L`` 对所有行必须相同。
        返回: ``[B, H, kv_len, D]``，其中 ``kv_len = max(lens) + L``。
        """
        del cache_kwargs
        b, h, length, d = k.shape
        assert b == self.batch and length == self._new_len, (
            f"expect [B={self.batch}, H, L={self._new_len}, D], got {tuple(k.shape)}"
        )
        if self.k[layer_idx] is None:
            self.k[layer_idx] = torch.zeros(
                b, h, self.max_len, d, dtype=self.dtype, device=self.device
            )
            self.v[layer_idx] = torch.zeros_like(self.k[layer_idx])
        kk = self.k[layer_idx]
        vv = self.v[layer_idx]
        assert kk is not None and vv is not None
        # 逐行写：行内是一次连续 slice，B 很小（≤ max_num_seqs），开销可忽略
        for r in range(b):
            off = int(self.lens[r])
            kk[r, :, off : off + length] = k[r]
            vv[r, :, off : off + length] = v[r]
        kv_len = int((self.lens + length).max())
        return kk[:, :, :kv_len], vv[:, :, :kv_len]

    def mask(self, kv_len: int, q_len: int) -> torch.Tensor:
        """返回 additive attention mask ``[B, 1, q_len, kv_len]``。

        padding 位置必须填 **-inf**（填 0 等于放行，attention 会读到噪声）。
        """
        valid = self.valid[:, :kv_len]
        additive = torch.zeros(
            self.batch, 1, 1, kv_len, dtype=torch.float32, device=self.device
        )
        additive = additive.masked_fill(~valid.unsqueeze(1).unsqueeze(2), float("-inf"))
        return additive.expand(self.batch, 1, q_len, kv_len)

    # ---------------------------------------------------------------- #
    # 步末：只留 context，丢掉块内候选
    # ---------------------------------------------------------------- #

    def crop(self, seg_rows: list[int]) -> None:
        """块内候选的 KV 用完即弃（上游 ``crop(start)`` 的批量化版本）。"""
        assert len(seg_rows) == self.batch
        seg = torch.tensor(seg_rows, dtype=torch.long, device=self.device)
        keep = self.lens + seg
        positions = torch.arange(self.max_len, device=self.device).unsqueeze(0)
        self.valid &= positions < keep.unsqueeze(1)
        self.lens = keep
        self._new_len = 0
