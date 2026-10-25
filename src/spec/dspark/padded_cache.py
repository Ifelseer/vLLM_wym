"""草稿批量化用的 KV 缓存：**每行长度不同，物理上 padded 到同一长度**。

## 为什么不能直接用 `DynamicCache`

草稿批量化要把 B 条序列的块前向合到一次调用里（否则权重被读 B 遍：4.7GB × B，
实测占 b=4 步时的 61%）。但 `DynamicCache` 的布局是 `[B, heads, seq, dim]`，要求
**所有行等长**；而各序列的 `num_cached_tokens` 天生不同（prompt 长度不同、每步接受
的草稿数不同）。所以换成：每行写到自己的偏移，读时按上界取前缀，再用**逐行有效位
掩码**把 padding 藏掉。

## 为什么要 `rows` 参数

行是**长期持有**的（`seq_id → 行号`，由 DSparkDrafter 分配），但每一步参与批量的
序列集合可能不同。所以所有方法都接受 `rows`——只有这些行会推进，其余行原样保留。
写成"推进所有行"会把没参与本步的序列的 KV 也往前推一格，之后它的位置就全错。

## 四条不变量（写错都不会报错，只让接受率无声下滑）

1. **写入偏移按行**：第 r 行的新 K/V 落在 `[lens[r], lens[r]+L)`，且所有**参与行**的
   `L` 必须相同（调用方把 ctx 段 pad 到 `seg_max`）；
2. **未参与行不推进**：`lens` 只对 `rows` 更新；
3. **padding 在 mask 里是 -inf**（填 0 等于放行，attention 会读到噪声）；
4. **`crop` 按行**：块内候选的 KV 用完即弃，保留 `lens + seg`。
"""

from __future__ import annotations

import torch

__all__ = ["PaddedKVCache"]


class PaddedKVCache:
    """按行存储、逐行不等长的草稿 KV（padded 到同一前缀长度）。"""

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_rows: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.num_rows = int(num_rows)
        self.max_len = int(max_len)
        self.device = device
        self.dtype = dtype

        self.k: list[torch.Tensor | None] = [None] * self.num_layers
        self.v: list[torch.Tensor | None] = [None] * self.num_layers
        #: 每行**已有**的有效长度（不含本步新增）
        self.lens = torch.zeros(self.num_rows, dtype=torch.long, device=device)
        #: [num_rows, max_len] bool：该位置是否参与 attention
        self.valid = torch.zeros(
            self.num_rows, self.max_len, dtype=torch.bool, device=device
        )
        self._new_len = 0

    # ---------------------------------------------------------------- #
    # 行管理
    # ---------------------------------------------------------------- #

    def reset_row(self, row: int) -> None:
        """释放一行（序列结束 / 状态漂移）。"""
        self.lens[row] = 0
        self.valid[row].zero_()

    def row_len(self, row: int) -> int:
        return int(self.lens[row])

    def seed_row(self, row: int, n: int) -> None:
        """声明某行已有 ``n`` 个有效 context（``lens`` 与 ``valid`` 必须同步）。

        直接改 ``lens`` 而忘了 ``valid`` 会让 mask 把已有历史也 mask 掉——症状是
        草稿只能看到 anchor，接受率骤降但不报错。所以对外提供这个成对接口。
        """
        assert 0 <= n <= self.max_len
        self.lens[row] = n
        self.valid[row].zero_()
        self.valid[row, :n] = True

    # ---------------------------------------------------------------- #
    # 步进
    # ---------------------------------------------------------------- #

    def begin_step(
        self, rows: list[int], seg_rows: list[int], seg_max: int, block: int
    ) -> int:
        """登记本步新增区段的有效位，返回本步 attention 的 KV 长度。

        新增区段布局（每行 L = seg_max + block）：

            [ 真实 ctx: seg_rows[i] ][ pad: seg_max - seg_rows[i] ][ block 个候选 ]

        返回 ``max(参与行的 lens) + L``。
        """
        assert len(rows) == len(seg_rows)
        new_len = seg_max + block
        for r, seg in zip(rows, seg_rows):
            s = int(self.lens[r])
            end = s + new_len
            assert end <= self.max_len, (
                f"row {r} 需要 {end} > max_len={self.max_len}"
            )
            self.valid[r, s : s + seg] = True           # 真实 ctx 段
            self.valid[r, s + seg_max : end] = True      # block 个候选
        self._new_len = new_len
        return int(self.lens[rows].max()) + new_len

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, rows: list[int]):
        """写入参与行的 K/V，返回这些行的整段 padded K/V。

        ``k``/``v``: ``[len(rows), H, L, D]``，``L`` 必须等于 ``begin_step`` 的
        ``seg_max + block``。返回 ``[len(rows), H, kv_len, D]``。
        """
        n = len(rows)
        b, h, length, d = k.shape
        assert b == n and length == self._new_len, (
            f"expect [rows={n}, H, L={self._new_len}, D], got {tuple(k.shape)}"
        )
        if self.k[layer_idx] is None:
            self.k[layer_idx] = torch.zeros(
                self.num_rows, h, self.max_len, d, dtype=self.dtype, device=self.device
            )
            self.v[layer_idx] = torch.zeros_like(self.k[layer_idx])
        kk = self.k[layer_idx]
        vv = self.v[layer_idx]
        assert kk is not None and vv is not None

        idx = torch.tensor(rows, dtype=torch.long, device=self.device)
        for i, r in enumerate(rows):
            off = int(self.lens[r])
            kk[r, :, off : off + length] = k[i]
            vv[r, :, off : off + length] = v[i]
        kv_len = int(self.lens[idx].max()) + length
        return kk[idx][:, :, :kv_len], vv[idx][:, :, :kv_len]

    def mask(self, rows: list[int], kv_len: int, q_len: int) -> torch.Tensor:
        """additive attention mask ``[len(rows), 1, q_len, kv_len]``。"""
        idx = torch.tensor(rows, dtype=torch.long, device=self.device)
        valid = self.valid[idx][:, :kv_len]
        additive = torch.zeros(
            len(rows), 1, 1, kv_len, dtype=torch.float32, device=self.device
        )
        additive = additive.masked_fill(
            ~valid.unsqueeze(1).unsqueeze(2), float("-inf")
        )
        return additive.expand(len(rows), 1, q_len, kv_len)

    def crop(self, rows: list[int], seg_rows: list[int]) -> None:
        """块内候选的 KV 用完即弃（上游 ``crop(start)`` 的批量化版本）。"""
        assert len(rows) == len(seg_rows)
        positions = torch.arange(self.max_len, device=self.device)
        for r, seg in zip(rows, seg_rows):
            keep = int(self.lens[r]) + seg
            self.valid[r] &= positions < keep
            self.lens[r] = keep
        self._new_len = 0
