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
        #: 本步参与的行 / 每行的真实 ctx 段长 / 本步 attention 的 kv 长度
        self._rows: list[int] = []
        self._seg_rows: list[int] = []
        self._kv_len = 0
        #: **host 镜像**：每行已写长度。用它算写入偏移与 kv_len，避免 `int(device_lens[r])`
        #: 这种隐式同步（在 ZOS 下它会等上一拍算完，把 lookahead 掐死）。
        self._lens_host: list[int] = [0] * self.num_rows
        #: ZOS 模式：seg 由设备给出，`lens` 每拍按**常量** `seg_max` 推进（不按真实 1+j）
        self._device_segs = False
        self._seg_max = 0

    # ---------------------------------------------------------------- #
    # 行管理
    # ---------------------------------------------------------------- #

    def reset_row(self, row: int) -> None:
        """释放一行（序列结束 / 状态漂移）。"""
        self.lens[row] = 0
        self.valid[row].zero_()
        self._lens_host[row] = 0

    def get_seq_length(self) -> int:
        """供 `Cache` 协议使用（部分调用方会问；取参与行的最大长度）。"""
        if not self._rows:
            return 0
        idx = torch.tensor(self._rows, dtype=torch.long, device=self.device)
        return int(self.lens[idx].max())

    def row_len(self, row: int) -> int:
        return int(self.lens[row])

    def seed_row(self, row: int, n: int) -> None:
        """声明某行已有 ``n`` 个有效 context（``lens`` 与 ``valid`` 必须同步）。

        直接改 ``lens`` 而忘了 ``valid`` 会让 mask 把已有历史也 mask 掉——症状是
        草稿只能看到 anchor，接受率骤降但不报错。所以对外提供这个成对接口。
        """
        assert 0 <= n <= self.max_len
        self.lens[row] = n
        self._lens_host[row] = n
        self.valid[row].zero_()
        self.valid[row, :n] = True

    def seed_from_dynamic(self, row: int, past) -> int:
        """把单序列 `DynamicCache` 里已提交的 K/V 拷进本行。

        ZOS 的第一步（同步 verify）走 `propose()`，它用 `DynamicCache`；后续
        ahead 走 `propose_batch_ids()`，用本类。两条路径的缓存不是同一个对象，
        不把首拍的 prompt K/V 搬过来，后续草稿就只看得到最后一个 ctx 段——
        accept 率会掉，但不会报错。所以切换时调用一次。

        Returns:
            实际搬过来的 token 数（0 表示没有可搬的内容）。
        """
        length = int(past.get_seq_length())
        if length <= 0:
            return 0
        # transformers 4.5x 后期 / 5.x：DynamicCache.layers[i].keys/values；
        # 老布局：DynamicCache.key_cache[i]。两套都兼容（只认老布局会静默返回 0）。
        cache_layers = getattr(past, "layers", None)
        if cache_layers is not None:
            pairs = [
                (getattr(layer, "keys", None), getattr(layer, "values", None))
                for layer in cache_layers
            ]
        else:
            key_cache = getattr(past, "key_cache", None)
            value_cache = getattr(past, "value_cache", None)
            if key_cache is None or value_cache is None:
                return 0
            pairs = [
                (
                    key_cache[i] if i < len(key_cache) else None,
                    value_cache[i] if i < len(value_cache) else None,
                )
                for i in range(len(key_cache))
            ]
        for layer in range(self.num_layers):
            k, v = pairs[layer] if layer < len(pairs) else (None, None)
            if k is None or v is None or k.numel() == 0:
                continue
            if self.k[layer] is None:
                self.k[layer] = torch.zeros(
                    self.num_rows,
                    k.shape[1],
                    self.max_len,
                    k.shape[3],
                    dtype=self.dtype,
                    device=self.device,
                )
                self.v[layer] = torch.zeros_like(self.k[layer])
            n = min(length, int(k.shape[2]), self.max_len)
            self.k[layer][row, :, :n] = k[0, :, :n].to(self.dtype)
            self.v[layer][row, :, :n] = v[0, :, :n].to(self.dtype)
        self.seed_row(row, min(length, self.max_len))
        return min(length, self.max_len)

    # ---------------------------------------------------------------- #
    # 步进
    # ---------------------------------------------------------------- #

    def begin_step(
        self, rows: list[int] | None = None, seg_rows: list[int] | None = None,
        seg_max: int = 0, block: int = 0,
        device_segs: torch.Tensor | None = None,
    ) -> int:
        """登记本步新增区段的有效位，返回本步 attention 的 KV 长度。

        新增区段布局（每行 L = seg_max + block）：

            [ 真实 ctx: seg_rows[i] ][ pad: seg_max - seg_rows[i] ][ block 个候选 ]

        返回 ``max(参与行的 lens) + L``。
        """
        if rows is None:
            rows = self._rows
        rows = list(rows)
        new_len = seg_max + block
        self._device_segs = device_segs is not None
        self._seg_max = seg_max

        if device_segs is not None:
            # ZOS：seg 是**设备标量**（1+j），不能用来切 host 下标。
            # 于是"按满 seg_max + block 追加"，有效位 = arange(m) < seg_i
            # （host 切片 + device 掩码，不产生同步）；块段恒有效。
            assert len(rows) == int(device_segs.numel())
            t = torch.arange(new_len, device=self.device)
            for i, r in enumerate(rows):
                s = self._lens_host[r]
                end = s + new_len
                assert end <= self.max_len, (
                    f"row {r} 需要 {end} > max_len={self.max_len}"
                )
                val = t < device_segs[i].to(torch.long)
                val[seg_max:] = True
                self.valid[r, s:end] = val
        else:
            if seg_rows is None:
                seg_rows = self._seg_rows
            assert len(rows) == len(seg_rows)
            for r, seg in zip(rows, seg_rows):
                s = self._lens_host[r]
                end = s + new_len
                assert end <= self.max_len, (
                    f"row {r} 需要 {end} > max_len={self.max_len}"
                )
                self.valid[r, s : s + seg] = True           # 真实 ctx 段
                self.valid[r, s + seg_max : end] = True      # block 个候选

        self._rows = rows
        self._seg_rows = list(seg_rows) if seg_rows is not None else []
        self._new_len = new_len
        self._kv_len = max(self._lens_host[r] for r in rows) + new_len
        return self._kv_len

    def update(
        self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, cache_kwargs=None
    ):
        """写入参与行的 K/V，返回这些行的整段 padded K/V。

        ``k``/``v``: ``[len(rows), H, L, D]``，``L`` 必须等于 ``begin_step`` 的
        ``seg_max + block``。返回 ``[len(rows), H, kv_len, D]``。
        """
        del cache_kwargs
        rows = self._rows
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
            # 用 host 镜像而不是 `int(self.lens[r])`：后者是一次隐式 D2H 同步，
            # 在 ZOS 下会等上一拍算完，把 lookahead 掐死。
            off = self._lens_host[r]
            kk[r, :, off : off + length] = k[i]
            vv[r, :, off : off + length] = v[i]
        kv_len = self._kv_len or (max(self._lens_host[r] for r in rows) + length)
        return kk[idx][:, :, :kv_len], vv[idx][:, :, :kv_len]

    def mask(self, q_len: int, dtype: torch.dtype | None = None) -> torch.Tensor:
        """additive attention mask ``[B, 1, q_len, kv_len]``（B = 本步参与行数）。"""
        rows = self._rows
        kv_len = self._kv_len
        idx = torch.tensor(rows, dtype=torch.long, device=self.device)
        valid = self.valid[idx][:, :kv_len]
        # ⚠️ dtype 必须与 query 一致（bf16 模型 + fp32 mask 会被 SDPA 拒绝：
        # "invalid dtype for bias"）。默认用缓存自身的 dtype。
        out_dtype = self.dtype if dtype is None else dtype
        additive = torch.zeros(
            len(rows), 1, 1, kv_len, dtype=out_dtype, device=self.device
        )
        additive = additive.masked_fill(
            ~valid.unsqueeze(1).unsqueeze(2), float("-inf")
        )
        return additive.expand(len(rows), 1, q_len, kv_len)

    def crop(self) -> None:
        """块内候选的 KV 用完即弃（上游 ``crop(start)`` 的批量化版本）。"""
        rows = self._rows
        if self._device_segs:
            # ZOS：长度按**常量** seg_max 推进（真实 1+j 只体现在 valid 上）。
            # 多出来的部分是 mask 掉的，缓冲留 1.5 倍余量即可，不必压缩。
            for r in rows:
                self._lens_host[r] += self._seg_max
                self.lens[r] = self._lens_host[r]
        else:
            positions = torch.arange(self.max_len, device=self.device)
            for r, seg in zip(rows, self._seg_rows):
                keep = self._lens_host[r] + seg
                self.valid[r] &= positions < keep
                self.lens[r] = keep
                self._lens_host[r] = keep
        self._new_len = 0
