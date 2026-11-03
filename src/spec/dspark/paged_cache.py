"""页式草稿 KV：页池 + 页表，替代「每行预留满 max_len」的稠密布局。

## 为什么需要它

稠密布局 `[num_rows, kv_heads, max_len, head_dim]` 的容量是 `O(行数 × 上限)`，
与实际用量无关：

| 场景 | 稠密预留 | 真实需要 |
|---|---|---|
| b=1，n=128（8 行 × 4096） | 671 MB | ~2.5 MB |
| b=4，n=128（16 行 × 4096） | 1.34 GB | ~17 MB |
| 1M 上下文 | 不可行（20 GB/序列起） | 20 GB（仍需多卡） |

页式把容量变成 `O(实际 token)`，并按需分配。

## 两条读路径（收益来自存储，不来自读路径）

| 档 | 做法 | 代价 | 何时用 |
|---|---|---|---|
| **一（默认）** | `read_dense()` 按页表把有效页 gather 成 `[B, kv_len, ...]`，再走原来的 SDPA | 每步每层一次拷贝（128 token ≈ 12 MB） | 现在；**不改 vendored attention** |
| 二（未来） | `flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens, block_table)` | 必须改 `modeling.py`；**页大小须为 256 的倍数**（本仓 `config.py` 硬校验，FA2 paged 约束） | 长上下文 |

所以本类默认 `page_size=16`（只服务路径一）；开 FA 快路径时必须传 256。

> **关于「去掉掩码」的边界**：只有路径二能真正去掉。路径一为了让 SDPA 拿到矩形
> 张量，仍需 `mask()`；但 pad 占用的页会在 `crop` 时退还，所以浪费是**瞬时且不累积**
> 的（稠密布局是永久预留）。这也是"收益来自存储、不来自读路径"的含义。

## 页号编码：int64，高位留给扩展（现在零成本）

```
bit  0..35  页内编号（单机）
bit 36..43  tier   0=GPU 1=CPU 2=SSD
bit 44..51  device 0..255（TP / CP）
负数        未分配
```

页表类型因此不必随扩展改动——将来 offload / 跨卡只是「填不同的值」。

## 三条不变量（写错都不报错，只让接受率无声下滑）

1. **页表覆盖的页数必须够放下逻辑长度**（`ceil(len/page) <= len(table)`），否则读到空页；
2. **crop 必须退还尾部整页**：只做「逻辑缩短」会让页数按 `m` 增长，1.33 倍浪费照旧，等于白改；
3. **未参与本步的行绝不能被推进或改写**（行池长期持有，每步参与者不同）。

三条各配一条单测，见 `tests/unit/test_dspark_paged_cache.py`。
"""

from __future__ import annotations

from collections import deque

import torch

__all__ = ["PagedDraftKV", "encode_page", "decode_page", "TIER_GPU"]

# 页号编码位宽（见模块 docstring）
_PAGE_BITS = 36
_TIER_SHIFT = 36
_DEVICE_SHIFT = 44

TIER_GPU = 0
TIER_CPU = 1
TIER_SSD = 2


def encode_page(page: int, tier: int = TIER_GPU, device: int = 0) -> int:
    """把 (页号, 层级, 设备) 编成 int64 页号。"""
    assert 0 <= page < (1 << _PAGE_BITS), f"page {page} out of range"
    assert 0 <= tier < 8, f"tier {tier} out of range"
    assert 0 <= device < 256, f"device {device} out of range"
    return page | (tier << _TIER_SHIFT) | (device << _DEVICE_SHIFT)


def decode_page(value: int) -> tuple[int, int, int]:
    """拆回 (页号, 层级, 设备)。"""
    if value < 0:
        return -1, -1, -1
    return (
        value & ((1 << _PAGE_BITS) - 1),
        (value >> _TIER_SHIFT) & 0x7,
        (value >> _DEVICE_SHIFT) & 0xFF,
    )


class PagedDraftKV:
    """页式草稿 KV。接口与 transformers ``Cache`` 对齐，供 vendored attention 直接使用。

    Args:
        num_layers / num_kv_heads / head_dim: 草稿模型结构参数。
        num_rows: 行池大小（与 drafter 的 `max_rows` 一致，行号长期持有）。
        num_pages: 页池容量。**这是唯一的容量上限**；不足时 `allocate` 返回 False，
            调用方退化为「本步不起草」（只损收益，不影响正确性）。
        page_size: 页大小。路径一可任意（默认 16）；路径二必须是 256 的倍数。
        device / dtype: 池的张量属性。
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_rows: int,
        num_pages: int,
        page_size: int = 16,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        use_fa: bool = False,
    ) -> None:
        if num_pages <= 0:
            raise ValueError(f"num_pages must be > 0, got {num_pages}")
        if page_size <= 0:
            raise ValueError(f"page_size must be > 0, got {page_size}")
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.num_rows = int(num_rows)
        self.num_pages = int(num_pages)
        self.page_size = int(page_size)
        self.device = torch.device(device)
        self.dtype = dtype
        #: 读路径二开关（attention 侧据此改走 flash_attn_with_kvcache）。
        #: ⚠️ 本仓库当前只在单卡 24G 上验证过路径一；路径二需要 GPU 实测，
        #: 且页大小必须 256 的倍数（见 `fa_args`）。
        self.is_paged_fa = bool(use_fa)

        # 页池：k/v 各 num_layers 张 [num_pages, page_size, heads, dim]，按层懒分配
        # （未用到的层不占显存，与稠密实现一致）。
        self.k: list[torch.Tensor | None] = [None] * self.num_layers
        self.v: list[torch.Tensor | None] = [None] * self.num_layers

        # 页表：每行逻辑页 → 物理页号（int64，高 8 位留 tier/device）
        # 页表宽度按**最坏情况**取：一行可能占满整个页池。人均份额会让长序列
        # 在 `table[r, have] = ...` 上 IndexError（而不是优雅降级）。
        # 代价是 num_rows × num_pages × 8B（8×512×8 = 32KB），可忽略。
        max_pages_per_row = max(1, num_pages)
        self.table = torch.full(
            (self.num_rows, max_pages_per_row), -1, dtype=torch.int64, device=self.device
        )
        #: 每行已提交的逻辑长度（只增，crop 后 = 已有 + seg）
        self.lens = torch.zeros(self.num_rows, dtype=torch.long, device=self.device)
        #: host 镜像：算偏移与 kv_len 用它，避免 `int(self.lens[r])` 的隐式同步
        self._lens_host: list[int] = [0] * self.num_rows

        self._free: deque[int] = deque(range(self.num_pages))
        self._n_free = self.num_pages

        # 本步状态
        self._rows: list[int] = []
        self._segs: list[int] = []
        self._seg_max = 0
        self._block = 0
        self._kv_len = 0

    # ---------------------------------------------------------------- #
    # 容量
    # ---------------------------------------------------------------- #

    @property
    def num_free(self) -> int:
        return self._n_free

    def _pages_for(self, n_tokens: int) -> int:
        return (int(n_tokens) + self.page_size - 1) // self.page_size

    def _alloc_page(self) -> int:
        if not self._free:
            return -1
        self._n_free -= 1
        return self._free.popleft()

    def _release_page(self, page: int) -> None:
        if page < 0:
            return
        idx, tier, dev = decode_page(page)
        if tier != TIER_GPU:
            # 换出的页（未来 offload）不能直接进 GPU 空闲链
            return
        self._free.append(idx)
        self._n_free += 1

    # ---------------------------------------------------------------- #
    # 行生命周期
    # ---------------------------------------------------------------- #

    def reset_row(self, row: int) -> None:
        """释放该行占用的所有页并清空长度。"""
        self._release_row_pages(row)
        self.table[row].fill_(-1)
        self.lens[row] = 0
        self._lens_host[row] = 0

    def _release_row_pages(self, row: int, keep: int = 0) -> None:
        """把该行第 ``keep`` 页之后的页还给页池（crop 的物理部分）。"""
        n = int((self.table[row] >= 0).sum())
        for i in range(keep, n):
            self._release_page(int(self.table[row, i]))
            self.table[row, i] = -1

    def seed_row(self, row: int, n_tokens: int) -> bool:
        """声明某行已有 ``n_tokens`` 个有效位置（按需分配页，不足返回 False）。"""
        need = self._pages_for(n_tokens)
        have = int((self.table[row] >= 0).sum())
        for _ in range(max(0, need - have)):
            page = self._alloc_page()
            if page < 0:
                return False
            self.table[row, have] = encode_page(page)
            have += 1
        if have > need:
            self._release_row_pages(row, keep=need)
        self.lens[row] = int(n_tokens)
        self._lens_host[row] = int(n_tokens)
        return True

    def row_len(self, row: int) -> int:
        return self._lens_host[row]

    @property
    def active_rows(self) -> list[int]:
        """本步**实际参与**的行（页池不足时会被剔除，调用方应据此退化为不起草）。"""
        return list(self._rows)

    def get_seq_length(self) -> int:
        """供 ``Cache`` 协议使用（本步参与行的最大长度）。"""
        if not self._rows:
            return 0
        return max(self._lens_host[r] for r in self._rows)

    # ---------------------------------------------------------------- #
    # 步进
    # ---------------------------------------------------------------- #

    def begin_step(
        self,
        rows: list[int] | None = None,
        seg_rows: list[int] | None = None,
        seg_max: int = 0,
        block: int = 0,
        device_segs: torch.Tensor | None = None,
    ) -> int:
        """登记本步参与的行与各自的真实 ctx 长度，并预留 ``lens + seg_max + block`` 的页。

        ``device_segs``（ZOS 投机用）：页式下 seg 只影响掩码与退页，长度在设备上即可，
        但**读路径一**仍需要 host 侧的 ``seg_max``。真正零同步要等读路径二（FA +
        ``cache_seqlens``），所以这里显式拒绝而不是静默忽略。

        页不足时**不抛错**：把该行从本步移除（调用方视为「不起草」），保证长跑不会
        因为某个序列超长而整批失败。

        Returns:
            本步 attention 需要的 ``kv_len``（= max(lens) + seg_max + block）。
        """
        if device_segs is not None:
            raise NotImplementedError(
                "页式缓存暂不支持设备侧 seg（ZOS+投机需读路径二：FA + cache_seqlens）；"
                "请关闭 zos_spec 或使用稠密路径"
            )
        if rows is None:
            rows = self._rows
        if seg_rows is None:
            seg_rows = self._segs
        # 不变量：**每行已分配的页必须覆盖批内共享的 `_kv_len`**。
        # 因为 gather 是矩形读（`max(lens) + seg_max + block`），短行若只按自己的
        # `lens + seg` 预留，`_slots` 就会越界断言（多行不等长时必现）。
        kv_len = (
            (max(self._lens_host[r] for r in rows) + int(seg_max) + int(block))
            if rows
            else 0
        )
        need = self._pages_for(kv_len)
        kept_rows: list[int] = []
        kept_segs: list[int] = []
        for r, seg in zip(rows, seg_rows):
            before = int((self.table[r] >= 0).sum())
            have = before
            ok = True
            for _ in range(max(0, need - have)):
                page = self._alloc_page()
                if page < 0:
                    ok = False
                    break
                self.table[r, have] = encode_page(page)
                have += 1
            if not ok:
                # 回滚本次已给该行加的页：否则它们既不在 `lens` 里、也不会被
                # `crop` 退还 → 长跑逐渐泄漏。
                self._release_row_pages(r, keep=before)
                continue
            kept_rows.append(r)
            kept_segs.append(int(seg))
        self._rows = kept_rows
        self._segs = kept_segs
        self._seg_max = int(seg_max)
        self._block = int(block)
        # 有意不因剔除行而下调：多读的 pad 页会被 mask 遮掉、crop 时退还，
        # 而缩小上界反而会让已分配页与新上界不自洽。
        self._kv_len = kv_len if kept_rows else 0
        return self._kv_len

    def _pool(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.k[layer_idx]
        if k is None:
            shape = (self.num_pages, self.page_size, self.num_kv_heads, self.head_dim)
            k = torch.zeros(shape, dtype=self.dtype, device=self.device)
            v = torch.zeros(shape, dtype=self.dtype, device=self.device)
            self.k[layer_idx] = k
            self.v[layer_idx] = v
        v = self.v[layer_idx]
        assert v is not None
        return k, v

    def _slots(self, row: int, positions: torch.Tensor) -> torch.Tensor:
        """把逻辑位置映射成页池里的线性槽号。"""
        page_idx = positions // self.page_size
        in_page = positions % self.page_size
        pages = self.table[row, page_idx]
        assert bool((pages >= 0).all()), "页表不足：逻辑位置超出已分配页"
        return pages * self.page_size + in_page

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, cache_kwargs=None):
        """按页表写入本步 K/V，并返回**稠密读**结果（路径一）。

        Args:
            k, v: ``[B, heads, L, dim]``，B = 本步参与行数，
                L = ``seg_max + block``（ctx 段 pad 到 seg_max，块内候选在后）。
            layer_idx: 层号。
            cache_kwargs: transformers 的 cache 附加参数（本实现不用）。

        写入只覆盖**真实位置**：行 r 的 ctx 只写前 ``seg_r`` 个（pad 段丢弃），
        块内候选写 ``[start_r, start_r+block)``。因此不需要 padding 掩码。
        """
        del cache_kwargs
        pool_k, pool_v = self._pool(layer_idx)
        flat_k = pool_k.view(-1, self.num_kv_heads, self.head_dim)
        flat_v = pool_v.view(-1, self.num_kv_heads, self.head_dim)

        n = len(self._rows)
        assert k.shape[0] == n, f"k batch {k.shape[0]} != 参与行数 {n}"
        length = k.shape[2]
        assert length >= self._seg_max + self._block, (
            f"k length {length} < seg_max+block {self._seg_max + self._block}"
        )

        for i, r in enumerate(self._rows):
            seg = self._segs[i]
            start = self._lens_host[r] + seg
            # ctx 段：源 [0, seg)，目标 [lens, lens+seg)
            if seg > 0:
                src = torch.arange(seg, device=self.device)
                dst = torch.arange(self._lens_host[r], start, device=self.device)
                slots = self._slots(r, dst)
                flat_k[slots] = k[i, :, src].transpose(0, 1)
                flat_v[slots] = v[i, :, src].transpose(0, 1)
            # 块内候选：源 [seg_max, seg_max+block)，目标 [start, start+block)
            blk_src = torch.arange(
                self._seg_max, self._seg_max + self._block, device=self.device
            )
            blk_dst = torch.arange(start, start + self._block, device=self.device)
            slots = self._slots(r, blk_dst)
            flat_k[slots] = k[i, :, blk_src].transpose(0, 1)
            flat_v[slots] = v[i, :, blk_src].transpose(0, 1)

        return self.read_dense_layer(layer_idx)

    # ---------------------------------------------------------------- #
    # 读路径二：交给 flash-attn 的 paged kernel（默认关）
    # ---------------------------------------------------------------- #

    def update_paged(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int) -> None:
        """只写页池、不 gather（读路径二用）。写入语义与 `update` 完全一致。"""
        pool_k, pool_v = self._pool(layer_idx)
        flat_k = pool_k.view(-1, self.num_kv_heads, self.head_dim)
        flat_v = pool_v.view(-1, self.num_kv_heads, self.head_dim)
        n = len(self._rows)
        assert k.shape[0] == n, f"k batch {k.shape[0]} != 参与行数 {n}"
        for i, r in enumerate(self._rows):
            seg = self._segs[i]
            start = self._lens_host[r] + seg
            if seg > 0:
                src = torch.arange(seg, device=self.device)
                dst = torch.arange(self._lens_host[r], start, device=self.device)
                slots = self._slots(r, dst)
                flat_k[slots] = k[i, :, src].transpose(0, 1)
                flat_v[slots] = v[i, :, src].transpose(0, 1)
            blk_src = torch.arange(
                self._seg_max, self._seg_max + self._block, device=self.device
            )
            blk_dst = torch.arange(start, start + self._block, device=self.device)
            slots = self._slots(r, blk_dst)
            flat_k[slots] = k[i, :, blk_src].transpose(0, 1)
            flat_v[slots] = v[i, :, blk_src].transpose(0, 1)

    def fa_args(self, layer_idx: int) -> tuple:
        """返回 ``(k_cache, v_cache, cache_seqlens, block_table)`` 供
        ``flash_attn_with_kvcache(..., block_table=...)`` 使用。

        约束（已核实本仓 `config.py` 对目标模型有同样校验）：**页大小必须是 256 的
        倍数**，否则 FA2 的 paged kernel 不匹配。这里 fail loud 而不是静默降级。

        注意：本方法依赖 `paged_cache` 的页池布局恰好等于 FA 期望的
        ``[num_blocks, page_block_size, nheads_k, headdim]``。
        """
        if self.page_size % 256 != 0:
            raise ValueError(
                f"flash-attn paged 要求页大小为 256 的倍数，当前 {self.page_size}；"
                "请设 spec_draft_page_size=256 或关闭 spec_draft_paged_fa"
            )
        pool_k, pool_v = self._pool(layer_idx)
        seqlens = torch.tensor(
            [self._lens_host[r] + self._segs[i] + self._block for i, r in enumerate(self._rows)],
            dtype=torch.int32,
            device=self.device,
        )
        return pool_k, pool_v, seqlens, self.block_table().to(torch.int32)

    def read_dense_layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """单层 gather：``[B, heads, kv_len, dim]``。"""
        pool_k, pool_v = self._pool(layer_idx)
        flat_k = pool_k.view(-1, self.num_kv_heads, self.head_dim)
        flat_v = pool_v.view(-1, self.num_kv_heads, self.head_dim)
        positions = torch.arange(self._kv_len, device=self.device)
        ks, vs = [], []
        for r in self._rows:
            slots = self._slots(r, positions)
            ks.append(flat_k[slots].transpose(0, 1))
            vs.append(flat_v[slots].transpose(0, 1))
        return torch.stack(ks), torch.stack(vs)

    def mask(self, q_len: int, dtype: torch.dtype | None = None) -> torch.Tensor:
        """additive 掩码 ``[B, 1, q_len, kv_len]``：行 r 的有效范围是 ``[0, lens+seg+block)``。

        为什么路径一必须有它：矩形 gather 会把短行的 pad 区域也读出来（那里是别的
        轮次留下的旧 K/V），不遮掉就会污染 attention。路径二（FA + ``cache_seqlens``）
        不需要它 —— 那是"去掉掩码"这个收益只在路径二兑现的原因。

        ⚠️ dtype 必须与 query 一致（bf16 模型 + fp32 mask 会被 SDPA 拒绝）。
        """
        out_dtype = self.dtype if dtype is None else dtype
        additive = torch.zeros(
            len(self._rows), 1, 1, self._kv_len, dtype=out_dtype, device=self.device
        )
        positions = torch.arange(self._kv_len, device=self.device)
        for i, r in enumerate(self._rows):
            valid_to = self._lens_host[r] + self._segs[i] + self._block
            additive[i, 0, 0, positions >= valid_to] = float("-inf")
        return additive.expand(len(self._rows), 1, q_len, self._kv_len)

    def crop(self) -> None:
        """丢弃块内候选，把逻辑长度推进到 ``已有 + seg``，并**退还尾部整页**。"""
        for i, r in enumerate(self._rows):
            seg = self._segs[i]
            new_len = self._lens_host[r] + seg
            keep = self._pages_for(new_len)
            self._release_row_pages(r, keep=keep)
            self.lens[r] = new_len
            self._lens_host[r] = new_len
        self._seg_max = 0
        self._block = 0

    # ---------------------------------------------------------------- #
    # 兼容旧接口（drafter 接线用）
    # ---------------------------------------------------------------- #

    def block_table(self, *, subset: bool = True) -> torch.Tensor:
        """页表视图（行内按逻辑页序）。

        ``subset=True``（默认）只返回**本步参与的行**：FA 要求 ``block_table`` 的
        batch 维与 q 对齐，把整张 ``[num_rows, ...]`` 交过去会直接报
        ``block_table must have shape (batch_size, max_num_blocks_per_seq)``。
        列宽只在本步参与行上取 max —— 避免为一个已被剔除的长行开出过宽页表。
        """
        rows = self._rows if (subset and self._rows) else list(range(self.num_rows))
        max_used = max((int((self.table[r] >= 0).sum()) for r in rows), default=0)
        return self.table[rows, : max(1, max_used)]

    def seed_from_dynamic(self, row: int, past) -> int:
        """把单序列 ``DynamicCache`` 的 K/V 搬进本行（ZOS 切换用）。"""
        length = int(past.get_seq_length())
        if length <= 0:
            return 0
        if not self.seed_row(row, min(length, self.num_pages * self.page_size)):
            return 0
        layers = getattr(past, "layers", None)
        if layers is not None:
            pairs = [
                (getattr(layer, "keys", None), getattr(layer, "values", None))
                for layer in layers
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
        n = min(length, int(pairs[0][0].shape[2])) if pairs and pairs[0][0] is not None else 0
        if n <= 0:
            return 0
        positions = torch.arange(n, device=self.device)
        slots = self._slots(row, positions)
        for layer in range(min(self.num_layers, len(pairs))):
            k, v = pairs[layer]
            if k is None or v is None or k.numel() == 0:
                continue
            pool_k, pool_v = self._pool(layer)
            pool_k.view(-1, self.num_kv_heads, self.head_dim)[slots] = k[0, :, :n].transpose(0, 1).to(self.dtype)
            pool_v.view(-1, self.num_kv_heads, self.head_dim)[slots] = v[0, :, :n].transpose(0, 1).to(self.dtype)
        self.seed_row(row, n)
        return n
