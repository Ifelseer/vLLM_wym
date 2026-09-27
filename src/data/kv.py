from __future__ import annotations

from .block_manager import BlockManager
from .prefix import PrefixBackend, RadixPrefixIndex, make_prefix_index
from .sequence import Sequence

# ==============================================================================
# driver 侧 KV 账本：给序列发逻辑页、查前缀命中、用完回收。
#
# Scheduler 调 can_allocate / allocate / may_append / deallocate。
# GPU 上真正的 cache 张量在 compute/kv_pool.py，本类不 import torch。
# ==============================================================================


class KVManager:
    def __init__(
        self,
        *,
        num_blocks: int,
        block_size: int,
        prefix_backend: PrefixBackend = "hash",
    ) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        self.block_size = block_size
        self.prefix_backend: PrefixBackend = prefix_backend
        self.num_blocks_total = num_blocks
        self.num_blocks_schedulable = num_blocks
        self.blocks = BlockManager(self.num_blocks_schedulable, block_size)
        self.index = make_prefix_index(prefix_backend)
        self.blocks.on_evict = lambda bid: self.index.invalidate(bid, self.blocks)
        if isinstance(self.index, RadixPrefixIndex):
            self.blocks.reclaim_fn = lambda n: self.index.reclaim_pages(n, self.blocks)
        self.last_num_cached_tokens = 0

    @property
    def num_free(self) -> int:
        return self.blocks.num_free

    def _fit_prefix(self, seq: Sequence) -> int:
        """前缀复用预检查：计算当前序列最多可复用多少连续前缀 KV 块。"""
        need = seq.num_blocks
        if need == 0:
            return 0
        hit = self.index.match(seq, self.blocks)
        max_cached = min(len(hit), need)
        reclaimable = sum(
            1 for b in hit[:max_cached] if b in self.blocks.cached_ids
        )
        other_cached = len(self.blocks.cached_ids) - reclaimable
        new_needed = need - max_cached
        if self.blocks.num_free + other_cached >= new_needed:
            return max_cached
        return -1

    def can_allocate(self, seq: Sequence) -> bool:
        return self._fit_prefix(seq) >= 0

    def allocate(self, seq: Sequence) -> None:
        seq.block_table.clear()
        seq.num_cached_tokens = 0
        need = seq.num_blocks
        if need == 0:
            self.last_num_cached_tokens = 0
            return

        max_cached = self._fit_prefix(seq)
        if max_cached < 0:
            raise RuntimeError("not enough KV blocks for sequence")
        hit = self.index.match(seq, self.blocks)

        for bid in hit[:max_cached]:
            self.blocks.acquire_hit(bid)
            seq.block_table.append(bid)

        for _ in range(max_cached, need):
            seq.block_table.append(self.blocks.allocate_fresh())

        seq.num_cached_tokens = max_cached * self.block_size
        self.last_num_cached_tokens = seq.num_cached_tokens

        if isinstance(self.index, RadixPrefixIndex) and max_cached > 0:
            self.index.pin_match(seq, max_cached)

    def can_append(self, seq: Sequence) -> bool:
        return self.blocks.can_append(seq)

    def may_append(self, seq: Sequence) -> None:
        self.blocks.may_append(seq)

    def sync_prefix(self, seq: Sequence) -> None:
        """本步写满的新块登记进前缀索引，供后来的请求命中。"""
        if self.prefix_backend == "none" or seq.num_scheduled_tokens <= 0:
            return
        prev = seq.num_cached_tokens - seq.num_scheduled_tokens
        start = prev // self.block_size
        end = min(seq.num_full_blocks(), seq.num_cached_tokens // self.block_size)
        if start >= end:
            return
        self.index.publish(seq, end, self.blocks, start=start)

    def deallocate(self, seq: Sequence, *, publish: bool = True) -> None:
        n_full = seq.num_full_blocks() if publish else 0
        if publish and n_full > 0 and self.prefix_backend != "none":
            self.index.publish(seq, n_full, self.blocks)

        self.index.release_seq(seq.seq_id)

        published = set(seq.block_table[:n_full]) if n_full > 0 else set()
        for bid in list(seq.block_table):
            keep = bid in published and self.prefix_backend != "none"
            if not keep and self.prefix_backend == "hash":
                self.index.invalidate(bid, self.blocks)
            self.blocks.release_to_cached_or_free(bid, keep_cached=keep)
            if not keep and self.prefix_backend == "radix":
                self.index.invalidate(bid, self.blocks)

        seq.block_table.clear()
        seq.num_cached_tokens = 0
