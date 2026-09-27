from __future__ import annotations

from typing import TYPE_CHECKING

from .hash_util import compute_block_hash

if TYPE_CHECKING:
    from ..block_manager import BlockManager
    from ..sequence import Sequence

# ==============================================================================
# 前缀索引：hash → 逻辑页号。
#
# KVManager.allocate 调 match；写满一块后 publish。
# 同一哈希被后写的块覆盖；驱逐时 invalidate 删掉映射。
# ==============================================================================


class HashPrefixIndex:
    def __init__(self) -> None:
        self.hash_to_block_id: dict[int, int] = {}

    def match(self, seq: Sequence, pool: BlockManager) -> list[int]:
        h = -1
        hit: list[int] = []
        for i in range(seq.num_full_blocks()):
            toks = seq.block_token_ids(i)
            h = compute_block_hash(toks, h)
            bid = self.hash_to_block_id.get(h, -1)
            if bid < 0:
                break
            block = pool.blocks[bid]
            if block.token_ids != toks:
                break
            if bid not in pool.used_ids and bid not in pool.cached_ids:
                break
            hit.append(bid)
        return hit

    def publish(
        self,
        seq: Sequence,
        num_full_blocks: int,
        pool: BlockManager,
        *,
        start: int = 0,
    ) -> None:
        if num_full_blocks <= start:
            return
        h = -1
        if start > 0:
            h = pool.blocks[seq.block_table[start - 1]].hash
            if h == -1:
                start = 0
        for i in range(start, num_full_blocks):
            bid = seq.block_table[i]
            toks = seq.block_token_ids(i)
            h = compute_block_hash(toks, h)
            pool.blocks[bid].update(h, toks)
            old = self.hash_to_block_id.get(h, -1)
            if old < 0 or old == bid or pool.blocks[old].token_ids == toks:
                self.hash_to_block_id[h] = bid

    def invalidate(self, block_id: int, pool: BlockManager) -> None:
        block = pool.blocks[block_id]
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]

    def release_seq(self, seq_id: int) -> None:
        return
