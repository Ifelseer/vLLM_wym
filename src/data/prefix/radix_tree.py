from __future__ import annotations

import heapq
import time
from collections.abc import Sequence as AbcSequence

# ==============================================================================
# 压缩 radix 树：key 是链式块哈希，value 是逻辑页号。
#
# RadixPrefixIndex 用来 match / insert / 按叶子 LRU evict。
# 节点带 ref_count；只有 ref=0 的叶子才能被摘掉。
# ==============================================================================


def _prefix_len(a: AbcSequence[int], b: AbcSequence[int]) -> int:
    n = 0
    m = min(len(a), len(b))
    while n < m and a[n] == b[n]:
        n += 1
    return n


class RadixNode:
    __slots__ = ("children", "parent", "key", "value", "ref_count", "access_time")

    def __init__(
        self,
        *,
        parent: RadixNode | None = None,
        key: tuple[int, ...] = (),
        value: tuple[int, ...] = (),
        ref_count: int = 0,
    ) -> None:
        self.children: dict[int, RadixNode] = {}
        self.parent = parent
        self.key = key
        self.value = value
        self.ref_count = ref_count
        self.access_time = time.monotonic()

    def __lt__(self, other: RadixNode) -> bool:
        return self.access_time < other.access_time


class RadixTree:
    def __init__(self) -> None:
        self.root = RadixNode(ref_count=1)
        self._block_to_node: dict[int, RadixNode] = {}

    def contains_block(self, block_id: int) -> bool:
        return block_id in self._block_to_node

    def match_prefix(self, key: list[int]) -> tuple[list[int], RadixNode]:
        if not key:
            return [], self.root
        node = self.root
        values: list[int] = []
        child_key = key[0]
        while key and child_key in node.children:
            child = node.children[child_key]
            child.access_time = time.monotonic()
            pl = _prefix_len(child.key, key)
            if pl < len(child.key):
                new_node = self._split(child, pl)
                values.extend(new_node.value)
                return values, new_node
            values.extend(child.value)
            node = child
            key = key[pl:]
            child_key = key[0] if key else -1
        return values, node

    def insert(self, key: list[int], value: list[int]) -> tuple[int, RadixNode]:
        """插入 key→页号。返回（已有公共前缀长度, 叶子）。"""
        if not key:
            return 0, self.root
        node = self.root
        child_key = key[0]
        matched = 0
        while key and child_key in node.children:
            child = node.children[child_key]
            child.access_time = time.monotonic()
            pl = _prefix_len(child.key, key)
            matched += pl
            key = key[pl:]
            value = value[pl:]
            child_key = key[0] if key else -1
            if pl < len(child.key):
                node = self._split(child, pl)
                break
            node = child
        last = node
        if key:
            last = self._add(node, tuple(key), tuple(value))
        return matched, last

    def inc_ref(self, node: RadixNode) -> None:
        while node is not self.root:
            node.ref_count += 1
            node = node.parent or self.root

    def dec_ref(self, node: RadixNode) -> None:
        while node is not self.root:
            node.ref_count -= 1
            node = node.parent or self.root

    def detach_leaf_for_block(self, block_id: int) -> list[int]:
        """页被池子丢掉时，若所属节点无人引用，整段从树上摘掉。"""
        node = self._block_to_node.get(block_id)
        if node is None or node is self.root or node.ref_count > 0:
            return []
        freed = list(node.value)
        self._remove(node)
        return freed

    def evict(self, num_slots: int) -> list[int]:
        """按 access_time 丢掉无人引用的叶子，直到凑够页数。"""
        leaves = self._leaves()
        heapq.heapify(leaves)
        freed: list[int] = []
        while len(freed) < num_slots and leaves:
            node = heapq.heappop(leaves)
            if node is self.root or node.ref_count > 0:
                continue
            freed.extend(node.value)
            parent = node.parent
            self._remove(node)
            if parent is not None and parent is not self.root and not parent.children:
                heapq.heappush(leaves, parent)
        return freed

    def _index(self, node: RadixNode) -> None:
        for bid in node.value:
            self._block_to_node[bid] = node

    def _unindex(self, node: RadixNode) -> None:
        for bid in node.value:
            if self._block_to_node.get(bid) is node:
                del self._block_to_node[bid]

    def _leaves(self) -> list[RadixNode]:
        out: list[RadixNode] = []

        def walk(n: RadixNode) -> None:
            if not n.children:
                if n is not self.root:
                    out.append(n)
                return
            for c in n.children.values():
                walk(c)

        walk(self.root)
        return out

    def _remove(self, node: RadixNode) -> None:
        if node.parent is None or not node.key:
            return
        self._unindex(node)
        node.parent.children.pop(node.key[0], None)

    def _add(
        self, parent: RadixNode, key: tuple[int, ...], value: tuple[int, ...]
    ) -> RadixNode:
        node = RadixNode(parent=parent, key=key, value=value)
        parent.children[key[0]] = node
        self._index(node)
        return node

    def _split(self, node: RadixNode, split_at: int) -> RadixNode:
        self._unindex(node)
        new_node = RadixNode(
            parent=node.parent,
            key=node.key[:split_at],
            value=node.value[:split_at],
            ref_count=node.ref_count,
        )
        new_node.children = {node.key[split_at]: node}
        node.parent = new_node
        node.key = node.key[split_at:]
        node.value = node.value[split_at:]
        if new_node.parent is not None and new_node.key:
            new_node.parent.children[new_node.key[0]] = new_node
        self._index(new_node)
        self._index(node)
        return new_node
