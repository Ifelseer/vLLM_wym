from __future__ import annotations

from typing import Literal

from .hash_index import HashPrefixIndex
from .noop import NoopPrefixIndex
from .radix_index import RadixPrefixIndex

# KVManager 按 config.prefix_backend 在这里选实现。
PrefixBackend = Literal["none", "hash", "radix"]


def make_prefix_index(backend: PrefixBackend):
    if backend == "hash":
        return HashPrefixIndex()
    if backend == "radix":
        return RadixPrefixIndex()
    return NoopPrefixIndex()


__all__ = ["PrefixBackend", "make_prefix_index"]
