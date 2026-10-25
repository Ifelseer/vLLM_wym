from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..data.sequence import Sequence


@runtime_checkable
class Drafter(Protocol):
    """草稿器接口：给定一条序列，给出至多 ``k`` 个候选 token。

    约定：

    - 返回值**不包含** anchor（``seq.last_token``）——anchor 由 verify 路径
      自己放在第 0 位；
    - 长度可以小于 ``k``（没有把握时少给），也可以是空（本步退化为普通 decode）；
    - 不得修改 ``seq``（纯函数语义，便于单测与并行预取）。
    """

    def propose(self, seq: Sequence, k: int) -> list[int]:
        """返回至多 ``k`` 个候选 token id。"""
        ...
