from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count

from ..sampling_params import SamplingParams

_SEQ_IDS = count(1)

# ==============================================================================
# 一条请求在 CPU 上的状态。
#
# Engine.add_request / Scheduler / KVManager 读写本对象。
# 发给 Worker 前由 ScheduledBatch.to_wire() 收成 SeqSchedule。
# ==============================================================================


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


def slots_for_range(
    block_table: list[int],
    start: int,
    end: int,
    block_size: int,
) -> list[int]:
    """[start, end) 每个 token 对应的物理 slot。按块 stretch range，避免逐 token 做 // 和 %。"""
    n = end - start
    if n == 0:
        return []
    b = start // block_size
    off = start - b * block_size
    if off + n <= block_size:
        base = block_table[b] * block_size
        return list(range(base + off, base + off + n))
    slots: list[int] = []
    last = (end - 1) // block_size
    while b <= last:
        lo = start if b == start // block_size else b * block_size
        hi = end if b == last else (b + 1) * block_size
        base = block_table[b] * block_size
        slots.extend(range(base + lo - b * block_size, base + hi - b * block_size))
        b += 1
    return slots


@dataclass
class Sequence:
    token_ids: list[int]
    block_size: int
    seq_id: int = 0
    status: SequenceStatus = SequenceStatus.WAITING
    block_table: list[int] = field(default_factory=list)
    num_cached_tokens: int = 0
    num_scheduled_tokens: int = 0
    # 本步 prefill 是否已经覆盖完整 prompt，需要采样。
    need_sample: bool = False
    is_prefill: bool = True
    num_prompt_tokens: int = 0
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int = 128
    ignore_eos: bool = False
    last_token: int = 0

    @classmethod
    def from_prompt(
        cls,
        token_ids: list[int],
        *,
        block_size: int,
        sampling: SamplingParams | None = None,
    ) -> Sequence:
        sp = sampling or SamplingParams()
        ids = copy(token_ids)
        if not ids:
            raise ValueError("empty prompt")
        return cls(
            token_ids=ids,
            block_size=block_size,
            seq_id=next(_SEQ_IDS),
            num_prompt_tokens=len(ids),
            last_token=ids[-1],
            temperature=sp.temperature,
            top_p=sp.top_p,
            top_k=sp.top_k,
            min_p=sp.min_p,
            max_tokens=sp.max_tokens,
            ignore_eos=sp.ignore_eos,
        )

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def num_blocks(self) -> int:
        n = self.num_tokens
        if n == 0:
            return 0
        return (n + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        if self.num_blocks == 0:
            return 0
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def num_full_blocks(self) -> int:
        return self.num_tokens // self.block_size

    def block_token_ids(self, i: int) -> list[int]:
        assert 0 <= i < self.num_blocks
        start = i * self.block_size
        end = min(start + self.block_size, self.num_tokens)
        return self.token_ids[start:end]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.last_token = token_id
