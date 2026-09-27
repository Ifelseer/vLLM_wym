from __future__ import annotations

from dataclasses import dataclass, field

import msgspec

from .sequence import Sequence

# ==============================================================================
# 控制面与计算面之间的 CPU 数据结构。
#
# Scheduler.schedule 产出 ScheduledBatch（仍持有本进程 Sequence）。
# Executor.run 调用 to_wire()，得到可 msgspec 编码的 SchedulerOutput，
# 交给 Worker / ModelRunner 的 build_decode、build_prefill。
# Scheduler.update 产出 EngineCoreOutput，经 Client 回到 Engine。
#
# GPU 上的静态输入缓冲在 compute/batch.py。
# ==============================================================================


@dataclass
class ScheduledBatch:
    """本步要跑的 prefill / decode 序列。Scheduler 与 EngineCore 使用；Worker 只吃 to_wire()。"""

    prefills: list[Sequence] = field(default_factory=list)
    decodes: list[Sequence] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.prefills and not self.decodes

    def to_wire(self) -> "SchedulerOutput":
        """抽出本步前向需要的字段，不把整条 token 历史发给 Worker。"""
        return SchedulerOutput(
            prefills=[_prefill_schedule(s) for s in self.prefills],
            decodes=[_decode_schedule(s) for s in self.decodes],
        )


def _sampling_fields(s: Sequence) -> tuple[float, float | None, int | None, float | None]:
    return s.temperature, s.top_p, s.top_k, s.min_p


def _decode_schedule(s: Sequence) -> "SeqSchedule":
    """decode：一个新 token、缓存长度、块表（与 Sequence 同一 list，不拷贝）。"""
    assert len(s.token_ids) == s.num_cached_tokens + 1, (
        f"decode invariant: len(token_ids)={len(s.token_ids)} vs "
        f"num_cached_tokens+1={s.num_cached_tokens + 1}"
    )
    temperature, top_p, top_k, min_p = _sampling_fields(s)
    return SeqSchedule(
        seq_id=s.seq_id,
        block_table=s.block_table,
        num_cached_tokens=s.num_cached_tokens,
        token_ids=[s.last_token],
        need_sample=False,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )


def _prefill_schedule(s: Sequence) -> "SeqSchedule":
    """prefill：只发本步 scheduled 的 token 切片，不含已缓存前缀和尚未调度的尾巴。"""
    start = s.num_cached_tokens
    end = start + s.num_scheduled_tokens
    temperature, top_p, top_k, min_p = _sampling_fields(s)
    return SeqSchedule(
        seq_id=s.seq_id,
        block_table=s.block_table,
        num_cached_tokens=s.num_cached_tokens,
        token_ids=s.token_ids[start:end],
        need_sample=s.need_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )


class SeqSchedule(msgspec.Struct):
    """
        单条序列的前向输入（Gloo / 本进程都可以用）。

        token_ids：decode 为 [last_token]；prefill 为本步切片。
        num_cached_tokens：本步之前已写入 KV 的长度，用来算 position / cache_seqlen。
        block_table：与 Sequence 共享引用；GPU 侧按 seq_id 做行缓存，增量 H2D。
    """

    seq_id: int
    block_table: list[int]
    num_cached_tokens: int
    token_ids: list[int]
    need_sample: bool = False
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None


class SchedulerOutput(msgspec.Struct):
    """to_wire 的结果：Worker.execute_model 的入参。"""

    prefills: list[SeqSchedule] = msgspec.field(default_factory=list)
    decodes: list[SeqSchedule] = msgspec.field(default_factory=list)


class EngineCoreOutput(msgspec.Struct):
    """
        本步一条序列的增量，给 Engine / MPClient。

        new_token_ids 为空：本步未采样（chunked prefill 中间段，或 max_tokens 已用尽）。
    """

    seq_id: int
    new_token_ids: list[int] = msgspec.field(default_factory=list)
    is_finished: bool = False
