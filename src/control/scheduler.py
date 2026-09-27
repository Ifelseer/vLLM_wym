from __future__ import annotations

from collections import deque

from ..data import EngineCoreOutput, KVManager, ScheduledBatch, Sequence, SequenceStatus


class Scheduler:
    """请求调度器（并行无关）。

    只管理 waiting / running 队列与 KV 逻辑块预算，产出 ScheduledBatch。
    TP/PP/DP 均不在此体现：同一份 schedule 由 Executor 广播（TP/PP）或由各 DP
    副本各自独立调度（DP）。
    """

    def __init__(
        self,
        kv: KVManager,
        *,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        eos_token_id: int,
        mix_prefill_decode: bool = True,
    ) -> None:
        self.kv = kv
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos = eos_token_id
        self.mix_prefill_decode = mix_prefill_decode
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def schedule(self) -> ScheduledBatch:
        if self.mix_prefill_decode:
            p, d = self._schedule_mixed()
        else:
            p, d = self._schedule_prefill_first()
        return ScheduledBatch(prefills=p, decodes=d)

    def _room(self, prefills: list[Sequence], decodes: list[Sequence]) -> bool:
        return len(prefills) + len(decodes) < self.max_num_seqs

    def _try_alloc(self, seq: Sequence) -> bool:
        if seq.block_table:
            return True
        if not self.kv.can_allocate(seq):
            return False
        self.kv.allocate(seq)
        return True

    def _schedule_decode(
        self,
        prefills: list[Sequence],
        decodes: list[Sequence],
        tokens_used: int,
    ) -> int:
        while self.running and self._room(prefills, decodes):
            if tokens_used + 1 > self.max_num_batched_tokens:
                break

            seq = self.running.popleft()

            while not self.kv.can_append(seq) and self.running:
                self.preempt(self.running.pop())

            if not self.kv.can_append(seq):
                self.preempt(seq)
                break

            seq.num_scheduled_tokens = 1
            self.kv.may_append(seq)
            decodes.append(seq)
            tokens_used += 1

        if decodes:
            self.running.extendleft(reversed(decodes))

        return tokens_used

    def _pull_prefill_chunk(
        self,
        prefills: list[Sequence],
        decodes: list[Sequence],
        tokens_used: int,
        *,
        allow_trailing_partial: bool = True,
    ) -> int:
        if not self.waiting or not self._room(prefills, decodes):
            return tokens_used

        n_partial = 0
        for seq in list(self.waiting):
            if not self._room(prefills, decodes):
                break

            remaining = self.max_num_batched_tokens - tokens_used
            if remaining <= 0:
                break

            if not self._try_alloc(seq):
                break

            need = seq.num_tokens - seq.num_cached_tokens
            if need <= 0:
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                self.running.append(seq)
                continue

            if remaining < need and prefills and not allow_trailing_partial:
                break

            chunk = min(need, remaining)
            seq.num_scheduled_tokens = chunk
            prefills.append(seq)
            tokens_used += chunk

            if seq.num_cached_tokens + chunk == seq.num_tokens:
                seq.need_sample = True
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                self.running.append(seq)
            else:
                seq.need_sample = False
                n_partial += 1
                if n_partial >= 1:
                    break

        return tokens_used

    def _schedule_prefill_first(self) -> tuple[list[Sequence], list[Sequence]]:
        prefills: list[Sequence] = []
        decodes: list[Sequence] = []
        self._pull_prefill_chunk(prefills, decodes, 0, allow_trailing_partial=False)
        if prefills:
            return prefills, []
        self._schedule_decode(prefills, decodes, 0)
        return [], decodes

    def _schedule_mixed(self) -> tuple[list[Sequence], list[Sequence]]:
        prefills: list[Sequence] = []
        decodes: list[Sequence] = []
        used = self._schedule_decode(prefills, decodes, 0)
        if decodes:
            self._pull_prefill_chunk(prefills, decodes, used)
            return prefills, decodes
        return self._schedule_prefill_first()

    def preempt(self, seq: Sequence) -> None:
        if seq.status == SequenceStatus.FINISHED:
            return
        seq.status = SequenceStatus.WAITING
        self.kv.deallocate(seq, publish=False)
        self.waiting.appendleft(seq)

    def update(
        self, sched: ScheduledBatch, p_ids: list[int], d_ids: list[int]
    ) -> list[EngineCoreOutput]:
        outputs: list[EngineCoreOutput] = []
        if sched.decodes:
            outputs.extend(self._update_batch(sched.decodes, d_ids, is_prefill=False))
        if sched.prefills:
            outputs.extend(self._update_batch(sched.prefills, p_ids, is_prefill=True))
        return outputs

    def _update_batch(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        *,
        is_prefill: bool,
    ) -> list[EngineCoreOutput]:
        finished: set[int] = set()
        outputs: list[EngineCoreOutput] = []

        for seq, token_id in zip(seqs, token_ids):
            seq.num_cached_tokens += seq.num_scheduled_tokens
            self.kv.sync_prefix(seq)
            seq.num_scheduled_tokens = 0

            if is_prefill and seq.num_cached_tokens < seq.num_prompt_tokens:
                outputs.append(EngineCoreOutput(seq_id=seq.seq_id))
                continue

            if seq.max_tokens <= 0:
                seq.status = SequenceStatus.FINISHED
                self.kv.deallocate(seq, publish=True)
                finished.add(seq.seq_id)
                outputs.append(EngineCoreOutput(seq_id=seq.seq_id, is_finished=True))
                continue

            seq.append_token(token_id)

            hit_eos = (not seq.ignore_eos) and token_id == self.eos
            hit_len = seq.num_completion_tokens >= seq.max_tokens
            if hit_eos or hit_len:
                seq.status = SequenceStatus.FINISHED
                self.kv.deallocate(seq, publish=True)
                finished.add(seq.seq_id)

            outputs.append(
                EngineCoreOutput(
                    seq_id=seq.seq_id,
                    new_token_ids=[token_id],
                    is_finished=seq.seq_id in finished,
                )
            )

        if finished:
            self.running = deque(s for s in self.running if s.seq_id not in finished)
        return outputs
