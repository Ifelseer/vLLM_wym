from __future__ import annotations

from collections import deque

from ..data import EngineCoreOutput, KVManager, ScheduledBatch, Sequence, SequenceStatus
from ..spec import Drafter, make_drafter


class Scheduler:
    """请求调度器（并行无关）。

    只管理 waiting / running 队列与 KV 逻辑块预算，产出 ScheduledBatch。
    TP/PP/DP 均不在此体现：同一份 schedule 由 Executor 广播（TP/PP）或由各 DP
    副本各自独立调度（DP）。

    投机解码（``spec_tokens > 0``）：decode 阶段先由 ``Drafter`` 产出草稿，
    本步写入 ``1+k`` 个位置做验证；验证结果里「接受 j 个 + 1 个新 token」
    由 ``update`` 落账（``num_cached_tokens += 1+j``）。草稿不进入
    ``Sequence.token_ids``，被拒的草稿不会污染长度与块账本。
    """

    def __init__(
        self,
        kv: KVManager,
        *,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        eos_token_id: int,
        mix_prefill_decode: bool = True,
        spec_tokens: int = 0,
        spec_draft: str = "ngram",
        max_seq_len: int | None = None,
        drafter: Drafter | None = None,
    ) -> None:
        self.kv = kv
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos = eos_token_id
        self.mix_prefill_decode = mix_prefill_decode
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

        # --- 投机解码 ---
        self.spec_tokens = max(0, spec_tokens)
        self.max_seq_len = max(0, max_seq_len or 0)
        if self.spec_tokens > 0 and max_num_seqs > 1:
            # v1 限制：verify 路径按单序列实现（树形验证需要每序列一棵树，
            # 且要按序列拼平 cu_seqlens，留到 v2）。
            raise ValueError(
                "speculative decoding (spec_tokens > 0) requires "
                f"max_num_seqs == 1 in v1, got {max_num_seqs}"
            )
        if self.spec_tokens <= 0:
            self.drafter: Drafter | None = None
        elif drafter is not None:
            # 允许注入（测试用的确定性草稿；Phase 3 也可能在外部构造 GPU 草稿）
            self.drafter = drafter
        else:
            self.drafter = make_drafter(spec_draft, num_spec_tokens=self.spec_tokens)
        # 观测口径（bench / 测试断言用）：提出与接受的草稿数、验证步数。
        # 接受率 = accepted / proposed 是投机解码最直接的收益指标。
        self.spec_stats: dict[str, int] = {
            "steps": 0,
            "proposed": 0,
            "accepted": 0,
        }

    def _propose(self, seq: Sequence, tokens_used: int) -> tuple[list[int], int]:
        """产出本步草稿并算出要写入的位置数（1 = 无草稿的普通 decode）。"""
        if self.drafter is None or self.spec_tokens <= 0:
            return [], 1
        k = self.spec_tokens
        # 位置 p 是本步 anchor；草稿占 p+1..p+k，必须留在一个 context 内
        if self.max_seq_len > 0:
            k = min(k, self.max_seq_len - 1 - seq.num_cached_tokens)
        # token budget：anchor 占 1，其余留给草稿
        k = min(k, self.max_num_batched_tokens - tokens_used - 1)
        if k <= 0:
            return [], 1
        if getattr(self.drafter, "worker_side", False):
            # 草稿在 worker 侧产出：这里只按上限预留位置/块
            return [], 1 + k
        drafts = self.drafter.propose(seq, k)[:k]
        self.spec_stats["proposed"] += len(drafts)
        return drafts, 1 + len(drafts)

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

            drafts, write = self._propose(seq, tokens_used)
            # 块不够（KV 紧张）：先退化成普通 decode，再不行才抢占
            if write > 1 and not self.kv.can_append(seq, write):
                drafts, write = [], 1

            while not self.kv.can_append(seq, write) and self.running:
                self.preempt(self.running.pop())

            if not self.kv.can_append(seq, write):
                self.preempt(seq)
                break

            seq.draft_token_ids = drafts
            seq.num_scheduled_tokens = write
            self.kv.may_append(seq, write)
            decodes.append(seq)
            tokens_used += write

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
        self,
        sched: ScheduledBatch,
        p_ids: list[int],
        d_ids: list[int],
        d_lens: list[int] | None = None,
    ) -> list[EngineCoreOutput]:
        outputs: list[EngineCoreOutput] = []
        if sched.decodes:
            if self.spec_tokens > 0:
                outputs.extend(self._update_spec_decode(sched.decodes, d_ids, d_lens))
            else:
                outputs.extend(
                    self._update_batch(sched.decodes, d_ids, is_prefill=False)
                )
        if sched.prefills:
            outputs.extend(self._update_batch(sched.prefills, p_ids, is_prefill=True))
        return outputs

    def _update_spec_decode(
        self,
        seqs: list[Sequence],
        d_ids: list[int],
        d_lens: list[int] | None,
    ) -> list[EngineCoreOutput]:
        """投机 decode 的落账：接受 j 个草稿 + 1 个新 token。

        verify 一次写入了 ``1+k`` 个位置的 KV，但只有前 ``1+j`` 个是有效的
        （anchor + 被接受的草稿）。因此：

        - ``num_cached_tokens += 1 + j``：推进有效 KV 长度；
        - 用 ``num_scheduled_tokens = 1 + j`` 临时告给 ``sync_prefix``，
          让它只把「完全由有效 KV 覆盖」的块登记进前缀索引（被拒草稿所在
          的块仍可能被后续写入覆盖，不能 publish）；
        - 只把验证通过的 token append 进 ``token_ids``。
        """
        finished: set[int] = set()
        outputs: list[EngineCoreOutput] = []
        off = 0

        for i, seq in enumerate(seqs):
            n = d_lens[i] if d_lens is not None else 1
            toks = list(d_ids[off : off + n])
            off += n
            if not toks:
                # 理论上不会发生（verify 至少返回 1 个 token）；保守退化
                seq.draft_token_ids = []
                seq.num_scheduled_tokens = 0
                outputs.append(EngineCoreOutput(seq_id=seq.seq_id))
                continue

            accepted = max(0, len(toks) - 1)
            self.spec_stats["steps"] += 1
            self.spec_stats["accepted"] += accepted
            seq.num_accepted = accepted
            seq.draft_token_ids = []
            seq.num_cached_tokens += 1 + accepted
            seq.num_scheduled_tokens = 1 + accepted
            self.kv.sync_prefix(seq)
            seq.num_scheduled_tokens = 0

            appended: list[int] = []
            is_finished = False
            for token_id in toks:
                if seq.max_tokens <= 0:
                    is_finished = True
                    break
                seq.append_token(token_id)
                appended.append(token_id)
                hit_eos = (not seq.ignore_eos) and token_id == self.eos
                hit_len = seq.num_completion_tokens >= seq.max_tokens
                if hit_eos or hit_len:
                    is_finished = True
                    break

            if is_finished:
                seq.status = SequenceStatus.FINISHED
                self.kv.deallocate(seq, publish=True)
                finished.add(seq.seq_id)

            outputs.append(
                EngineCoreOutput(
                    seq_id=seq.seq_id,
                    new_token_ids=appended,
                    is_finished=is_finished,
                )
            )

        if finished:
            self.running = deque(s for s in self.running if s.seq_id not in finished)
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
