"""投机解码 Phase 1 的 CPU 单测：草稿器 / 块账本 / 调度落账。

这三部分都不需要 GPU，但决定了**正确性的地基**：

- 草稿器给错 token 无所谓（会被 verify 拒掉），但给的长度超过预算会让
  verify 越界写 KV；
- 块账本算错会让被拒草稿占用的块无法复用（显存泄漏）或重复分配；
- 落账算错会让 ``len(token_ids) == num_cached_tokens + 1`` 这个全引擎依赖的
  不变量失效。
"""

from __future__ import annotations

import pytest

from src.control.scheduler import Scheduler
from src.data import KVManager, ScheduledBatch, Sequence
from src.spec import Drafter, NgramDrafter, make_drafter
from src.spec.verify import accept_greedy

BLOCK = 16


def _seq(
    ids: list[int],
    *,
    cached: int | None = None,
    num_prompt: int | None = None,
    max_tokens: int = 64,
    block_table: list[int] | None = None,
) -> Sequence:
    n_prompt = len(ids) - 1 if num_prompt is None else num_prompt
    return Sequence(
        token_ids=list(ids),
        block_size=BLOCK,
        seq_id=1,
        num_cached_tokens=len(ids) - 1 if cached is None else cached,
        num_prompt_tokens=n_prompt,
        max_tokens=max_tokens,
        block_table=list(block_table or []),
    )


# ---------------------------------------------------------------- #
# 1. NgramDrafter
# ---------------------------------------------------------------- #
class TestNgramDrafter:
    def test_finds_recent_match(self):
        d = NgramDrafter(n=3, max_tokens=7)
        # 末尾 [1,2,3] 在 pos=0 出现过，其后是 [4,5,1]
        seq = _seq([1, 2, 3, 4, 5, 1, 2, 3])
        assert d.propose(seq, 3) == [4, 5, 1]

    def test_respects_k(self):
        d = NgramDrafter(n=3, max_tokens=7)
        seq = _seq([1, 2, 3, 4, 5, 1, 2, 3])
        assert d.propose(seq, 1) == [4]

    def test_respects_max_tokens(self):
        d = NgramDrafter(n=3, max_tokens=2)
        seq = _seq([1, 2, 3, 4, 5, 1, 2, 3])
        assert d.propose(seq, 7) == [4, 5]

    def test_no_match_returns_empty(self):
        d = NgramDrafter(n=3)
        assert d.propose(_seq([1, 2, 3, 9, 8, 7]), 4) == []

    def test_too_short(self):
        d = NgramDrafter(n=3)
        assert d.propose(_seq([1, 2, 3]), 4) == []

    def test_most_recent_match_wins(self):
        d = NgramDrafter(n=2)
        # [7,8] 出现两次：pos=0 后面是 9，pos=5 后面是 6
        seq = _seq([7, 8, 9, 0, 0, 7, 8, 6, 7, 8])
        assert d.propose(seq, 2) == [6, 7]

    def test_generation_budget(self):
        d = NgramDrafter(n=2)
        # num_completion = len - num_prompt = 1，max_tokens=1 → 无预算
        seq = _seq([7, 8, 9, 7, 8], num_prompt=4, max_tokens=1)
        assert d.propose(seq, 2) == []

    def test_factory(self):
        assert isinstance(make_drafter("ngram", num_spec_tokens=4), NgramDrafter)
        assert isinstance(NgramDrafter(), Drafter)  # runtime_checkable Protocol

    def test_dspark_factory_is_worker_side_marker(self):
        """dspark 的草稿在 worker 侧产出：调度器只拿一个"预留额度"标记。"""
        marker = make_drafter("dspark", num_spec_tokens=4)
        assert getattr(marker, "worker_side", False) is True
        assert marker.propose(None, 4) == []

    def test_unknown_draft_raises(self):
        with pytest.raises(NotImplementedError):
            make_drafter("medusa", num_spec_tokens=4)


# ---------------------------------------------------------------- #
# 1b. greedy 接受裁决（off-by-one 的集中地，必须 CPU 全枚举）
# ---------------------------------------------------------------- #
class TestAcceptGreedy:
    def test_all_accepted_returns_bonus(self):
        # drafts 3 个全中 → 追加 3 个草稿 + 1 个 bonus（位置 p+3 的 argmax）
        toks, acc = accept_greedy([10, 11, 12], [10, 11, 12, 13])
        assert toks == [10, 11, 12, 13]
        assert acc == 3

    def test_first_rejected(self):
        # preds[0] 已经不等于 d1 → 一个都不接受，用它作为新 token
        toks, acc = accept_greedy([10, 11], [99, 11, 12])
        assert toks == [99]
        assert acc == 0

    def test_middle_rejected(self):
        # d1 中、d2 不中 → 追加 [d1] + preds[1]
        toks, acc = accept_greedy([10, 11, 12], [10, 99, 12, 13])
        assert toks == [10, 99]
        assert acc == 1

    def test_no_drafts(self):
        toks, acc = accept_greedy([], [7])
        assert toks == [7]
        assert acc == 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="len\\(drafts\\)\\+1"):
            accept_greedy([10, 11], [10, 11])


# ---------------------------------------------------------------- #
# 2. 块账本：一次追加 n 个位置
# ---------------------------------------------------------------- #
class TestBlockAppendN:
    def test_allocates_for_multiple_tokens(self):
        kv = KVManager(num_blocks=32, block_size=BLOCK, prefix_backend="none")
        seq = _seq(list(range(9)), cached=8)  # 已缓存 8，本步写 1+4=5 → 到 13
        assert kv.can_append(seq, 5)
        kv.may_append(seq, 5)
        assert len(seq.block_table) == 1  # ceil(13/16)=1
        # 再写 4 个（到 17）→ 需要第 2 块
        seq.num_cached_tokens = 13
        kv.may_append(seq, 4)
        assert len(seq.block_table) == 2

    def test_idempotent_after_rejection(self):
        """被拒草稿不会重复分配：块表已多于逻辑需求时不再申请。"""
        kv = KVManager(num_blocks=32, block_size=BLOCK, prefix_backend="none")
        seq = _seq(list(range(33)), cached=32)
        kv.may_append(seq, 1)  # 写位置 32 → ceil(33/16)=3 块
        assert len(seq.block_table) == 3
        # 模拟上一步 verify 写了 5 个位置但只接受 0 个：逻辑长度仍是 32
        seq.num_cached_tokens = 32
        kv.may_append(seq, 1)
        assert len(seq.block_table) == 3, "逻辑长度未增长时不应再分配"

    def test_can_append_respects_capacity(self):
        kv = KVManager(num_blocks=2, block_size=BLOCK, prefix_backend="none")
        seq = _seq(list(range(2)), cached=1)
        kv.may_append(seq, 1)  # 用掉 1 块
        seq.num_cached_tokens = 16
        # 还需要 ceil(17/16)=2 块 → 已有 1 块 → 缺 1 块，正好可用
        assert kv.can_append(seq, 1)
        kv.may_append(seq, 1)
        # 现在没有空闲块，写第 33 个位置需要第 3 块 → 不可
        seq.num_cached_tokens = 32
        assert not kv.can_append(seq, 1)


# ---------------------------------------------------------------- #
# 3. 调度：草稿预算 + 落账不变量
# ---------------------------------------------------------------- #
def _scheduler(*, spec_tokens: int = 4, max_seq_len: int = 0,
               max_num_batched_tokens: int = 8192) -> Scheduler:
    kv = KVManager(num_blocks=64, block_size=BLOCK, prefix_backend="none")
    return Scheduler(
        kv,
        max_num_seqs=1,
        max_num_batched_tokens=max_num_batched_tokens,
        eos_token_id=2,
        spec_tokens=spec_tokens,
        max_seq_len=max_seq_len,
    )


class TestSchedulerSpec:
    def test_requires_single_sequence(self):
        kv = KVManager(num_blocks=8, block_size=BLOCK, prefix_backend="none")
        with pytest.raises(ValueError, match="max_num_seqs == 1"):
            Scheduler(
                kv,
                max_num_seqs=4,
                max_num_batched_tokens=64,
                eos_token_id=2,
                spec_tokens=4,
            )

    def test_propose_respects_context_limit(self):
        s = _scheduler(max_seq_len=8)
        seq = _seq(list(range(8)), cached=7)  # 位置 7 是 anchor，没有空间给草稿
        drafts, write = s._propose(seq, tokens_used=0)
        assert drafts == [] and write == 1

    def test_propose_respects_token_budget(self):
        s = _scheduler(spec_tokens=7, max_num_batched_tokens=4)
        seq = _seq(list(range(4)), cached=3)
        drafts, write = s._propose(seq, tokens_used=0)
        assert write == 1 + len(drafts) <= 4

    def test_update_advances_cache_by_accepted_plus_anchor(self):
        s = _scheduler(spec_tokens=4)
        seq = _seq(list(range(6)), cached=5)  # len=6 == cached+1
        seq.draft_token_ids = [100, 101, 102, 103]
        sched = ScheduledBatch(decodes=[seq])

        # verify 结果：接受 2 个草稿 + 1 个新 token → 3 个 token
        s.update(sched, p_ids=[], d_ids=[100, 101, 900], d_lens=[3])

        assert seq.num_accepted == 2
        assert seq.num_cached_tokens == 5 + 1 + 2
        assert seq.token_ids[-3:] == [100, 101, 900]
        assert seq.draft_token_ids == []
        # 全引擎依赖的不变量
        assert len(seq.token_ids) == seq.num_cached_tokens + 1

    def test_update_with_zero_accept(self):
        s = _scheduler(spec_tokens=4)
        seq = _seq(list(range(6)), cached=5)
        seq.draft_token_ids = [100, 101]
        sched = ScheduledBatch(decodes=[seq])
        s.update(sched, p_ids=[], d_ids=[777], d_lens=[1])
        assert seq.num_accepted == 0
        assert seq.num_cached_tokens == 6
        assert seq.token_ids[-1] == 777
        assert len(seq.token_ids) == seq.num_cached_tokens + 1

    def test_update_stops_at_eos(self):
        s = _scheduler(spec_tokens=4)
        seq = _seq(list(range(6)), cached=5, max_tokens=64)
        seq.draft_token_ids = [5, 2, 6]  # 第 2 个是被接受的 EOS
        sched = ScheduledBatch(decodes=[seq])
        s.update(sched, p_ids=[], d_ids=[5, 2, 7], d_lens=[3])
        assert seq.is_finished
        assert seq.token_ids[-1] == 2, "EOS 之后的 token 不应被追加"
        assert not s.running

    def test_update_stops_at_max_tokens(self):
        s = _scheduler(spec_tokens=4)
        seq = _seq(list(range(4)), cached=3, num_prompt=3, max_tokens=2)
        sched = ScheduledBatch(decodes=[seq])
        s.update(sched, p_ids=[], d_ids=[10, 11, 12], d_lens=[3])
        assert seq.is_finished
        assert seq.num_completion_tokens == 2

    def test_no_spec_keeps_old_behaviour(self):
        """spec_tokens=0 时必须走原来的一 token 路径。"""
        s = _scheduler(spec_tokens=0)
        seq = _seq(list(range(6)), cached=5)
        seq.num_scheduled_tokens = 1  # _schedule_decode 会设置
        sched = ScheduledBatch(decodes=[seq])
        s.update(sched, p_ids=[], d_ids=[42], d_lens=None)
        assert seq.num_cached_tokens == 6
        assert seq.token_ids[-1] == 42
        assert len(seq.token_ids) == seq.num_cached_tokens + 1
