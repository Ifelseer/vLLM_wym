"""CPU 上的「假 target + 真调度器/KV 账本」集成测试。

为什么需要它：Phase 1 的单测只验证单个动作（一次落账、一次分块），但真实
错误往往长在**多步串接**上——块表增量同步、prefill→decode 的交接、
满页边界、接受数在 0..k 之间跳动时的偏移。这些在 GPU 上要加载模型才能暴露，
而账本本身与模型无关，可以先在 CPU 上跑几十步把不变量压住。

做法：构造一条"标准答案"token 序列 GT，用一个确定性草稿器（直接给 GT 的
后续 token）+ 一个假 target（argmax 永远等于 GT）驱动真实的
``Scheduler.schedule()`` / ``update()``，于是：

- 全接受时，生成结果必须**逐 token 等于 GT**（证明位置/偏移没错）；
- 每一步都必须满足 ``len(token_ids) == num_cached_tokens + 1``；
- 块表长度既不能小于需求（少了会越界写），也不能无上限增长（泄漏）。
"""

from __future__ import annotations

import pytest

from src.control.scheduler import Scheduler
from src.data import KVManager, Sequence
from src.spec.verify import accept_greedy

BLOCK = 16
PROMPT_LEN = 10
MAX_TOKENS = 40
SPEC_K = 4

# 标准答案：任意确定的长序列（0 保留给 pad 语义，这里只用 100+ 的 id）
GT = [100 + i for i in range(PROMPT_LEN + MAX_TOKENS + SPEC_K + 4)]


class _GroundTruthDrafter:
    """草稿 = 标准答案的后续 token（全部会被接受）。"""

    def propose(self, seq: Sequence, k: int) -> list[int]:
        pos = len(seq.token_ids)
        return GT[pos : pos + k]


class _NoisyDrafter:
    """草稿 = 标准答案，但 target 会周期性拒绝（模拟部分接受）。"""

    def __init__(self) -> None:
        self.calls = 0

    def propose(self, seq: Sequence, k: int) -> list[int]:
        self.calls += 1
        pos = len(seq.token_ids)
        return GT[pos : pos + k]


def _scheduler(drafter, *, spec_k: int, num_blocks: int = 64) -> Scheduler:
    kv = KVManager(num_blocks=num_blocks, block_size=BLOCK, prefix_backend="none")
    return Scheduler(
        kv,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        eos_token_id=2,
        spec_tokens=spec_k,
        max_seq_len=PROMPT_LEN + MAX_TOKENS + SPEC_K + 4,
        drafter=drafter,
    )


def _invariants(seq: Sequence, num_blocks: int) -> None:
    """每一步都必须成立的账本不变量（序列已结束时状态会被拆除，见调用点）。"""
    assert len(seq.token_ids) == seq.num_cached_tokens + 1, (
        f"len={len(seq.token_ids)} cached={seq.num_cached_tokens}"
    )
    # 块只需要覆盖「已写入 KV 的位置」0..cached-1；最后那个 pending token 的 KV
    # 要等下一步才写，届时 may_append 会再分配。写成 ceil((cached+1)/B) 会在
    # cached 恰好是块边界时误报（这一步写的是 0..cached-1，正好填满）。
    need = (seq.num_cached_tokens + BLOCK - 1) // BLOCK
    assert len(seq.block_table) >= need, (
        f"块表 {len(seq.block_table)} 少于需求 {need}（已写 KV 越界）"
    )
    assert len(seq.block_table) <= num_blocks
    assert len(set(seq.block_table)) == len(seq.block_table), "块号重复"


def _run_loop(sched: Scheduler, *, reject_at=None) -> Sequence:
    seq = Sequence(
        token_ids=GT[:PROMPT_LEN],
        block_size=BLOCK,
        seq_id=1,
        num_prompt_tokens=PROMPT_LEN,
        max_tokens=MAX_TOKENS,
        ignore_eos=True,
    )
    sched.add(seq)

    guard = 0
    while not seq.is_finished:
        guard += 1
        assert guard < 500, "scheduler loop did not terminate"
        batch = sched.schedule()
        assert not batch.is_empty()

        if batch.prefills:
            assert len(batch.prefills) == 1
            s = batch.prefills[0]
            # 假 target 的 prefill 采样 = GT 的下一个 token
            sched.update(batch, [GT[len(s.token_ids)]], [])
        else:
            s = batch.decodes[0]
            k = len(s.draft_token_ids)
            pos = len(s.token_ids)  # = anchor 位置 + 1
            preds = GT[pos : pos + k + 1]  # 位置 p+i 预测 GT[p+i+1]
            if reject_at is not None and guard % reject_at == 0:
                # 人为让某个位置"预测不同" → 触发拒绝分支
                preds = list(preds)
                preds[0] = 999
            toks, _ = accept_greedy(list(s.draft_token_ids), preds)
            sched.update(batch, [], toks, [len(toks)])

        if seq.is_finished:
            # 结束时会 deallocate → block_table 清空、num_cached_tokens 归零
            assert seq.block_table == [] and seq.num_cached_tokens == 0
        else:
            _invariants(seq, sched.kv.num_blocks_total)
    return seq


class TestSchedulerLoop:
    def test_full_acceptance_reproduces_ground_truth(self):
        sched = _scheduler(_GroundTruthDrafter(), spec_k=SPEC_K)
        seq = _run_loop(sched)
        want = GT[: PROMPT_LEN + MAX_TOKENS]
        assert seq.token_ids == want, "全接受时生成结果必须逐 token 等于 GT"
        # 投机确实生效了：步数远小于 token 数
        assert sched.spec_stats["steps"] < MAX_TOKENS
        assert sched.spec_stats["accepted"] > 0

    def test_single_token_mode_same_output(self):
        """spec=0 用同一套 GT 语义跑，输出必须与全接受一致（对照系）。"""
        sched = _scheduler(_GroundTruthDrafter(), spec_k=0)
        seq = _run_loop(sched)
        assert seq.token_ids == GT[: PROMPT_LEN + MAX_TOKENS]
        assert sched.spec_stats["steps"] == 0

    def test_partial_rejection_keeps_invariants(self):
        """周期性拒绝：只校验不变量（不校验输出，因为拒绝会改写 token）。"""
        drafter = _NoisyDrafter()
        sched = _scheduler(drafter, spec_k=SPEC_K)
        seq = _run_loop(sched, reject_at=3)
        assert seq.is_finished
        assert seq.num_completion_tokens == MAX_TOKENS
        # 拒绝确实发生过：接受数少于提出的草稿数
        assert sched.spec_stats["accepted"] < sched.spec_stats["proposed"]

    def test_tight_kv_forces_fallback(self):
        """KV 很紧时应退化为普通 decode 而不是崩/越界。"""
        sched = _scheduler(_GroundTruthDrafter(), spec_k=SPEC_K, num_blocks=12)
        seq = _run_loop(sched)
        assert seq.token_ids == GT[: PROMPT_LEN + MAX_TOKENS]

    def test_context_limit_stops_drafting(self):
        """接近 context 上限时不再提草稿（k 被夹到 0）。"""
        sched = _scheduler(_GroundTruthDrafter(), spec_k=SPEC_K)
        seq = Sequence(
            token_ids=GT[: PROMPT_LEN],
            block_size=BLOCK,
            seq_id=2,
            num_cached_tokens=PROMPT_LEN - 1,
            num_prompt_tokens=PROMPT_LEN,
            max_tokens=MAX_TOKENS,
            ignore_eos=True,
        )
        sched.max_seq_len = PROMPT_LEN  # 只剩 anchor 一个位置
        drafts, write = sched._propose(seq, tokens_used=0)
        assert drafts == [] and write == 1
