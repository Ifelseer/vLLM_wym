"""ZOS 回落路径的回归测试：**不能再 schedule() 一次**。

真实事故：`AsyncDecode.step` 先 `schedule()` 判断这批能不能进流水线，不 eligible 时
调 `_step_sync()`，而 `_step_sync` 内部又 `schedule()` 了一次。第一次已经把请求从
waiting 移到 running 并设好 `num_scheduled_tokens=prompt_len`；第二次就把这条**刚
落地的 prefill 序列当成 decode 行**（`num_cached_tokens=0`，只调度 1 个 token），
于是 prefill 永远不执行，最后在 `to_wire()` 的不变量断言上炸掉：

    AssertionError: decode invariant: len(token_ids)=30 vs num_cached_tokens+1=1

所以 `_step_sync` 必须接受调用方已经排好的批。这里用最小替身把这条钉住。
"""

from __future__ import annotations

from types import SimpleNamespace

from src.control.core import EngineCore


class _CountingScheduler:
    def __init__(self, batch):
        self.batch = batch
        self.schedule_calls = 0
        self.updated = None
        self.eos = frozenset({2})

    def schedule(self):
        self.schedule_calls += 1
        return self.batch

    def is_finished(self):
        return False

    def update(self, sched, p_ids, d_ids, d_lens, d_prop):
        self.updated = (sched, p_ids, d_ids)
        return ["done"]


def _core(sched, batch):
    exec_ = SimpleNamespace(
        run=lambda s: (1, 0),
        read_samples=lambda n_p, n_d: ([7], [], None, None),
    )
    return SimpleNamespace(
        scheduler=sched,
        executor=exec_,
        kv=SimpleNamespace(num_free=8, last_num_cached_tokens=0),
        last_num_cached_tokens=0,
        _idle_steps=0,
        _MAX_IDLE_STEPS=1024,
        config=SimpleNamespace(spec_tokens=0),
    )


class TestStepSyncReuse:
    def test_reuses_given_batch_without_rescheduling(self):
        seq = SimpleNamespace(num_scheduled_tokens=5, num_cached_tokens=0)
        batch = SimpleNamespace(
            prefills=[seq], decodes=[], is_empty=lambda: False, sequences=[seq]
        )
        sched = _CountingScheduler(batch)
        core = _core(sched, batch)

        EngineCore._step_sync(core, batch)

        assert sched.schedule_calls == 0, "传入批时必须复用，不能再 schedule 一次"
        assert sched.updated is not None
        assert sched.updated[0] is batch

    def test_schedules_when_no_batch_given(self):
        """不传批时保持原行为（自排一次）。"""
        seq = SimpleNamespace(num_scheduled_tokens=1, num_cached_tokens=4)
        batch = SimpleNamespace(
            prefills=[], decodes=[seq], is_empty=lambda: False, sequences=[seq]
        )
        sched = _CountingScheduler(batch)
        core = _core(sched, batch)

        EngineCore._step_sync(core)

        assert sched.schedule_calls == 1
