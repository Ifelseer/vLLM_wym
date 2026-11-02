"""ZOS（Zero-Overhead Scheduling）：把下一拍提交提前到上一拍结果处理之前。

## 要解决什么

稳定 decode 的每拍是串行的：

```
GPU 前向+采样 N → D2H 回读 token N → CPU 更新状态/备输入 → GPU 前向 N+1
                ↑ 这里 GPU 在空等（nsys 上就是相邻两次 forward 之间的 gap）
```

但下一拍要的输入**大部分已经确定**：行序不变、位置 +1、KV 长度 +1、跨页时换页。
唯一"未知"的 token 其实已经在 GPU 的采样输出里。所以 ZOS 把顺序改成：

```
提交 forward N+1（用设备上的 token N）→ 处理 N 的结果 → 提交 sample N+1
```

CPU 于是**落后一拍**：它处理的是上一拍的结果，而 GPU 已经开始下一拍。步时从
`G + C` 变成接近 `max(G, C)`。

## 三条不变量（写错不报错，只算错）

1. **行序必须完全不变**。`advance_decode` 靠"第 r 行还是上拍那条序列"来推进，
   所以一次 drain 之后要重新从普通调度器进入流水线；
2. **CPU 状态落后一拍**：`consume` 里才 `scheduler.update`，绝不能在提交前推进；
3. **在途计算不能被 CPU 动过的显存影响**：序列要结束（`deallocate` 释放页）或
   radix 要换页时，必须先等这一拍跑完（`eos_fences` / `prefix_fences`）。

## 什么时候停下来（drain）

任一条成立就回普通路径，本次不提前提交：

| 条件 | 计数 | 原因 |
|---|---|---|
| waiting 队列非空 | `waiting_drains` | 新请求要 prefill，批次结构会变 |
| running 行序与上拍不一致 | `batch_drains` | 行序变了，advance 会写错行 |
| 有序列即将结束 | `terminal_drains` | 下一拍会超 max_tokens / 已 EOS |
| KV 页不够 | `memory_drains` | 提前一拍要预留到 `num_cached + 1 + k` |

## 适用范围

- **只处理 greedy 的纯 decode 批**（温度采样要走 `Sampler` 的分支，拿不到固定缓冲）；
- **只支持单进程**（`tp=pp=dp=1`）——每个 rank 一张票的协调层尚未实现；
- 投机：接入依赖"接受裁决留在设备上"（`zos_device_accept`），`advances` 由设备侧给出。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data import ScheduledBatch

__all__ = ["AsyncDecode", "DecodeTicket"]


@dataclass
class DecodeTicket:
    """一拍的在途状态：GPU 已提交，CPU 尚未消费。"""

    sched: ScheduledBatch
    n_p: int
    n_d: int
    #: 本拍采样结果（device）——下一拍直接拿它当输入，不必等 D2H
    tokens_dev: torch.Tensor | None = None
    #: 下一拍每行的位置推进量（普通 decode 全 1；投机为 1+j，由设备侧给出）
    advances: torch.Tensor | None = None

    @property
    def seqs(self) -> list:
        return list(self.sched.decodes)


class AsyncDecode:
    def __init__(self, core) -> None:
        self.core = core
        self.pending: DecodeTicket | None = None
        self.stats: dict[str, int] = dict(
            steps=0,
            submitted=0,
            lookahead_submitted=0,
            drains=0,
            waiting_drains=0,
            batch_drains=0,
            terminal_drains=0,
            memory_drains=0,
            eos_fences=0,
            prefix_fences=0,
            ub_drains=0,
        )
        # 每 N 拍强制 drain 一次，把 ub（host 上界）用精确值重锚。
        # 不做的话 ub 只增不减，长生成最终所有桶都装不下 → 全部回退 eager。
        self._lookahead_streak = 0
        self._max_lookahead = 32
        #: 行序在两拍之间是否变过（换页需要刷新设备页表）
        self._tables_dirty = False

    # ---------------------------------------------------------------- #
    # 主循环
    # ---------------------------------------------------------------- #

    def step(self) -> tuple[list, int]:
        core = self.core
        self.stats["steps"] += 1

        if self.pending is None:
            sched = core.scheduler.schedule()
            if not self._eligible(sched):
                # 复用刚排好的批：再 schedule() 一次会把刚落地的 prefill 当成 decode
                return core._step_sync(sched)
            self.pending = self._submit(sched)
            # 首拍不返回任何 token：这一拍的结果下一拍才被消费
            return [], len(sched.decodes)

        prev = self.pending
        nxt: DecodeTicket | None = None
        if self._can_lookahead(prev):
            nxt = self._submit_ahead(prev)
            self.stats["lookahead_submitted"] += 1
        else:
            self.stats["drains"] += 1

        outputs = self._consume(prev, nxt)
        if nxt is not None and all(s.is_finished for s in nxt.seqs):
            # 消费完这一拍后全部结束了：在途那拍的结果不再需要，等它落地后丢弃
            # （不等的话，下一拍可能写进已释放的页）。**同时要丢掉它的回读条目**，
            # 否则下一次 `read_samples` 会先弹出这条过期数据，把 token 记到别人头上。
            core.executor.sync_samples()
            core.executor.discard_pending()
            nxt = None
        self.pending = nxt
        return outputs, (len(nxt.seqs) if nxt is not None else 0)

    def close(self) -> None:
        """销毁前排空在途计算。"""
        self.pending = None

    # ---------------------------------------------------------------- #
    # 能否进入 / 能否提前
    # ---------------------------------------------------------------- #

    def _eligible(self, sched: ScheduledBatch) -> bool:
        """只有"纯 greedy decode 批"且非投机时才进流水线。

        投机暂时排除：verify 走 ``build_prefill``，而那份元数据
        （positions / slot_mapping / cu_seqlens）是 **CPU 按 `num_cached_tokens` 建的**
        ——ZOS 下 CPU 状态落后一拍，用它建出来就是错的位置。要让投机进 ZOS，得先
        把 verify 的输入也做成设备侧推进（下一步，对应 DeepInfer 的"乐观预留 +
        设备侧纠正"）。
        """
        if sched.prefills or not sched.decodes:
            return False
        if self.core.config.spec_tokens > 0 and not self.core.config.zos_spec:
            return False
        if len(sched.decodes) > self.core.config.max_num_seqs:
            return False
        # 任一序列要温度采样 → 交给普通路径（图内只有 argmax）
        return not any(s.temperature > 0 for s in sched.decodes)

    def _can_lookahead(self, prev: DecodeTicket) -> bool:
        sched = self.core.scheduler
        kv = self.core.kv
        seqs = prev.seqs

        if not seqs or kv is None:
            return False
        if sched.waiting:  # 新请求要让调度器排 prefill
            self.stats["waiting_drains"] += 1
            return False
        if len(sched.running) != len(seqs) or any(
            cur is not old for cur, old in zip(sched.running, seqs)
        ):
            # 行序变了：advance 会按行号推进"别人"的状态
            self.stats["batch_drains"] += 1
            return False
        if any(
            s.is_finished or s.num_completion_tokens + 1 >= s.max_tokens for s in seqs
        ):
            self.stats["terminal_drains"] += 1
            return False
        if self._lookahead_streak >= self._max_lookahead:
            # 周期性退回同步拍：那一拍 host 值是精确的，正好把 ub 重锚回去
            self.stats["ub_drains"] += 1
            self._lookahead_streak = 0
            return False

        # 提前一拍要覆盖到 num_cached + n（投机为 1+k，普通为 1）
        n = 1 + max(0, self.core.config.spec_tokens)
        blocks = getattr(kv, "blocks", None)
        if blocks is not None and hasattr(blocks, "needed_append"):
            # **按总量判**：多行同时跨页时，逐行 `can_append` 各自都能过，
            # 合计需求却可能超过可用页
            if sum(blocks.needed_append(s, n) for s in seqs) > blocks.available_slots:
                self.stats["memory_drains"] += 1
                return False
        else:
            for s in seqs:
                if not kv.can_append(s, n):
                    self.stats["memory_drains"] += 1
                    return False
        before = [len(s.block_table) for s in seqs]
        for s in seqs:
            kv.may_append(s, n)
        self._tables_dirty = self._tables_dirty or (
            [len(s.block_table) for s in seqs] != before
        )
        self._lookahead_streak += 1
        return True

    # ---------------------------------------------------------------- #
    # 提交 / 消费
    # ---------------------------------------------------------------- #

    def _submit(self, sched: ScheduledBatch) -> DecodeTicket:
        n_p, n_d = self.core.executor.run(sched)
        self.stats["submitted"] += 1
        return DecodeTicket(
            sched=sched,
            n_p=n_p,
            n_d=n_d,
            tokens_dev=self.core.executor.last_decode_dev(),
            advances=None,
        )

    def _submit_ahead(self, prev: DecodeTicket) -> DecodeTicket:
        """用 prev 的设备侧采样结果，直接提交下一拍。"""
        seqs = prev.seqs
        tokens, advances = self._next_input(prev)
        # 必须放**真实 Sequence**：`to_wire()` 按 Sequence 取 last_token/draft_ids
        # 并断言 len(token_ids)==num_cached+1；而且下一拍 `update` 要用它 append。
        wire = ScheduledBatch(decodes=list(seqs), prefills=[])
        # radix 后端内部 remap 时，`_sync_block_tables` 的快速比对（只看首/末/长度）
        # 可能判成「没变」而跳过 H2D → 保守起见每次都刷页表（一次小 H2D，值这个确定性）
        n_p, n_d = self.core.executor.run_decode_advance(
            wire,
            tokens,
            advances,
            refresh_tables=self._tables_dirty
            or self.core.config.prefix_backend == "radix",
        )
        self._tables_dirty = False
        self.stats["submitted"] += 1
        return DecodeTicket(
            sched=wire,
            n_p=n_p,
            n_d=n_d,
            tokens_dev=self.core.executor.last_decode_dev(),
            advances=advances,
        )

    def _next_input(
        self, prev: DecodeTicket
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """下一拍的 (输入 token, 位置推进量)。

        普通 decode：token = 上一拍采样结果，advance 全 1。
        投机：两者都由设备侧接受裁决给出（步骤 B），此处直接取。
        """
        if self.core.config.spec_tokens > 0:
            # 投机拍产出的是 _SpecSamples，`tokens_dev` 本来就是 None，
            # 所以这一支必须**先**判断，不能落到下面的 None 检查。
            nxt, adv = self.core.executor.last_accept()
            if nxt is None or adv is None:
                raise RuntimeError("zos: spec decode without device-side accept")
            return nxt, adv
        tokens = prev.tokens_dev
        if tokens is None:
            raise RuntimeError("zos: previous round has no device tokens")
        ones = torch.ones(
            tokens.numel(),
            dtype=torch.int32,
            device=tokens.device,
        )
        return tokens, ones



    def _consume(self, prev: DecodeTicket, ahead: DecodeTicket | None) -> list:
        core = self.core
        seqs = prev.seqs

        # 先回读，再判断"是否结束"——因为要判断的是**本拍产出的 token**。
        # 用 `s.last_token` 是错的：那是本拍的**输入** token（CPU 状态落后一拍），
        # 拿它比 EOS 会漏 fence，在途那拍就会写进马上被释放的页。
        p_ids, d_ids, d_lens, d_prop = core.executor.read_samples(prev.n_p, prev.n_d)

        # 真实 Scheduler 里 eos 可能是 int（不是 frozenset），`tid in eos` 会 TypeError
        eos = core.scheduler.eos
        eos_set = eos if isinstance(eos, (set, frozenset)) else {eos}
        # 投机一拍产出 1+j 个 token（d_ids 是扁平的），普通 decode 一行一个。
        # 结束判断必须按**每行全部产出**：① 任一 token 命中 EOS；② 本拍产出后
        # 会达到 max_tokens（一次验证可能跨过上限，用 +1 会漏 fence）。
        rows: list[tuple] = []
        if d_lens:
            off = 0
            for n_i in d_lens:
                rows.append((d_ids[off : off + n_i], n_i))
                off += n_i
        else:
            rows = [(d_ids[i : i + 1], 1) for i in range(len(seqs))]
        will_finish = any(
            (not s.ignore_eos and any(t in eos_set for t in toks))
            or s.num_completion_tokens + n_i >= s.max_tokens
            for s, (toks, n_i) in zip(seqs, rows)
        )
        may_remap = (
            core.config.prefix_backend == "radix"
            and any(
                (s.num_cached_tokens + 1) // core.config.block_size
                > s.num_cached_tokens // core.config.block_size
                for s in seqs
            )
        )
        if ahead is not None and (will_finish or may_remap):
            # 把在途那拍等完：随后 update 会释放页 / radix 会换页，
            # 而在途计算还在往这些页里写
            core.executor.sync_samples()
            if will_finish:
                self.stats["eos_fences"] += 1
            if may_remap:
                self.stats["prefix_fences"] += 1
                self._tables_dirty = True

        # 提前提交那拍的 batch 没有经过 scheduler.schedule()，`num_scheduled_tokens`
        # 还是上一拍 update 末尾清成的 0；不重置就会 `num_cached_tokens += 0`
        # 而 token 却 append 了（下一拍 to_wire 断言 fail）。普通 decode 恒为 1。
        for s in seqs:
            s.num_scheduled_tokens = 1
        return core.scheduler.update(prev.sched, p_ids, d_ids, d_lens, d_prop)
