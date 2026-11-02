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

from ..data import ScheduledBatch, SeqSchedule

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
        )
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
                return core._step_sync()  # 非稳定 decode：走原路径
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
            # （不等的话，下一拍可能写进已释放的页）
            core.executor.sync_samples()
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
        if self.core.config.spec_tokens > 0:
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

        # 提前一拍要覆盖到 num_cached + n（投机为 1+k，普通为 1）
        n = 1 + max(0, self.core.config.spec_tokens)
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
        wire = ScheduledBatch(
            decodes=[self._lagged_schedule(s) for s in seqs],
            prefills=[],
        )
        n_p, n_d = self.core.executor.run_decode_advance(
            wire, tokens, advances, refresh_tables=self._tables_dirty
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
        tokens = prev.tokens_dev
        if tokens is None:
            raise RuntimeError("zos: previous round has no device tokens")
        if self.core.config.spec_tokens > 0:
            nxt, adv = self.core.executor.last_accept()
            if nxt is None or adv is None:
                raise RuntimeError("zos: spec decode without device-side accept")
            return nxt, adv
        ones = torch.ones(
            tokens.numel(),
            dtype=torch.int32,
            device=tokens.device,
        )
        return tokens, ones

    @staticmethod
    def _lagged_schedule(seq) -> SeqSchedule:
        """构造下一拍的调度行。

        ZOS 下 CPU 状态落后一拍，这里**只用** `seq_id` 与 `block_table`
        （推进由设备侧 kernel 完成）；位置/长度等字段不会被读取，填滞后值即可。
        """
        return SeqSchedule(
            seq_id=seq.seq_id,
            block_table=seq.block_table,
            num_cached_tokens=seq.num_cached_tokens,
            token_ids=[seq.last_token],
        )

    def _consume(self, prev: DecodeTicket, ahead: DecodeTicket | None) -> list:
        core = self.core
        seqs = prev.seqs

        will_finish = any(
            (not s.ignore_eos and s.last_token in core.scheduler.eos)
            or s.num_completion_tokens + 1 >= s.max_tokens
            for s in seqs
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
            # 先把在途那拍等完：随后 postprocess 会释放页 / radix 会换页，
            # 而在途计算还在写这些页
            core.executor.sync_samples()
            if will_finish:
                self.stats["eos_fences"] += 1
            if may_remap:
                self.stats["prefix_fences"] += 1
                self._tables_dirty = True

        p_ids, d_ids, d_lens, d_prop = core.executor.read_samples(prev.n_p, prev.n_d)
        return core.scheduler.update(prev.sched, p_ids, d_ids, d_lens, d_prop)
