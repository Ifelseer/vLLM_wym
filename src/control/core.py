from __future__ import annotations

import gc

import torch

from ..config import EngineConfig
from ..data import EngineCoreOutput, KVManager, ScheduledBatch, Sequence
from .executor import create_executor
from .scheduler import Scheduler


# ==============================================================================
# EngineCore：推理引擎的核心循环。
#
# 位于 Engine 与 Executor 之间，负责：
#   驱动每一步的 调度 → 执行 → 采样 → 后处理 流程。
#   管理 KV cache 的生命周期（初始化、重建）。
#   监控调度器空闲状态，防止 KV 耗尽导致的静默死循环。
#
# EngineCore 只关心拿到 ScheduledBatch，交给 Executor 跑完，收回采样结果。
# ==============================================================================


class EngineCore:

    # 调度器连续返回空的最大容忍步数。
    # 若仍有未完成请求却持续无调度产出，大概率是 KV 池耗尽，直接报错。
    _MAX_IDLE_STEPS = 1024

    def __init__(
        self,
        config: EngineConfig,
        *,
        eos_token_id: int,
        device: torch.device,
        dtype: torch.dtype,
        executor=None,
    ) -> None:
        self.config = config
        self.eos_token_id = eos_token_id

        # executor 可由外部注入（torchrun 场景下，rank0 的 MultiprocExecutor
        # 已在进程启动阶段创建好）。未注入时，由 create_executor 按配置构建。
        self.executor = executor or create_executor(config, device, dtype)

        self._idle_steps = 0
        self._init_kv()

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #

    def _build_scheduler(self) -> Scheduler:
        """
            构建调度器。
            调度器需要持有 KVManager 引用，用于在调度时查询剩余可用块数。
        """
        return Scheduler(
            self.kv,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            eos_token_id=self.eos_token_id,
            mix_prefill_decode=self.config.mix_prefill_decode,
        )

    def _init_kv(self) -> None:
        """
            初始化 KV cache 并重建调度器。
            KV 池由 Executor 分配（涉及显存分配与块表构建），
            调度器依赖 KV 池的块数信息，因此两者需要一起初始化。
        """
        self.kv: KVManager = self.executor.init_kv()
        self.scheduler = self._build_scheduler()
        self.last_num_cached_tokens = 0

    # ------------------------------------------------------------------ #
    # 请求管理
    # ------------------------------------------------------------------ #

    def add_request(self, seq: Sequence) -> None:
        """将一条已编码的请求加入调度队列。"""
        self.scheduler.add(seq)

    def has_unfinished(self) -> bool:
        """是否还有未完成的请求（等待中或正在生成）。"""
        return not self.scheduler.is_finished()

    # ------------------------------------------------------------------ #
    # 核心循环
    # ------------------------------------------------------------------ #

    def step(self) -> tuple[list[EngineCoreOutput], int]:
        """
            执行一步完整推理：调度 → 前向 → 采样 → 后处理。

            Returns:
                (outputs, metric)
                - outputs：本步有增量产出的请求列表（新 token / 完成状态）。
                - metric：本步处理的 token 总数（prefill tokens + decode 请求数），
                        用于上层统计吞吐。
        """
        # ---- 调度 ----
        sched: ScheduledBatch = self.scheduler.schedule()

        if sched.is_empty():
            # 调度器没有产出，但仍有未完成请求。
            # 这通常意味着 KV 池已满、无法为新请求或续写请求分配块。
            # 连续空闲超过阈值后主动报错，避免静默死循环。
            if not self.scheduler.is_finished():
                self._idle_steps += 1
                if self._idle_steps >= self._MAX_IDLE_STEPS:
                    raise RuntimeError(
                        "scheduler returned empty for too long while requests remain "
                        f"(idle_steps={self._idle_steps}). Likely KV pool exhaustion "
                        f"(free_blocks={self.kv.num_free})."
                    )
            return [], 0

        self._idle_steps = 0

        # ---- 执行 ----
        # run 返回 (n_prefill, n_decode)，即本步实际跑的请求数。
        n_p, n_d = self.executor.run(sched)
        self.last_num_cached_tokens = self.kv.last_num_cached_tokens

        # 本步处理的 token 数：
        #   prefill 按实际调度的 token 数计（可能命中前缀缓存后少于原始长度）
        #   decode 每个请求只算 1 个 token
        prefill_toks = sum(s.num_scheduled_tokens for s in sched.prefills)
        metric = prefill_toks + len(sched.decodes)

        # ---- 采样与后处理 ----
        # 从执行器读回采样结果（prefill 和 decode 分开返回）。
        p_ids, d_ids = self.executor.read_samples(n_p, n_d)

        # 调度器根据采样结果更新各序列状态：
        #   追加新 token、判断是否结束、回收已完成序列的 KV 块。
        outputs = self.scheduler.update(sched, p_ids, d_ids)
        return outputs, metric

    # ------------------------------------------------------------------ #
    # 资源释放
    # ------------------------------------------------------------------ #

    def destroy(self) -> None:
        """
            释放引擎核心资源。
            先销毁执行器（释放模型权重与 KV 显存），
            再断开对 KV 和调度器的引用，最后触发 GC 与 CUDA 缓存回收。
        """
        self.executor.destroy()
        self.kv = None
        self.scheduler = None
        gc.collect()
        torch.cuda.empty_cache()