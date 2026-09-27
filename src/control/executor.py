from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod

import msgspec
import torch
import torch.distributed as dist

from ..compute.worker import Worker
from ..config import EngineConfig
from ..data import KVManager, ScheduledBatch, SchedulerOutput
from ..distributed import ControlChannel, get_world_group

# ==============================================================================
# Executor：EngineCore 与 Worker 之间的执行桥。
#
# EngineCore.step 调用 run / read_samples，本模块负责将这两个调用
# 路由到实际的 Worker（持有模型权重、执行前向和采样的进程）。
#
# 分叉逻辑只看 tp × pp，不涉及 DP：
#   tp=1, pp=1 → UniprocExecutor：同进程直接调用，无通信开销。
#   否则       → MultiprocExecutor：rank0 兼做 driver 与本地 Worker，
#                  通过 Gloo 将 SchedulerOutput 广播给所有 rank，
#                  各 rank 独立执行前向，采样结果回传 rank0。
# ==============================================================================


def _sampler_rank(tp_size: int, pp_size: int) -> int:
    """
        采样发生在全局哪个 rank。

        只有最后一个 PP stage 才有完整的 logits，
        采样固定落在末 PP stage 的 tp0 上。
    """
    return (pp_size - 1) * tp_size


def _terminate_procs(procs: list[mp.Process]) -> None:
    """
        优雅关闭子进程。

        先发 SIGTERM 让子进程自行清理（释放 CUDA 上下文、关闭 NCCL 通信器等），
        等待 3 秒后若仍存活则 SIGKILL 强杀，
        避免子进程卡在 CUDA / NCCL 阻塞调用上导致主进程无法退出。
    """
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=3)
    for p in procs:
        if p.is_alive():
            p.kill()
    for p in procs:
        p.join(timeout=5)
    procs.clear()


# ------------------------------------------------------------------ #
# 控制消息
# ------------------------------------------------------------------ #


class ControlCommand(msgspec.Struct):
    """
        driver（rank0）通过 Gloo 广播给所有 Worker 的控制消息。
        用 msgspec 序列化，走 ControlChannel（CPU 后端）
    """

    op: str                              # "execute" | "reset_kv" | "shutdown"
    sched: SchedulerOutput | None = None  # execute 时携带的调度数据
    prefix_backend: str | None = None     # reset_kv 时可选切换缓存后端


# ------------------------------------------------------------------ #
# Executor 抽象基类
# ------------------------------------------------------------------ #


class Executor(ABC):
    """
        EngineCore 调用 Worker 的统一接口。

        EngineCore 只依赖这组方法，不感知底层是单进程直接调用还是多进程广播。
    """

    def __init__(
        self, config: EngineConfig, device: torch.device, dtype: torch.dtype
    ) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
        self.sampler_rank = _sampler_rank(config.tp_size, config.pp_size)
        self.kv: KVManager | None = None

    def _kv_manager(self, num_blocks: int) -> KVManager:
        """根据 Worker 报告的可用块数，构造门面侧的 KVManager。"""
        return KVManager(
            num_blocks=num_blocks,
            block_size=self.config.block_size,
            prefix_backend=self.config.prefix_backend,
        )

    @abstractmethod
    def init_kv(self) -> KVManager: ...

    @abstractmethod
    def reset_kv(self, prefix_backend: str | None = None) -> KVManager: ...

    @abstractmethod
    def run(self, sched: ScheduledBatch) -> tuple[int, int]: ...

    @abstractmethod
    def read_samples(self, n_p: int, n_d: int) -> tuple[list[int], list[int]]: ...

    @abstractmethod
    def destroy(self) -> None: ...

    def collective_rpc(self, op: str, **payload) -> object:
        """向所有 Worker 广播控制命令。基类默认拒绝未知操作。"""
        raise ValueError(f"unknown control op: {op}")

    def shutdown(self) -> None:
        """通知所有 Worker 退出。单进程实现无需操作。"""
        return None


# ------------------------------------------------------------------ #
# 子进程入口与命令循环
# ------------------------------------------------------------------ #


def _spawn_worker(
    config: EngineConfig, rank: int, tp_rank: int, pp_rank: int
) -> None:
    """
        子进程入口（rank ≥ 1）。

        初始化 CUDA 环境与模型权重，完成分布式握手后进入命令循环，
        等待 driver 广播指令。进程退出时清理 CUDA 上下文。
    """
    worker = Worker(config, rank, tp_rank, pp_rank)
    try:
        worker.init_environment()
        worker.initialize()
        dist.barrier()
        worker_loop(worker)
    finally:
        worker.destroy_environment()


def worker_loop(worker: Worker) -> None:
    """
        非 driver rank（rank ≥ 1）的主循环。

        不断等待 driver 通过 Gloo 广播的 ControlCommand：
        "execute"：执行前向，若本 rank 是采样 rank，
                    则读取采样结果并送回 driver（rank0）。
        "reset_kv"：重建 KV cache。
        "shutdown"：退出循环，进程结束。

        注意：采样结果通过 Gloo送回。
        这意味着 driver 侧的 recv 是阻塞的，无法与下一次 schedule 重叠。
    """
    channel = ControlChannel(get_world_group().cpu_group)
    enc = msgspec.msgpack.Encoder()
    dec = msgspec.msgpack.Decoder(ControlCommand)
    sampler_rank = _sampler_rank(worker.tp_size, worker.pp_size)

    while True:
        # 阻塞等待 driver 广播的下一条命令。
        cmd = dec.decode(channel.broadcast(None, src=0))

        if cmd.op == "shutdown":
            break

        if cmd.op == "reset_kv":
            worker.reset_kv(cmd.prefix_backend)
            continue

        if cmd.op != "execute":
            raise ValueError(f"unknown control op: {cmd.op}")

        # 执行前向。所有 rank 都参与（TP 需要集合通信，PP 需要逐段执行）。
        counts = worker.execute_model(cmd.sched)

        # 只有采样 rank 需要读结果并回传。
        if worker.rank != sampler_rank:
            continue

        n_p, n_d = counts
        channel.send(enc.encode([n_p, n_d]), dst=0)
        p_ids, d_ids = worker.read_samples(n_p, n_d)
        channel.send(enc.encode([p_ids, d_ids]), dst=0)


# ------------------------------------------------------------------ #
# 单进程执行器（tp=1, pp=1）
# ------------------------------------------------------------------ #


class UniprocExecutor(Executor):
    """
        单卡执行器：Worker 与 EngineCore 在同一进程内所有调用直接转发。
    """

    def __init__(
        self, config: EngineConfig, device: torch.device, dtype: torch.dtype
    ) -> None:
        super().__init__(config, device, dtype)
        self.worker = Worker(config, rank=0, tp_rank=0, pp_rank=0)
        self.worker.init_environment()
        # initialize() 延迟到首次 init_kv 时调用，
        # 避免在 Engine.__init__ 阶段过早分配模型显存。
        self._ready = False

    def init_kv(self) -> KVManager:
        """首次调用时初始化模型权重，之后每次调用构造 KVManager。"""
        if not self._ready:
            self.worker.initialize()
            self._ready = True
        self.kv = self._kv_manager(self.worker.get_kv_cache_size())
        return self.kv

    def reset_kv(self, prefix_backend: str | None = None) -> KVManager:
        if prefix_backend is not None:
            self.config.prefix_backend = prefix_backend
        self.worker.reset_kv(prefix_backend)
        self.kv = self._kv_manager(self.worker.get_kv_cache_size())
        return self.kv

    def run(self, sched: ScheduledBatch) -> tuple[int, int]:
        """直接调用 Worker 执行前向，返回 (n_prefill, n_decode)。"""
        counts = self.worker.execute_model(sched.to_wire())
        assert counts is not None
        return counts

    def read_samples(self, n_p: int, n_d: int) -> tuple[list[int], list[int]]:
        return self.worker.read_samples(n_p, n_d)

    def collective_rpc(self, op: str, **payload) -> object:
        """单进程下无需广播，直接本地执行。"""
        if op == "reset_kv":
            self.worker.reset_kv(payload.get("prefix_backend"))
            return None
        return super().collective_rpc(op, **payload)

    def destroy(self) -> None:
        self.kv = None
        self.worker.destroy_environment()


# ------------------------------------------------------------------ #
# 多进程执行器（tp × pp > 1）
# ------------------------------------------------------------------ #


class MultiprocExecutor(Executor):
    """
    多卡执行器：WORLD = tp × pp。

    rank0 同时承担两个角色：
      - driver：接收 EngineCore 的调用，通过 Gloo 广播调度指令。
      - 本地 Worker：执行自己那份模型前向。

    构造方式有两种：
      - worker=None（run.py 路径）：本进程负责 spawn 所有子进程。
      - worker=已初始化的Worker（torchrun 路径）：
        各进程已在外部完成初始化，这里只包装 rank0 的 Worker。
    """

    def __init__(
        self,
        config: EngineConfig,
        device: torch.device,
        dtype: torch.dtype,
        worker: Worker | None = None,
    ) -> None:
        super().__init__(config, device, dtype)

        # msgspec 编解码器，用于 Gloo 控制消息的序列化。
        self._enc = msgspec.msgpack.Encoder()
        self._dec_counts = msgspec.msgpack.Decoder(list[int])
        self._dec_sample = msgspec.msgpack.Decoder(list[list[int]])

        self._procs: list[mp.Process] = []
        # 记录子进程是否由本实例创建。
        # torchrun 路径下子进程由外部管理，destroy 时不应 terminate。
        self._owns_workers = worker is None

        if worker is None:
            worker = self._spawn_world()

        self.worker = worker
        self.num_blocks = worker.get_kv_cache_size()
        self._channel = ControlChannel(get_world_group().cpu_group)

    def _spawn_world(self) -> Worker:
        """
            拉起所有子进程并初始化 rank0。

            顺序：先启动 rank ≥ 1 的子进程，再初始化本进程的 rank0 Worker。
            这样做是因为 torch.distributed 的 FileStore 会合机制
            不要求 rank0 先监听，所有 rank 可以并行完成握手。

            若初始化过程中任何一步失败，立即清理已启动的子进程。
        """
        driver: Worker | None = None
        try:
            ctx = mp.get_context("spawn")
            tp, pp = self.config.tp_size, self.config.pp_size

            # 先启动 rank ≥ 1 的子进程。
            for pp_rank in range(pp):
                for tp_rank in range(tp):
                    rank = pp_rank * tp + tp_rank
                    if rank == 0:
                        continue
                    p = ctx.Process(
                        target=_spawn_worker,
                        args=(self.config, rank, tp_rank, pp_rank),
                        daemon=False,
                    )
                    p.start()
                    self._procs.append(p)

            # 再初始化本进程的 rank0。
            driver = Worker(self.config, rank=0, tp_rank=0, pp_rank=0)
            driver.init_environment()
            driver.initialize()

            # 所有 rank 就绪后同步，确保模型权重和通信器都已初始化。
            dist.barrier()
            return driver

        except BaseException:
            # 初始化失败：清理已启动的子进程，避免残留。
            _terminate_procs(self._procs)
            if driver is not None:
                try:
                    driver.destroy_environment()
                except Exception:
                    pass
            raise

    def init_kv(self) -> KVManager:
        """
            构造门面侧的 KVManager。
            多进程下模型已在 __init__ 阶段初始化完毕，
            这里只需根据 Worker 报告的块数构造管理对象。
        """
        self.kv = self._kv_manager(self.num_blocks)
        return self.kv

    def reset_kv(self, prefix_backend: str | None = None) -> KVManager:
        """
            重建所有 rank 的 KV cache。
            先广播 reset_kv 命令让子进程执行，再本地执行，最后重建管理对象。
        """
        if prefix_backend is not None:
            self.config.prefix_backend = prefix_backend
        self.collective_rpc("reset_kv", prefix_backend=prefix_backend)
        self.worker.reset_kv(prefix_backend)
        self.num_blocks = self.worker.get_kv_cache_size()
        self.kv = self._kv_manager(self.num_blocks)
        return self.kv

    def collective_rpc(self, op: str, **payload) -> object:
        """
            通过 Gloo 向所有 Worker 广播控制命令。
            所有 rank（包括 rank0 自身）都会在 worker_loop 或
            execute_model 中收到并执行该命令。
        """
        cmd = ControlCommand(
            op=op,
            sched=payload.get("sched"),
            prefix_backend=payload.get("prefix_backend"),
        )
        self._channel.broadcast(self._enc.encode(cmd), src=0)
        return None

    def run(self, sched: ScheduledBatch) -> tuple[int, int]:
        """
            执行一步前向。

            1. 将 ScheduledBatch 序列化为 SchedulerOutput（to_wire）。
            2. 通过 Gloo 广播给所有子进程。
            3. rank0 本地也执行前向（它同时是 Worker）。
            4. 若采样不在 rank0（PP 场景），等待采样 rank 回传计数。
        """
        wire = sched.to_wire()

        # 广播调度数据，子进程在 worker_loop 中收到并执行。
        self.collective_rpc("execute", sched=wire)

        # rank0 本地执行。
        counts = self.worker.execute_model(wire)

        if self.sampler_rank == 0:
            # 纯 TP 或 pp=1：采样就在 rank0，直接返回。
            assert counts is not None
            return counts

        # PP 场景：采样在末 stage，等待采样 rank 回传 (n_p, n_d)。
        n_p, n_d = self._dec_counts.decode(self._channel.recv(self.sampler_rank))
        return n_p, n_d

    def read_samples(self, n_p: int, n_d: int) -> tuple[list[int], list[int]]:
        """
            读取采样结果。

            采样在 rank0 时直接本地读取；
            否则等待采样 rank 通过 Gloo 送回 token id 列表。
        """
        if self.sampler_rank == 0:
            return self.worker.read_samples(n_p, n_d)
        p_ids, d_ids = self._dec_sample.decode(self._channel.recv(self.sampler_rank))
        return p_ids, d_ids

    def shutdown(self) -> None:
        """广播 shutdown 命令，通知所有子进程退出命令循环。"""
        self.collective_rpc("shutdown")

    def destroy(self) -> None:
        """
            释放执行器资源。

            若子进程由本实例创建（run.py 路径），先广播 shutdown，
            再终止子进程，最后清理本地 Worker。
            torchrun 路径下子进程由外部管理，这里只断开引用。
        """
        if self._owns_workers:
            try:
                self.shutdown()
            except Exception:
                pass
            _terminate_procs(self._procs)
            self.worker.destroy_environment()
        self.kv = None


# ------------------------------------------------------------------ #
# 工厂函数
# ------------------------------------------------------------------ #


def create_executor(
    config: EngineConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    rank0_worker: Worker | None = None,
) -> Executor:
    """
        根据 tp / pp 配置选择执行器实现。

        - tp=1, pp=1：返回 UniprocExecutor，同进程直接调用。
        - 否则：返回 MultiprocExecutor。

        rank0_worker 仅在 torchrun 多机路径下使用：
        外部已在当前进程中初始化好 rank0 的 Worker，
        直接注入，避免 MultiprocExecutor 重复创建。
    """
    if config.tp_size == 1 and config.pp_size == 1:
        return UniprocExecutor(config, device, dtype)
    return MultiprocExecutor(config, device, dtype, worker=rank0_worker)