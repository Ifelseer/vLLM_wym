from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ..config import EngineConfig
from ..data import EngineCoreOutput, Sequence
from .core import EngineCore

# ==============================================================================
# EngineCoreClient：Engine 与 EngineCore 之间的调用层。
#
# Engine 不直接持有 EngineCore，而是通过这一层间接调用。
# 这样做的目的是让 DP 的分叉只发生在这里：
#   - dp=1：InprocClient，同进程直接转发，零开销。
#   - dp>1：MPClient，经 ZMQ 将请求路由到各副本子进程。
#
# TP / PP 的多进程展开不在这层处理，由 EngineCore 内部的 Executor 负责。
# ==============================================================================


class EngineCoreClient(ABC):
    """
        Engine 调用 EngineCore 的统一接口。

        无论底层是同进程转发还是 ZMQ 跨进程通信，
        Engine 侧的 add_request / step / has_unfinished / destroy
        都只依赖这四个方法，不感知具体的执行路径。
    """

    @abstractmethod
    def add_request(self, seq: Sequence) -> None: ...

    @abstractmethod
    def step(self) -> tuple[list[EngineCoreOutput], int]: ...

    @abstractmethod
    def has_unfinished(self) -> bool: ...

    @abstractmethod
    def destroy(self) -> None: ...


class InprocClient(EngineCoreClient):
    """
        单进程客户端（dp=1）。

        直接持有 EngineCore 实例，所有方法直接转发，与 Scheduler 共享同一对象。
    """

    def __init__(self, engine_core: EngineCore) -> None:
        self.engine_core = engine_core

    def add_request(self, seq: Sequence) -> None:
        self.engine_core.add_request(seq)

    def step(self) -> tuple[list[EngineCoreOutput], int]:
        return self.engine_core.step()

    def has_unfinished(self) -> bool:
        return self.engine_core.has_unfinished()

    def destroy(self) -> None:
        self.engine_core.destroy()


def make_client(
    config: EngineConfig,
    *,
    eos_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
    executor=None,
) -> EngineCoreClient:
    """
        工厂函数：根据 dp_size 选择客户端实现。

        dp=1：构造 EngineCore，包装为 InprocClient 返回。
        此时可通过 executor 参数注入已初始化好的 MultiprocExecutor

        dp>1：构造 MPClient，启动多个副本子进程。
        每个副本拥有独立的 NCCL world 和设备，不允许共享外部注入的 executor。
    """
    if config.dp_size > 1:
        from .mp_client import MPClient

        if executor is not None:
            raise ValueError(
                "dp_size>1 uses MPClient replicas; cannot inject a shared executor"
            )
        return MPClient(
            config, eos_token_id=eos_token_id, device=device, dtype=dtype
        )
    core = EngineCore(
        config, eos_token_id=eos_token_id, device=device, dtype=dtype, executor=executor
    )
    return InprocClient(core)