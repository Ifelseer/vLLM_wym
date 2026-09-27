from __future__ import annotations

import datetime
import logging
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

# ==============================================================================
# 单 Engine 内的 TP × PP 进程组。
#
# Worker.init_environment 调 init_distributed_environment + initialize_model_parallel。
# global_rank = pp_rank * tp_size + tp_rank。
# DP 不在这里建组（MPClient 每个副本各自一套 WORLD）。
# ==============================================================================


class Group:
    """一组 rank：device_group 走 NCCL，cpu_group 走 Gloo。size==1 不建组。"""

    def __init__(
        self,
        ranks: list[int],
        rank: int,
        device_group: ProcessGroup | None = None,
        cpu_group: ProcessGroup | None = None,
    ):
        self.ranks = ranks
        self.rank = rank
        self.size = len(ranks)
        self.group_rank = ranks.index(rank)
        self._owns_device_group = False
        self._owns_cpu_group = False
        if device_group is not None:
            self.device_group = device_group
        elif self.size > 1:
            self.device_group = dist.new_group(ranks=ranks)
            self._owns_device_group = True
        else:
            self.device_group = None
        if cpu_group is not None:
            self.cpu_group = cpu_group
        elif self.size > 1:
            self.cpu_group = dist.new_group(ranks=ranks, backend="gloo")
            self._owns_cpu_group = True
        else:
            self.cpu_group = None

    @property
    def is_first_rank(self) -> bool:
        return self.group_rank == 0

    @property
    def is_last_rank(self) -> bool:
        return self.group_rank == self.size - 1

    @property
    def prev_group_rank(self) -> int:
        """PP 上游，给 NCCL group_peer=。线性流水线，第一段没有前驱。"""
        if self.group_rank <= 0:
            raise RuntimeError("first PP stage has no previous peer")
        return self.group_rank - 1

    @property
    def next_group_rank(self) -> int:
        """PP 下游，给 NCCL group_peer=。"""
        if self.group_rank >= self.size - 1:
            raise RuntimeError("last PP stage has no next peer")
        return self.group_rank + 1

    @property
    def prev_rank(self) -> int:
        """上游的 WORLD rank，给 Gloo send/recv。NCCL 不要用这个。"""
        return self.ranks[self.prev_group_rank]

    @property
    def next_rank(self) -> int:
        return self.ranks[self.next_group_rank]

    def barrier(self) -> None:
        if self.size > 1 and self.cpu_group is not None:
            dist.barrier(self.cpu_group)

    def destroy(self) -> None:
        if self._owns_device_group and self.device_group is not None:
            dist.destroy_process_group(self.device_group)
        if self._owns_cpu_group and self.cpu_group is not None:
            dist.destroy_process_group(self.cpu_group)
        self.device_group = None
        self.cpu_group = None


WORLD: Group | None = None
TP: Group | None = None
PP: Group | None = None
_ENABLE_EP = False


def set_expert_parallel(enabled: bool) -> None:
    """``--ep``：MoE 在现有 TP ranks 上切 expert，不新建 NCCL 组。"""
    global _ENABLE_EP
    _ENABLE_EP = bool(enabled)


def expert_parallel_enabled() -> bool:
    return _ENABLE_EP


def get_world_group() -> Group:
    assert WORLD is not None, "Distributed process group is not initialized."
    return WORLD


def get_tp_group() -> Group:
    assert TP is not None, "Tensor parallel process group is not initialized."
    return TP


def get_pp_group() -> Group:
    assert PP is not None, "Pipeline parallel process group is not initialized."
    return PP


def get_ep_group() -> Group:
    """EP 复用 TP；未开 ``--ep`` 时返回 size=1 的占位组。"""
    tp = get_tp_group()
    if _ENABLE_EP:
        return tp
    return Group(ranks=[tp.rank], rank=tp.rank)


def initialize_model_parallel(
    tp_size: int,
    pp_size: int,
    tp_rank: int,
    pp_rank: int,
) -> None:
    """
        TP：同一 pp_rank 上连续 tp_size 个 global rank。
        PP：同一 tp_rank 上每隔 tp_size 取一个。
        pp==1 时 TP 复用 WORLD 的 NCCL 组，少建一次 clique。
    """
    global TP, PP
    if not (0 <= tp_rank < tp_size and 0 <= pp_rank < pp_size):
        raise ValueError(
            f"invalid ranks: tp_rank={tp_rank}/{tp_size}, pp_rank={pp_rank}/{pp_size}"
        )
    my_rank = pp_rank * tp_size + tp_rank
    tp_ranks = [pp_rank * tp_size + k for k in range(tp_size)]
    pp_ranks = [k * tp_size + tp_rank for k in range(pp_size)]
    if WORLD is not None and tp_ranks == WORLD.ranks:
        TP = Group(
            ranks=tp_ranks,
            rank=my_rank,
            device_group=WORLD.device_group,
            cpu_group=WORLD.cpu_group,
        )
    else:
        TP = Group(ranks=tp_ranks, rank=my_rank)
    PP = Group(ranks=pp_ranks, rank=my_rank)


def destroy_model_parallel() -> None:
    global TP, PP
    set_expert_parallel(False)
    if TP is not None:
        TP.destroy()
    if PP is not None:
        PP.destroy()
    TP = None
    PP = None


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    local_rank: int = -1,
    backend: str = "nccl",
    init_method: str = "env://",
    timeout: int = 600,
) -> None:
    global WORLD
    if dist.is_initialized():
        logger.info("Distributed process group is already initialized.")
        return
    device_id = local_rank if local_rank >= 0 else rank
    if backend == "nccl":
        torch.cuda.set_device(device_id)
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=timeout),
        device_id=torch.device("cuda", device_id) if backend == "nccl" else None,
    )
    default_device_group: ProcessGroup | None = None
    if backend == "nccl":
        from torch.distributed.distributed_c10d import _get_default_group

        default_device_group = _get_default_group()
    WORLD = Group(
        ranks=list(range(world_size)), rank=rank, device_group=default_device_group
    )


def destroy_distributed_environment() -> None:
    global WORLD
    if WORLD is not None:
        WORLD.destroy()
    WORLD = None
    if dist.is_initialized():
        dist.destroy_process_group()


def is_initialized() -> bool:
    return dist.is_initialized()


@contextmanager
def graph_capture(device: torch.device):
    """录 CUDA Graph 前后对齐 TP/PP，避免图内 all-reduce 各 rank 步调不一致。PP send/recv 仍在图外。"""
    torch.cuda.synchronize(device)
    get_tp_group().barrier()
    get_pp_group().barrier()
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        get_tp_group().barrier()
        get_pp_group().barrier()


def set_nccl_env(ifname: str = "", *, disable_ib: bool = False) -> None:
    def _set(name: str, value: str) -> None:
        os.environ.setdefault(name, value)

    if ifname:
        _set("NCCL_SOCKET_IFNAME", ifname)
    if disable_ib:
        _set("NCCL_IB_DISABLE", "1")
    else:
        _set("NCCL_IB_GID_INDEX", "3")
        _set("NCCL_NET_GDR_LEVEL", "5")
    _set("NCCL_DEBUG", "WARN")
    _set("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800")
