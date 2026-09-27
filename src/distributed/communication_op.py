from __future__ import annotations

import torch
import torch.distributed as dist

from .parallel_state import Group, get_tp_group, get_world_group

# ==============================================================================
# NCCL 集合通信与 PP 点对点。
#
# Linear / vocab 走 tensor_model_parallel_*（TP 组）。
# ModelRunner 图外 PP 走 send_tensors / recv_tensors。
# size==1 直接返回。
# ==============================================================================


def _resolve(group: Group | None) -> Group:
    return get_world_group() if group is None else group


def all_gather(
    input: torch.Tensor, dim: int = -1, group: Group | None = None
) -> torch.Tensor:
    """沿 dim 拼接各 rank 分片。NCCL 只在 dim0 拼接，其它维先按 rank 堆再 permute。"""
    group = _resolve(group)
    if group.size == 1:
        return input
    if dim < 0:
        dim += input.dim()
    src = input.contiguous()
    out = torch.empty(
        (src.shape[0] * group.size, *src.shape[1:]),
        dtype=src.dtype,
        device=src.device,
    )
    dist.all_gather_single(out, src, group=group.device_group)
    if dim == 0:
        return out
    return (
        out.view(group.size, *src.shape)
        .movedim(0, dim)
        .reshape(src.shape[:dim] + (src.shape[dim] * group.size,) + src.shape[dim + 1 :])
    )


def all_reduce(
    input: torch.Tensor, op=dist.ReduceOp.SUM, group: Group | None = None
) -> torch.Tensor:
    group = _resolve(group)
    if group.size == 1:
        return input
    dist.all_reduce(input, op, group=group.device_group)
    return input


def reduce_scatter(
    input: torch.Tensor,
    dim: int = -1,
    op=dist.ReduceOp.SUM,
    group: Group | None = None,
) -> torch.Tensor:
    group = _resolve(group)
    if group.size == 1:
        return input
    if dim < 0:
        dim += input.dim()
    src = input.movedim(dim, 0).contiguous()
    assert src.shape[0] % group.size == 0
    out = torch.empty(
        (src.shape[0] // group.size,) + src.shape[1:],
        dtype=src.dtype,
        device=src.device,
    )
    dist.reduce_scatter_single(out, src, op=op, group=group.device_group)
    return out if dim == 0 else out.movedim(0, dim).contiguous()


def tensor_model_parallel_all_reduce(
    input: torch.Tensor, op=dist.ReduceOp.SUM
) -> torch.Tensor:
    return all_reduce(input, op, group=get_tp_group())


def tensor_model_parallel_all_gather(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return all_gather(input, dim=dim, group=get_tp_group())


def tensor_model_parallel_reduce_scatter(
    input: torch.Tensor, dim: int = -1, op=dist.ReduceOp.SUM
) -> torch.Tensor:
    return reduce_scatter(input, dim=dim, op=op, group=get_tp_group())


def send_tensors(
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor],
    dst: int,
    group: Group | None = None,
) -> None:
    """阻塞发送。dst 是组内 rank（PP 用 next_group_rank），不是 WORLD rank。"""
    for work in _p2p(dist.isend, tensors, dst, group):
        work.wait()


def recv_tensors(
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor],
    src: int,
    group: Group | None = None,
) -> None:
    """阻塞接收。src 是组内 rank（PP 用 prev_group_rank）。"""
    for work in _p2p(dist.irecv, tensors, src, group):
        work.wait()


def _p2p(
    op,
    tensors: tuple[torch.Tensor, ...] | list[torch.Tensor],
    peer: int,
    group: Group | None,
) -> list:
    group = _resolve(group)
    if group.size == 1:
        return []
    if not 0 <= peer < group.size:
        raise ValueError(f"peer={peer} not in [0, {group.size})")
    ops = [
        dist.P2POp(op, t, group=group.device_group, group_peer=peer)
        for t in tensors
        if t.numel() > 0
    ]
    return list(dist.batch_isend_irecv(ops)) if ops else []
