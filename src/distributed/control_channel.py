from __future__ import annotations

import torch
import torch.distributed as dist

# ==============================================================================
# 同 Engine 内 driver ↔ worker 的 Gloo 字节通道。
#
# MultiprocExecutor / worker_loop 用来广播 SchedulerOutput、回传 token id。
# src/dst 是 WORLD rank。DP 的 ZMQ 不走这里。
# ==============================================================================


def _cpu_u8(data: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()


def _to_bytes(buf: torch.Tensor) -> bytes:
    return buf.contiguous().numpy().tobytes()


class ControlChannel:
    def __init__(self, group) -> None:
        self.group = group

    def broadcast(self, data: bytes | None, src: int = 0) -> bytes | None:
        rank = dist.get_rank(self.group)
        size = torch.tensor([len(data) if data is not None else 0], dtype=torch.long)
        dist.broadcast(size, src=src, group=self.group)
        n = int(size.item())
        if n == 0:
            return None
        if rank == src:
            dist.broadcast(_cpu_u8(data), src=src, group=self.group)
            return None
        buf = torch.empty(n, dtype=torch.uint8)
        dist.broadcast(buf, src=src, group=self.group)
        return _to_bytes(buf)

    def send(self, data: bytes, dst: int) -> None:
        dist.send(
            torch.tensor([len(data)], dtype=torch.long), dst=dst, group=self.group
        )
        if data:
            dist.send(_cpu_u8(data), dst=dst, group=self.group)

    def recv(self, src: int) -> bytes:
        size = torch.tensor([0], dtype=torch.long)
        dist.recv(size, src=src, group=self.group)
        n = int(size.item())
        if n == 0:
            return b""
        buf = torch.empty(n, dtype=torch.uint8)
        dist.recv(buf, src=src, group=self.group)
        return _to_bytes(buf)
