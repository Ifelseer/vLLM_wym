from __future__ import annotations

from bisect import bisect_left
from typing import Any, Callable

import torch

from ..distributed import graph_capture


class CUDAGraphRunner:
    def __init__(self, bs_list: list[int], device: torch.device):
        self.device = device
        self.bs_list = sorted(bs_list)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.outputs: dict[int, Any] = {}
        self._graph_pool = None

    def capture(self, body: Callable[[int], Any]) -> None:
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        for bs in sorted(self.bs_list, reverse=True):
            body(bs)  # warmup：初始化算子与 NCCL 集合通信顺序
            graph = torch.cuda.CUDAGraph()
            with graph_capture(self.device):
                with torch.cuda.graph(graph, pool=self._graph_pool):
                    self.outputs[bs] = body(bs)
            if self._graph_pool is None:
                self._graph_pool = graph.pool()
            self.graphs[bs] = graph
        torch.cuda.synchronize(self.device)

    def replay(self, bs: int) -> tuple[Any, int]:
        pbs = self._pad_bs(bs)
        self.graphs[pbs].replay()
        return self.outputs[pbs], pbs

    def _pad_bs(self, bs: int) -> int:
        return self.bs_list[bisect_left(self.bs_list, bs)]

    def destroy(self) -> None:
        self.graphs.clear()
        self.outputs.clear()
        self._graph_pool = None

    @staticmethod
    def make_bs_list(max_bs: int) -> list[int]:
        bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        bs = [b for b in bs if b <= max_bs]
        if max_bs not in bs:
            bs.append(max_bs)
        return bs
