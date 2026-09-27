"""Experts weights, EP map, and fused forward."""

from __future__ import annotations

import torch
from torch import nn

from ..distributed import divide, ensure_divisibility
from .fused import fused_moe_forward
from .parallel import MoEParallelConfig


def local_expert_range(num_experts: int, ep_size: int, ep_rank: int) -> tuple[int, int]:
    n_local = divide(num_experts, ep_size)
    start = ep_rank * n_local
    return start, start + n_local


def make_expert_map(
    num_experts: int,
    ep_size: int,
    ep_rank: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """``map[global_id] = local_id``; remote experts are ``-1``."""
    mapping = torch.full((num_experts,), -1, dtype=dtype, device=device)
    start, end = local_expert_range(num_experts, ep_size, ep_rank)
    mapping[start:end] = torch.arange(end - start, dtype=dtype, device=device)
    return mapping


def _hf_gate_up_to_cutlass(w13_hf: torch.Tensor) -> torch.Tensor:
    gate, up = w13_hf.chunk(2, dim=1)
    return torch.cat([up, gate], dim=1)


def _w13_slice(weight_name: str, intermediate_size: int) -> slice:
    if weight_name.endswith(("up_proj.weight", "w3")):
        return slice(0, intermediate_size)
    if weight_name.endswith(("gate_proj.weight", "w1")):
        return slice(intermediate_size, 2 * intermediate_size)
    raise ValueError(f"unknown gate/up weight: {weight_name}")


def _normalize_fused_weight(
    loaded: torch.Tensor,
    expected_shape: tuple[int, int, int],
    name: str,
) -> torch.Tensor:
    assert loaded.ndim == 3, f"{name} must be 3D, got {tuple(loaded.shape)}"
    if loaded.shape == expected_shape:
        return loaded
    transposed = (expected_shape[0], expected_shape[2], expected_shape[1])
    if loaded.shape == transposed:
        loaded = loaded.transpose(1, 2).contiguous()
    assert loaded.shape == expected_shape, (
        f"{name} shape {tuple(loaded.shape)} != {expected_shape}"
    )
    return loaded


class Experts(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        parallel: MoEParallelConfig,
    ) -> None:
        super().__init__()
        ensure_divisibility(intermediate_size, parallel.tp_size)
        ensure_divisibility(num_experts, parallel.ep_size)
        self.global_num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tp_size = parallel.tp_size
        self.tp_rank = parallel.tp_rank
        self.ep_size = parallel.ep_size
        self.ep_rank = parallel.ep_rank
        self.fused_backend = parallel.fused_backend
        self.expert_start, self.expert_end = local_expert_range(
            num_experts, parallel.ep_size, parallel.ep_rank
        )
        self.num_experts = self.expert_end - self.expert_start
        self.intermediate_size_per_partition = divide(intermediate_size, parallel.tp_size)
        i = self.intermediate_size_per_partition
        self.w13_weight = nn.Parameter(
            torch.empty(self.num_experts, 2 * i, hidden_size)
        )
        self.w2_weight = nn.Parameter(torch.empty(self.num_experts, hidden_size, i))
        setattr(self.w13_weight, "weight_loader", self.weight_loader_w13)
        setattr(self.w2_weight, "weight_loader", self.weight_loader_w2)
        self.register_buffer(
            "expert_map",
            make_expert_map(
                num_experts,
                parallel.ep_size,
                parallel.ep_rank,
                device=torch.device("cpu"),
            ),
            persistent=False,
        )

    def _local_id(self, expert_id: int) -> int | None:
        if expert_id < self.expert_start or expert_id >= self.expert_end:
            return None
        return expert_id - self.expert_start

    def weight_loader_w13(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        expert_id: int,
    ) -> None:
        local = self._local_id(expert_id)
        if local is None:
            return
        shard = self.intermediate_size_per_partition
        piece = loaded_weight.narrow(0, self.tp_rank * shard, shard)
        dest = param.data[local, _w13_slice(weight_name, shard)]
        assert dest.shape == piece.shape, f"{dest.shape} != {piece.shape}"
        dest.copy_(piece)

    def weight_loader_w2(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        expert_id: int,
    ) -> None:
        local = self._local_id(expert_id)
        if local is None:
            return
        shard = self.intermediate_size_per_partition
        piece = loaded_weight.narrow(1, self.tp_rank * shard, shard)
        dest = param.data[local]
        assert dest.shape == piece.shape, f"{dest.shape} != {piece.shape}"
        dest.copy_(piece)

    def load_fused_w13(self, loaded: torch.Tensor) -> None:
        loaded = _normalize_fused_weight(
            loaded,
            (self.global_num_experts, 2 * self.intermediate_size, self.hidden_size),
            "gate_up",
        )
        loaded = _hf_gate_up_to_cutlass(loaded)[self.expert_start : self.expert_end]
        shard = self.intermediate_size_per_partition
        start = self.tp_rank * shard
        u = loaded[:, start : start + shard, :]
        g = loaded[
            :,
            self.intermediate_size + start : self.intermediate_size + start + shard,
            :,
        ]
        self.w13_weight.data.copy_(torch.cat([u, g], dim=1))

    def load_fused_w2(self, loaded: torch.Tensor) -> None:
        loaded = _normalize_fused_weight(
            loaded,
            (self.global_num_experts, self.hidden_size, self.intermediate_size),
            "down",
        )[self.expert_start : self.expert_end]
        shard = self.intermediate_size_per_partition
        start = self.tp_rank * shard
        self.w2_weight.data.copy_(loaded[:, :, start : start + shard])

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if not hidden_states.is_cuda:
            raise RuntimeError("MoE fused kernels require CUDA tensors")
        mapping = self.expert_map.to(device=topk_ids.device, non_blocking=True)
        return fused_moe_forward(
            backend=self.fused_backend,
            hidden_states=hidden_states,
            topk_ids=mapping[topk_ids.long()],
            topk_weights=topk_weights,
            w13_weight=self.w13_weight,
            w2_weight=self.w2_weight,
        )
