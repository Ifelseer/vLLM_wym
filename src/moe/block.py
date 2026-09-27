"""Sparse MoE block: gate → route → fused experts → optional shared + AR."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..compute.layers import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    SiluAndMul,
)
from ..distributed import get_tp_group, tensor_model_parallel_all_reduce
from .experts import Experts
from .parallel import MoEParallelConfig


def route_experts(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    norm_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(scores, top_k, dim=-1)
    if norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(router_logits.dtype), topk_ids


class SparseMoeBlock(nn.Module):
    def __init__(self, config, parallel: MoEParallelConfig) -> None:
        super().__init__()
        self.parallel = parallel
        hidden = config.hidden_size
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", False))
        self.use_ep = parallel.use_ep

        self.gate = ReplicatedLinear(hidden, self.num_experts, bias=False)
        self.experts = Experts(
            self.num_experts,
            hidden,
            int(config.moe_intermediate_size),
            parallel,
        )
        self._init_shared_expert(config, hidden)

    def _init_shared_expert(self, config, hidden: int) -> None:
        shared_inter = int(getattr(config, "shared_expert_intermediate_size", 0) or 0)
        if shared_inter <= 0:
            self.shared_expert_gate = None
            self.shared_expert_gate_up = None
            self.shared_expert_down = None
            self.shared_activation = None
            return
        # EP: replicate shared; add after routed AR so it is not summed ep times.
        shard_tp = 1 if self.use_ep else None
        self.shared_expert_gate = ReplicatedLinear(hidden, 1, bias=False)
        self.shared_expert_gate_up = MergedColumnParallelLinear(
            hidden, [shared_inter] * 2, bias=False, tp_size=shard_tp
        )
        self.shared_expert_down = RowParallelLinear(
            shared_inter,
            hidden,
            bias=False,
            reduce_results=False,
            tp_size=shard_tp,
            input_is_parallel=not self.use_ep,
        )
        self.shared_activation = SiluAndMul()

    def _shared_expert(self, hidden: torch.Tensor) -> torch.Tensor:
        assert self.shared_expert_gate_up is not None
        assert self.shared_expert_down is not None
        assert self.shared_activation is not None
        assert self.shared_expert_gate is not None
        shared = self.shared_expert_down(
            self.shared_activation(self.shared_expert_gate_up(hidden))
        )
        return F.sigmoid(self.shared_expert_gate(hidden)) * shared

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden = hidden_states.view(-1, orig_shape[-1])
        topk_weights, topk_ids = route_experts(
            self.gate(hidden), self.top_k, norm_topk_prob=self.norm_topk_prob
        )
        out = self.experts(hidden, topk_ids, topk_weights)
        need_ar = get_tp_group().size > 1
        if self.use_ep:
            if need_ar:
                out = tensor_model_parallel_all_reduce(out)
            if self.shared_expert_gate_up is not None:
                out = out + self._shared_expert(hidden)
        else:
            if self.shared_expert_gate_up is not None:
                out = out + self._shared_expert(hidden)
            if need_ar:
                out = tensor_model_parallel_all_reduce(out)
        return out.view(orig_shape)
