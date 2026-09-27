"""Fused MoE: Triton / FlashInfer cutlass + optional CUDA aux kernels.

Layout::

    block.py      SparseMoeBlock + route_experts
    experts.py    weights, EP map, Experts.forward
    parallel.py   TP/EP config + resolve_moe_backend
    fused.py      cutlass / triton dispatch
    kernels/      moe_kernels.cu + cuda.py + triton.py
"""

from __future__ import annotations

from .block import SparseMoeBlock, route_experts
from .experts import local_expert_range, make_expert_map
from .fused import clear_fused_state
from .parallel import (
    MoEParallelConfig,
    cutlass_moe_available,
    resolve_moe_backend,
    triton_moe_available,
)

__all__ = [
    "SparseMoeBlock",
    "MoEParallelConfig",
    "route_experts",
    "clear_fused_state",
    "resolve_moe_backend",
    "cutlass_moe_available",
    "triton_moe_available",
    "local_expert_range",
    "make_expert_map",
]
