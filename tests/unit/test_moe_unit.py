"""CPU-only MoE helpers."""

from __future__ import annotations

import torch

from src.moe import (
    MoEParallelConfig,
    cutlass_moe_available,
    local_expert_range,
    make_expert_map,
    resolve_moe_backend,
    route_experts,
    triton_moe_available,
)
from src.config import EngineConfig
from tests.helpers import MOE_MODEL, require_model


def test_route_experts_norm():
    torch.manual_seed(0)
    w, ids = route_experts(torch.randn(4, 8), top_k=2, norm_topk_prob=True)
    assert w.shape == (4, 2) and ids.shape == (4, 2)
    assert torch.allclose(w.sum(-1), torch.ones(4), atol=1e-5)


def test_expert_map_linear():
    m = make_expert_map(8, 2, 1, device=torch.device("cpu"))
    assert m.tolist() == [-1, -1, -1, -1, 0, 1, 2, 3]
    assert local_expert_range(8, 2, 1) == (4, 8)


def test_parallel_ep_collapses_tp():
    cfg = MoEParallelConfig.from_engine(
        tp_size=4, tp_rank=2, enable_ep=True, fused_backend="triton"
    )
    assert cfg.use_ep and cfg.tp_size == 1 and cfg.ep_size == 4 and cfg.ep_rank == 2


def test_parallel_moe_tp_keeps_shard():
    cfg = MoEParallelConfig.from_engine(
        tp_size=2, tp_rank=1, enable_ep=False, fused_backend="triton"
    )
    assert not cfg.use_ep and cfg.tp_size == 2 and cfg.ep_size == 1


def test_resolve_backend_triton():
    assert resolve_moe_backend("triton") == "triton"
    assert isinstance(triton_moe_available(), bool)
    assert isinstance(cutlass_moe_available(), bool)


@require_model(MOE_MODEL)
def test_engine_config_moe_allows_cudagraph(tmp_path=None):
    c = EngineConfig(
        model=str(MOE_MODEL),
        enforce_eager=False,
        moe_backend="triton",
        num_kvcache_blocks=8,
        context_len=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.5,
        prefix_backend="none",
    )
    assert c.is_moe and c.enforce_eager is False


@require_model(MOE_MODEL)
def test_engine_config_moe_force_eager_env(monkeypatch):
    monkeypatch.setenv("VLLM_MOE_FORCE_EAGER", "1")
    c = EngineConfig(
        model=str(MOE_MODEL),
        enforce_eager=False,
        moe_backend="triton",
        num_kvcache_blocks=8,
        context_len=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.5,
        prefix_backend="none",
    )
    assert c.is_moe and c.enforce_eager is True
