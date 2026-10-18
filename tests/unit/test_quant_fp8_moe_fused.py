"""FP8 MoE 3D 融合权重的加载：权重与 scale 必须按同一段序重排。

HF 融合 checkpoint 的 gate_up 段序是 ``[gate; up]``，内部（cutlass）约定为
``[up; gate]``。权重与 per-block scale 是两张独立张量，**两者都要重排**；
只重排权重会让 scale 与行块错位（旧实现即如此，反量化误差 ~0.95）。
"""

from __future__ import annotations

import torch

from src.moe.experts import Experts
from src.moe.parallel import MoEParallelConfig
from src.quant.config import QuantConfig
from src.quant.kernels.fp8_gemm import dequantize_fp8_per_block, quantize_fp8_per_block


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


def test_fused_gate_up_scale_follows_weight_order() -> None:
    E, H, I = 2, 256, 256
    quant = QuantConfig(
        method="fp8", weight_scale="per_block_128x128", act_scale="dynamic"
    )
    parallel = MoEParallelConfig(
        tp_size=1, tp_rank=0, ep_size=1, ep_rank=0, use_ep=False,
        fused_backend="triton",
    )
    experts = Experts(E, H, I, parallel, quant=quant)

    torch.manual_seed(0)
    # 两段量级差 20 倍：段序错位会非常明显
    gate = torch.randn(E, I, H) * 0.1
    up = torch.randn(E, I, H) * 2.0
    w13_hf = torch.cat([gate, up], dim=1)  # HF [gate; up]

    w13q = torch.empty(E, 2 * I, H, dtype=torch.float8_e4m3fn)
    w13s = torch.empty(E, 2 * I // 128, H // 128)
    for e in range(E):
        q, s = quantize_fp8_per_block(w13_hf[e])
        w13q[e].copy_(q)
        w13s[e].copy_(s)

    experts.load_fused_w13(w13q)
    experts.load_fused_w13_scale(w13s)

    expected = torch.cat([up, gate], dim=1)  # 内部 [up; gate]
    for e in range(E):
        deq = dequantize_fp8_per_block(
            experts.w13_weight.data[e], experts.w13_scale[e], torch.float32
        )
        assert _rel(deq, expected[e]) < 0.08, (
            f"expert {e}: dequantized fused weight does not match [up; gate]"
        )

    # 反向断言：直接使用 checkpoint 原序（即旧 bug 的行为）应显著错位
    e0 = 0
    wrong = dequantize_fp8_per_block(
        experts.w13_weight.data[e0], w13s[e0], torch.float32
    )
    assert _rel(wrong, expected[e0]) > 0.5, (
        "test setup cannot distinguish the un-reordered scale layout"
    )
