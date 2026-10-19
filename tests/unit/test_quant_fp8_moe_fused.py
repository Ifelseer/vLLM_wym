"""FP8 MoE 3D 融合权重的加载：权重与 scale 必须按同一段序、同一分片切。

HF 融合 checkpoint 的 gate_up 段序是 ``[gate; up]``，内部（cutlass）约定为
``[up; gate]``。权重与 per-block scale 是两张独立张量，**两者都要重排**；
只重排权重会让 scale 与行块错位（旧实现即如此，反量化误差 ~0.95）。

TP>1 时还有第二个陷阱：重排后的 scale 行是 ``[up(全量) | gate(全量)]``，
本地分片必须**分别**从两段取（``up[rank]`` + ``gate[rank]``）。若按连续窗口
切 ``2*local_go`` 行，rank0 会拿到 ``[up0, up1]``（跨段）——这个错误在 tp=1
下不会出现，因此本测试对 tp=1 / tp=2 的两个 rank 都做参数化。
"""

from __future__ import annotations

import pytest
import torch

from src.moe.experts import Experts
from src.moe.parallel import MoEParallelConfig
from src.quant.config import QuantConfig
from src.quant.kernels.fp8_gemm import dequantize_fp8_per_block, quantize_fp8_per_block

_E, _H, _I = 2, 256, 256


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


def _build_fused(seed: int = 0):
    """HF 融合 gate_up 的 fp8 权重 + per-block scale，段序 ``[gate; up]``。

    两段量级差 20 倍，段内两个 128 行 block 再差 10 倍——同时暴露
    「段序错」与「TP 分片窗口跨段」两类错误。
    """
    torch.manual_seed(seed)
    gate = torch.randn(_E, _I, _H) * 0.1
    up = torch.randn(_E, _I, _H) * 2.0
    gate[:, 128:] *= 0.1
    up[:, 128:] *= 0.1
    w13_hf = torch.cat([gate, up], dim=1)

    w13q = torch.empty(_E, 2 * _I, _H, dtype=torch.float8_e4m3fn)
    w13s = torch.empty(_E, 2 * _I // 128, _H // 128)
    for e in range(_E):
        q, s = quantize_fp8_per_block(w13_hf[e])
        w13q[e].copy_(q)
        w13s[e].copy_(s)
    return gate, up, w13q, w13s


def _experts(tp_size: int, tp_rank: int) -> Experts:
    quant = QuantConfig(
        method="fp8", weight_scale="per_block_128x128", act_scale="dynamic"
    )
    parallel = MoEParallelConfig(
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=1,
        ep_rank=0,
        use_ep=False,
        fused_backend="triton",
    )
    return Experts(_E, _H, _I, parallel, quant=quant)


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1)])
def test_fused_gate_up_scale_follows_weight_order(tp_size: int, tp_rank: int) -> None:
    gate, up, w13q, w13s = _build_fused()
    experts = _experts(tp_size, tp_rank)
    experts.load_fused_w13(w13q)
    experts.load_fused_w13_scale(w13s)

    shard = _I // tp_size
    start = tp_rank * shard
    for e in range(_E):
        deq = dequantize_fp8_per_block(
            experts.w13_weight.data[e], experts.w13_scale[e], torch.float32
        )
        # 必须**分段**归一化：up 与 gate 相差 20 倍，用全局 max 会把 gate 段的
        # 2~10 倍错位稀释到 5% 以内，测试就变恒过了（段序 bug 正是这样漏掉的）
        e_up = _rel(deq[:shard], up[e, start : start + shard])
        e_gate = _rel(deq[shard:], gate[e, start : start + shard])
        assert e_up < 0.08, (
            f"tp={tp_size} rank={tp_rank} expert {e}: up segment rel={e_up:.4f}"
        )
        assert e_gate < 0.08, (
            f"tp={tp_size} rank={tp_rank} expert {e}: gate segment rel="
            f"{e_gate:.4f}（scale 段序/分片错位）"
        )


def test_unreordered_scale_layout_is_detectable() -> None:
    """反向断言：直接用 checkpoint 原序的 scale（旧 bug 的行为）必须显著错位。

    若这个断言失败，说明测试的数据构造无法区分段序，前面的正向断言也就
    失去了意义（防止「测试恒过」）。
    """
    gate, up, w13q, w13s = _build_fused()
    experts = _experts(1, 0)
    experts.load_fused_w13(w13q)
    # 故意不重排：直接按 HF 原序写 scale
    experts.w13_scale.copy_(w13s)

    e0 = 0
    expected = torch.cat([up[e0], gate[e0]], dim=0)
    wrong = dequantize_fp8_per_block(
        experts.w13_weight.data[e0], experts.w13_scale[e0], torch.float32
    )
    assert _rel(wrong, expected) > 0.5, (
        "test setup cannot distinguish the un-reordered scale layout"
    )
