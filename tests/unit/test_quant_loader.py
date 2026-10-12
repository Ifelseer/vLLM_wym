"""FP8 加载器与量化参数的 CPU 单元测试（无需 GPU）。

覆盖：
- Fp8Method 各 kind 的参数创建（fp8 权重 + scale 网格形状）
- 在线量化路径（BF16 → staging → finalize）的 TP 切分正确性
- 预量化路径（fp8 权重 + 独立 scale）的 TP 切分正确性
- per-block 量化/反量化数值往返
- Fp8FallbackCatcher（embed/lm_head 的 fp8 反量化配对）
- MoE Experts 的 FP8 参数与 per-expert 加载
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.quant.config import QuantConfig
from src.quant.kernels.fp8_gemm import (
    FP8_DTYPE,
    dequantize_fp8_per_block,
    quantize_fp8_per_block,
)
from src.quant.methods.fp8 import Fp8Method

FP8_MAX = 448.0


class _Layer(nn.Module):
    """承载量化参数的最小模块（替代真实并行层）。"""


def _method(
    kind: str,
    out_full: int,
    in_full: int,
    *,
    tp: int = 1,
    rank: int = 0,
    merged: list[int] | None = None,
    qkv_meta=None,
    online: bool = True,
) -> Fp8Method:
    """按 kind 的切分语义计算 rank 本地尺寸：
    column/merged_column/qkv 只切输出维；row 只切输入维；replicated 不切。
    """
    q = QuantConfig.from_name(
        "fp8", device_capability=(8, 9), quantize_at_load=online
    )
    shard_out = kind in ("column", "merged_column", "qkv")
    shard_in = kind == "row"
    return Fp8Method(
        q,
        kind=kind,
        out_size=out_full // tp if shard_out else out_full,
        in_size=in_full // tp if shard_in else in_full,
        tp_size=tp,
        tp_rank=rank,
        merged_sizes=merged,
        qkv_meta=qkv_meta,
    )


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-6)).item()


# ------------------------------------------------------------------ #
# 量化/反量化数值
# ------------------------------------------------------------------ #


class TestQuantMath:
    def test_block_roundtrip(self):
        torch.manual_seed(0)
        x = torch.randn(256, 512, dtype=torch.bfloat16)
        q, s = quantize_fp8_per_block(x)
        assert q.dtype == FP8_DTYPE and q.shape == x.shape
        assert s.shape == (2, 4) and s.dtype == torch.float32
        deq = dequantize_fp8_per_block(q, s, torch.bfloat16)
        assert _rel_err(deq, x) < 0.06  # fp8 e4m3 相对误差约 2^-3 ~ 6%

    def test_block_odd_shape(self):
        x = torch.randn(130, 300, dtype=torch.bfloat16)
        q, s = quantize_fp8_per_block(x)
        assert q.shape == (130, 300)
        assert s.shape == (2, 3)
        deq = dequantize_fp8_per_block(q, s, torch.bfloat16)
        assert deq.shape == x.shape

    def test_scale_values(self):
        x = torch.full((128, 128), 448.0, dtype=torch.bfloat16)
        q, s = quantize_fp8_per_block(x)
        assert torch.allclose(s, torch.ones_like(s))
        assert torch.allclose(q.float(), torch.full_like(q.float(), 448.0))


# ------------------------------------------------------------------ #
# 参数创建
# ------------------------------------------------------------------ #


class TestCreateWeights:
    @pytest.mark.parametrize(
        "kind,out_full,in_full,expect_scale",
        [
            ("column", 256, 512, (2, 4)),
            ("row", 256, 512, (2, 4)),
            ("replicated", 256, 512, (2, 4)),
        ],
    )
    def test_2d_shapes(self, kind, out_full, in_full, expect_scale):
        m = _method(kind, out_full, in_full)
        layer = _Layer()
        m.create_weights(layer)
        assert layer.weight.dtype == FP8_DTYPE
        assert layer.weight.shape == (out_full, in_full)
        assert layer.weight_scale.shape == expect_scale

    def test_merged_shapes(self):
        m = _method("merged_column", 512, 256, merged=[256, 256])
        layer = _Layer()
        m.create_weights(layer)
        assert layer.weight.shape == (512, 256)
        assert layer.weight_scale.shape == (4, 2)

    def test_qkv_shapes(self):
        # 8 heads × 64 dim = 512 q；4 kv × 64 = 256 kv；hidden 256
        m = _method(
            "qkv", 512 + 2 * 256, 256, qkv_meta=(8, 4, 64, 1)
        )
        layer = _Layer()
        m.create_weights(layer)
        assert layer.weight.shape == (1024, 256)
        assert layer.weight_scale.shape == (8, 2)


# ------------------------------------------------------------------ #
# 在线量化路径
# ------------------------------------------------------------------ #


class TestOnlineQuantize:
    def test_column_tp1(self):
        m = _method("column", 256, 512)
        layer = _Layer()
        m.create_weights(layer)
        w = torch.randn(256, 512, dtype=torch.bfloat16)
        layer.weight.weight_loader(layer.weight, w, 0)
        assert layer._quant_staging.shape == (256, 512)
        m.finalize(layer)
        assert layer._quant_staging is None
        deq = dequantize_fp8_per_block(layer.weight, layer.weight_scale, torch.bfloat16)
        assert _rel_err(deq, w) < 0.06

    def test_column_tp2_rank1(self):
        m = _method("column", 256, 512, tp=2, rank=1)
        layer = _Layer()
        m.create_weights(layer)
        w = torch.randn(256, 512, dtype=torch.bfloat16)
        layer.weight.weight_loader(layer.weight, w, 0)
        ref = w.narrow(0, 128, 128)
        m.finalize(layer)
        deq = dequantize_fp8_per_block(layer.weight, layer.weight_scale, torch.bfloat16)
        assert deq.shape == (128, 512)
        assert _rel_err(deq, ref) < 0.06

    def test_row_tp2_rank0(self):
        m = _method("row", 256, 512, tp=2, rank=0)
        layer = _Layer()
        m.create_weights(layer)
        w = torch.randn(256, 512, dtype=torch.bfloat16)
        layer.weight.weight_loader(layer.weight, w, 0)
        ref = w.narrow(1, 0, 256)
        m.finalize(layer)
        deq = dequantize_fp8_per_block(layer.weight, layer.weight_scale, torch.bfloat16)
        assert deq.shape == (256, 256)
        assert _rel_err(deq, ref) < 0.06

    def test_merged_two_segments(self):
        m = _method("merged_column", 512, 256, merged=[256, 256])
        layer = _Layer()
        m.create_weights(layer)
        gate = torch.randn(256, 256, dtype=torch.bfloat16)
        up = torch.randn(256, 256, dtype=torch.bfloat16)
        layer.weight.weight_loader(layer.weight, gate, 0)
        layer.weight.weight_loader(layer.weight, up, 1)
        m.finalize(layer)
        deq = dequantize_fp8_per_block(layer.weight, layer.weight_scale, torch.bfloat16)
        assert deq.shape == (512, 256)
        assert _rel_err(deq[:256], gate) < 0.06
        assert _rel_err(deq[256:], up) < 0.06

    def test_qkv_three_segments(self):
        meta = (8, 4, 64, 1)  # q=512, kv=256
        m = _method("qkv", 1024, 256, qkv_meta=meta)
        layer = _Layer()
        m.create_weights(layer)
        q = torch.randn(512, 256, dtype=torch.bfloat16)
        k = torch.randn(256, 256, dtype=torch.bfloat16)
        v = torch.randn(256, 256, dtype=torch.bfloat16)
        layer.weight.weight_loader(layer.weight, q, "q")
        layer.weight.weight_loader(layer.weight, k, "k")
        layer.weight.weight_loader(layer.weight, v, "v")
        m.finalize(layer)
        deq = dequantize_fp8_per_block(layer.weight, layer.weight_scale, torch.bfloat16)
        assert deq.shape == (1024, 256)
        assert _rel_err(deq[:512], q) < 0.06
        assert _rel_err(deq[512:768], k) < 0.06
        assert _rel_err(deq[768:], v) < 0.06

    def test_rejects_non_fp8_without_online(self):
        m = _method("column", 256, 512, online=False)
        layer = _Layer()
        m.create_weights(layer)
        with pytest.raises(ValueError, match="pre-quantized"):
            layer.weight.weight_loader(
                layer.weight, torch.randn(256, 512, dtype=torch.bfloat16), 0
            )


# ------------------------------------------------------------------ #
# 预量化路径（fp8 权重 + 独立 scale）
# ------------------------------------------------------------------ #


class TestPreQuantized:
    def test_column_tp_slices(self):
        w = torch.randn(256, 512, dtype=torch.bfloat16)
        wq, ws = quantize_fp8_per_block(w)
        m0 = _method("column", 256, 512, tp=2, rank=0, online=False)
        m1 = _method("column", 256, 512, tp=2, rank=1, online=False)
        l0, l1 = _Layer(), _Layer()
        m0.create_weights(l0)
        m1.create_weights(l1)
        for layer, meth, r in ((l0, m0, 0), (l1, m1, 1)):
            layer.weight.weight_loader(layer.weight, wq, 0)
            layer.weight_scale.weight_loader(layer.weight_scale, ws, 0)
        assert torch.equal(
            l0.weight.to(torch.bfloat16).view(torch.uint8),
            wq[:128].to(torch.bfloat16).view(torch.uint8),
        )
        assert torch.equal(l1.weight_scale, ws[1:2])
        d0 = dequantize_fp8_per_block(l0.weight, l0.weight_scale, torch.bfloat16)
        assert _rel_err(d0, w[:128]) < 0.06

    def test_row_tp_scale_cols(self):
        w = torch.randn(256, 512, dtype=torch.bfloat16)
        wq, ws = quantize_fp8_per_block(w)  # [256,512] → scale [2,4]
        m1 = _method("row", 256, 512, tp=2, rank=1, online=False)
        layer = _Layer()
        m1.create_weights(layer)
        layer.weight.weight_loader(layer.weight, wq, 0)
        layer.weight_scale.weight_loader(layer.weight_scale, ws, 0)
        assert layer.weight_scale.shape == (2, 2)
        assert torch.equal(layer.weight_scale, ws[:, 2:4])

    def test_qkv_pre_quantized(self):
        meta = (8, 4, 64, 1)
        q = torch.randn(512, 256, dtype=torch.bfloat16)
        k = torch.randn(256, 256, dtype=torch.bfloat16)
        v = torch.randn(256, 256, dtype=torch.bfloat16)
        qq, qs = quantize_fp8_per_block(q)
        kq, ks = quantize_fp8_per_block(k)
        vq, vs = quantize_fp8_per_block(v)
        m = _method("qkv", 1024, 256, tp=2, rank=1, qkv_meta=meta, online=False)
        layer = _Layer()
        m.create_weights(layer)
        for wq, ws, sid in ((qq, qs, "q"), (kq, ks, "k"), (vq, vs, "v")):
            layer.weight.weight_loader(layer.weight, wq, sid)
            layer.weight_scale.weight_loader(layer.weight_scale, ws, sid)
        # tp=2 rank1 本地布局：q[0:256]、k[256:384]、v[384:512]
        # 全局侧：q 取行 [256:512]；k/v head_idx=1 → 行 [128:256]
        assert torch.equal(
            layer.weight[:256].to(torch.bfloat16).view(torch.uint8),
            qq[256:512].to(torch.bfloat16).view(torch.uint8),
        )
        assert torch.equal(
            layer.weight[256:384].to(torch.bfloat16).view(torch.uint8),
            kq[128:256].to(torch.bfloat16).view(torch.uint8),
        )
        assert torch.equal(
            layer.weight[384:512].to(torch.bfloat16).view(torch.uint8),
            vq[128:256].to(torch.bfloat16).view(torch.uint8),
        )
        assert torch.equal(layer.weight_scale[0:2], qs[2:4])  # q 段（2 行网格）
        assert torch.equal(layer.weight_scale[2:3], ks[1:2])  # k 段
        assert torch.equal(layer.weight_scale[3:4], vs[1:2])  # v 段


# ------------------------------------------------------------------ #
# embed / lm_head fp8 反量化
# ------------------------------------------------------------------ #


class TestFallbackCatcher:
    def test_pending_then_scale(self):
        from utils.model_loader import Fp8FallbackCatcher

        layer = _Layer()
        embed = nn.Parameter(torch.zeros(256, 512, dtype=torch.bfloat16))
        layer.embed = embed
        params = {"model.embed_tokens.weight": embed}

        w = torch.randn(256, 512, dtype=torch.bfloat16)
        wq, ws = quantize_fp8_per_block(w)

        catcher = Fp8FallbackCatcher(params)
        assert catcher.is_pending_name("model.embed_tokens.weight")
        # 权重先到（safetensors 字典序 weight < weight_scale）
        assert catcher.feed("model.embed_tokens.weight", wq, load_fn=lambda p, x: p.data.copy_(x))
        assert catcher.unresolved() == ["model.embed_tokens.weight"]
        assert catcher.feed(
            "model.embed_tokens.weight_scale", ws, load_fn=lambda p, x: p.data.copy_(x)
        )
        assert catcher.unresolved() == []
        assert _rel_err(embed.data, w) < 0.06
        assert catcher.written == ["model.embed_tokens.weight"]

    def test_scale_first(self):
        from utils.model_loader import Fp8FallbackCatcher

        layer = _Layer()
        head = nn.Parameter(torch.zeros(256, 512, dtype=torch.bfloat16))
        params = {"lm_head.weight": head}
        w = torch.randn(256, 512, dtype=torch.bfloat16)
        wq, ws = quantize_fp8_per_block(w)
        catcher = Fp8FallbackCatcher(params)
        catcher.feed("lm_head.weight_scale", ws, load_fn=lambda p, x: p.data.copy_(x))
        catcher.feed("lm_head.weight", wq, load_fn=lambda p, x: p.data.copy_(x))
        assert _rel_err(head.data, w) < 0.06

    def test_normal_bf16_not_caught(self):
        from utils.model_loader import Fp8FallbackCatcher

        p = nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
        catcher = Fp8FallbackCatcher({"x.weight": p})
        w = torch.randn(4, 4, dtype=torch.bfloat16)
        assert catcher.feed("x.weight", w, load_fn=lambda pp, x: pp.data.copy_(x)) is False
        assert catcher.unresolved() == []


# ------------------------------------------------------------------ #
# MoE Experts FP8
# ------------------------------------------------------------------ #


def _experts(quant, tp=1, rank=0, num_experts=4, hidden=256, inter=128, ep=1):
    from src.moe.experts import Experts
    from src.moe.parallel import MoEParallelConfig

    parallel = MoEParallelConfig.from_engine(
        tp_size=tp, tp_rank=rank, enable_ep=(ep > 1), fused_backend="triton"
    )
    return Experts(num_experts, hidden, inter, parallel, quant=quant)


class TestExpertsFp8:
    def test_param_shapes(self):
        q = QuantConfig.from_name("fp8", device_capability=(8, 9), quantize_at_load=True)
        e = _experts(q)
        assert e.w13_weight.dtype == FP8_DTYPE
        assert e.w13_weight.shape == (4, 256, 256)  # [E, 2i, h]
        assert e.w2_weight.shape == (4, 256, 128)   # [E, h, i]
        assert e.w13_scale.shape == (4, 2, 2)
        assert e.w2_scale.shape == (4, 2, 1)

    def test_online_per_expert(self):
        q = QuantConfig.from_name("fp8", device_capability=(8, 9), quantize_at_load=True)
        e = _experts(q)
        torch.manual_seed(0)
        gate = torch.randn(128, 256, dtype=torch.bfloat16)
        up = torch.randn(128, 256, dtype=torch.bfloat16)
        down = torch.randn(256, 128, dtype=torch.bfloat16)
        e.weight_loader_w13(e.w13_weight, gate, "gate_proj.weight", 1)
        e.weight_loader_w13(e.w13_weight, up, "up_proj.weight", 1)
        e.weight_loader_w2(e.w2_weight, down, "down_proj.weight", 1)
        # cutlass 布局 [up; gate]：up 在前段
        d_up = dequantize_fp8_per_block(e.w13_weight.data[1, :128], e.w13_scale[1, :1], torch.bfloat16)
        d_gate = dequantize_fp8_per_block(e.w13_weight.data[1, 128:], e.w13_scale[1, 1:], torch.bfloat16)
        assert _rel_err(d_up, up) < 0.06
        assert _rel_err(d_gate, gate) < 0.06
        # w2 布局 [h, i] 与 HF down_proj 一致，无需转置
        d_down = dequantize_fp8_per_block(
            e.w2_weight.data[1], e.w2_scale[1], torch.bfloat16
        )
        assert _rel_err(d_down, down) < 0.06

    def test_pre_quantized_scales(self):
        q = QuantConfig.from_name("fp8", device_capability=(8, 9), quantize_at_load=False)
        # inter=256, tp=2 → i_local=128（128 对齐，block scale 网格完整）
        e = _experts(q, tp=2, rank=1, num_experts=4, hidden=256, inter=256)
        gate = torch.randn(256, 256, dtype=torch.bfloat16)
        gq, gs = quantize_fp8_per_block(gate)  # [256,256] → scale [2,2]
        e.weight_loader_w13(e.w13_weight, gq, "gate_proj.weight", 0)
        e.weight_loader_w13_scale(e.w13_scale, gs, "gate_proj.weight", 0)
        # rank1 取 gate 行 [128:256]（本地 i=128）；本地网格 1 行 ← 全局第 1 行
        assert e.w13_weight.data[0, 128:256].shape == (128, 256)
        assert torch.equal(e.w13_scale[0, 1:2], gs[1:2])
        assert torch.equal(e.w13_scale[0, 0:1], torch.zeros_like(gs[0:1]))  # up 段未写

    def test_ep_filter(self):
        q = QuantConfig.from_name("fp8", device_capability=(8, 9), quantize_at_load=True)
        # EP 需要 tp>1：tp=2 + enable_ep → ep_size=2，rank0 持有 expert 0,1
        e = _experts(q, tp=2, rank=0, num_experts=4, hidden=256, inter=256, ep=2)
        up = torch.randn(128, 256, dtype=torch.bfloat16)
        e.weight_loader_w13(e.w13_weight, up, "up_proj.weight", 3)
        assert e.w13_scale.abs().sum() == 0  # expert 3 非本地，未写入
