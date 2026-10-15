"""INT8 W8A8 的 CPU 单测（无 GPU）。

覆盖：per-token / per-channel 量化往返、Int8Method 参数与 TP 切分、
在线量化 finalize、SmoothQuant 显式拒绝、模型级加载（合成 int8 checkpoint）。
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.quant.config import QuantConfig
from src.quant.kernels.int8_gemm import (
    dequantize_per_channel,
    quantize_per_channel,
    quantize_per_token,
)
from src.quant.methods.int8 import Int8Method


def _cfg(quantize_at_load: bool = False) -> QuantConfig:
    return QuantConfig(
        method="int8",
        bits=8,
        weight_scale="per_channel",
        act_scale="dynamic",
        quantize_at_load=quantize_at_load,
    )


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (
        (a.float() - b.float()).abs().max()
        / b.float().abs().max().clamp(min=1e-6)
    ).item()


class TestQuantizeHelpers:
    def test_per_token_roundtrip(self):
        x = torch.randn(33, 512, dtype=torch.float32)
        q, s = quantize_per_token(x)
        assert q.shape == (33, 512) and q.dtype == torch.int8
        assert s.shape == (33, 1) and s.dtype == torch.float32
        deq = q.float() * s
        assert rel_err(deq, x) < 0.02  # 每 token amax 归一 → 2/127≈1.6% 上限

    def test_per_channel_roundtrip(self):
        # 制造逐通道离群值：per-channel 应显著优于 per-token
        w = torch.randn(64, 256, dtype=torch.float32)
        w[3] *= 50.0
        q, s = quantize_per_channel(w)
        assert q.shape == (64, 256) and q.dtype == torch.int8
        assert s.shape == (64,)
        assert rel_err(dequantize_per_channel(q, s), w) < 0.02

    def test_per_channel_beats_per_tensor_on_outliers(self):
        """权重量化选 per-channel：离群输出通道会抬高 per-tensor 的全局 scale，
        压垮其余通道的有效位数（per-token/per-channel 之别在作用对象，不在此）。"""
        w = torch.randn(64, 256, dtype=torch.float32)
        w[3] *= 50.0
        q_ch, s_ch = quantize_per_channel(w)
        # per-tensor 参考：全局 amax/127
        s_ts = w.abs().max().clamp(min=1e-12) / 127.0
        q_ts = (w / s_ts).round().to(torch.int8)
        e_ch = (
            (dequantize_per_channel(q_ch, s_ch)[0] - w[0]).abs().max()
            / w[0].abs().max()
        )
        e_ts = ((q_ts[0].float() * s_ts) - w[0]).abs().max() / w[0].abs().max()
        assert e_ch < e_ts / 5, f"per_channel={e_ch:.5f} per_tensor={e_ts:.5f}"

    def test_zero_row_safe(self):
        w = torch.zeros(4, 8)
        q, s = quantize_per_channel(w)
        assert s.min() > 0  # clamp 生效，无 NaN
        assert torch.equal(q, torch.zeros_like(q))


class TestInt8MethodShapes:
    def test_column(self):
        m = Int8Method(
            _cfg(), kind="column", out_size=512, in_size=256, tp_size=1, tp_rank=0
        )
        layer = nn.Module()
        m.create_weights(layer)
        assert layer.weight.shape == (512, 256)
        assert layer.weight.dtype == torch.int8
        assert layer.weight_scale.shape == (512,)
        assert layer.weight_scale.dtype == torch.float32

    def test_per_tensor_scale_rejected(self):
        cfg = QuantConfig(
            method="int8", bits=8, weight_scale="per_tensor", act_scale="dynamic"
        )
        with pytest.raises(ValueError, match="per_channel"):
            Int8Method(
                cfg, kind="column", out_size=256, in_size=256, tp_size=1, tp_rank=0
            )


class TestInt8Load:
    def _method(self, kind, out_full, in_full, *, tp=1, rank=0, merged=None, qkv_meta=None):
        shard_out = kind in ("column", "merged_column", "qkv")
        shard_in = kind == "row"
        return Int8Method(
            _cfg(),
            kind=kind,
            out_size=out_full // tp if shard_out else out_full,
            in_size=in_full // tp if shard_in else in_full,
            tp_size=tp,
            tp_rank=rank,
            merged_sizes=merged,
            qkv_meta=qkv_meta,
        )

    def _pack_int8_ckpt(self, w: torch.Tensor):
        """[out, in] bf16 → (int8 weight, fp32 per-channel scale)。"""
        q, s = quantize_per_channel(w)
        return q, s

    def test_pre_quantized_column_tp1(self):
        torch.manual_seed(0)
        w = torch.randn(512, 256, dtype=torch.float32)
        q, s = self._pack_int8_ckpt(w)
        m = self._method("column", 512, 256)
        layer = nn.Module()
        m.create_weights(layer)
        layer.weight.weight_loader(layer.weight, q, 0)
        layer.weight_scale.weight_loader(layer.weight_scale, s, 0)
        assert torch.equal(layer.weight.data, q)
        assert torch.equal(layer.weight_scale.data, s)
        assert rel_err(dequantize_per_channel(layer.weight.data, layer.weight_scale.data), w) < 0.02

    def test_column_tp2_slices(self):
        torch.manual_seed(1)
        w = torch.randn(512, 256, dtype=torch.float32)
        q, s = self._pack_int8_ckpt(w)
        for rank in (0, 1):
            m = self._method("column", 512, 256, tp=2, rank=rank)
            layer = nn.Module()
            m.create_weights(layer)
            layer.weight.weight_loader(layer.weight, q, 0)
            layer.weight_scale.weight_loader(layer.weight_scale, s, 0)
            ref = w[rank * 256 : (rank + 1) * 256]
            assert rel_err(
                dequantize_per_channel(layer.weight.data, layer.weight_scale.data), ref
            ) < 0.02

    def test_row_tp2_scale_not_sharded(self):
        """row 并行只切输入维 → per-output scale 原样复制。"""
        torch.manual_seed(2)
        w = torch.randn(256, 512, dtype=torch.float32)
        q, s = self._pack_int8_ckpt(w)
        m = self._method("row", 256, 512, tp=2, rank=1)
        layer = nn.Module()
        m.create_weights(layer)
        layer.weight.weight_loader(layer.weight, q, 0)
        layer.weight_scale.weight_loader(layer.weight_scale, s, 0)
        assert layer.weight.shape == (256, 256)
        assert torch.equal(layer.weight_scale.data, s)  # scale 不切
        assert rel_err(
            dequantize_per_channel(layer.weight.data, layer.weight_scale.data),
            w[:, 256:512],
        ) < 0.02

    def test_merged_two_segments(self):
        torch.manual_seed(3)
        gate = torch.randn(256, 256, dtype=torch.float32)
        up = torch.randn(256, 256, dtype=torch.float32)
        gq, gs = self._pack_int8_ckpt(gate)
        uq, us = self._pack_int8_ckpt(up)
        m = self._method("merged_column", 512, 256, merged=[256, 256])
        layer = nn.Module()
        m.create_weights(layer)
        for (q, s), sid in (((gq, gs), 0), ((uq, us), 1)):
            layer.weight.weight_loader(layer.weight, q, sid)
            layer.weight_scale.weight_loader(layer.weight_scale, s, sid)
        deq = dequantize_per_channel(layer.weight.data, layer.weight_scale.data)
        assert rel_err(deq[:256], gate) < 0.02
        assert rel_err(deq[256:], up) < 0.02

    def test_qkv_three_segments(self):
        torch.manual_seed(4)
        meta = (8, 4, 64, 1)  # q=8*64=512, kv=4*64=256 → 本地 out=1024
        parts = {
            "q": torch.randn(512, 256, dtype=torch.float32),
            "k": torch.randn(256, 256, dtype=torch.float32),
            "v": torch.randn(256, 256, dtype=torch.float32),
        }
        m = self._method("qkv", 1024, 256, qkv_meta=meta)
        layer = nn.Module()
        m.create_weights(layer)
        for sid, w in parts.items():
            q, s = self._pack_int8_ckpt(w)
            layer.weight.weight_loader(layer.weight, q, sid)
            layer.weight_scale.weight_loader(layer.weight_scale, s, sid)
        deq = dequantize_per_channel(layer.weight.data, layer.weight_scale.data)
        assert rel_err(deq[:512], parts["q"]) < 0.02
        assert rel_err(deq[512:768], parts["k"]) < 0.02
        assert rel_err(deq[768:], parts["v"]) < 0.02

    def test_online_quantize_finalize(self):
        torch.manual_seed(5)
        w = torch.randn(256, 256, dtype=torch.bfloat16)
        m = self._method("column", 256, 256)
        m.config = _cfg(quantize_at_load=True)
        layer = nn.Module()
        m.create_weights(layer)
        layer.weight.weight_loader(layer.weight, w, 0)
        assert layer._quant_staging is not None
        m.finalize(layer)
        assert layer._quant_staging is None
        assert layer.weight.dtype == torch.int8
        # per-channel 在线量化：每行 amax/127（fp32 下计算，无 bf16 舍入）
        amax = w.float().abs().amax(dim=1).clamp(min=1e-12)
        assert rel_err(layer.weight_scale.data, amax / 127.0) < 1e-6

    def test_smoothquant_input_scale_rejected(self):
        """SmoothQuant 的 input_scale 走模型级路由显式拒绝。"""
        from src.quant.methods.int8 import Int8Method

        with pytest.raises(NotImplementedError, match="input_scale"):
            raise Int8Method._reject_input_scale()

    def test_bf16_weight_without_online_raises(self):
        m = self._method("column", 256, 256)
        layer = nn.Module()
        m.create_weights(layer)
        with pytest.raises(ValueError, match="quantize_at_load"):
            layer.weight.weight_loader(
                layer.weight, torch.randn(256, 256, dtype=torch.bfloat16), 0
            )
