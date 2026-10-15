"""MoE 专家 INT4 weight-only 的 CPU 单测（无 GPU）。

验证：Experts 的 INT4 packed 参数布局、AWQ/GPTQ per-expert 加载归一化、
TP/EP 切分语义，以及反量化结果与原权重的误差（INT4 group=128 量级）。

kernel 数值由 tests/gpu/test_quant_int4.py 的 MoE 段覆盖（需 GPU；
本地 torch 2.11 与 moe_align_block_size 不兼容，属既有环境问题）。
"""

from __future__ import annotations

import pytest
import torch

from src.moe.experts import Experts
from src.moe.parallel import MoEParallelConfig
from src.quant.config import QuantConfig
from src.quant.kernels.int4_gemm import (
    dequantize_int4_reference,
    pack_int4_cols,
    pack_int4_rows,
)
from tests.unit.test_quant_int4 import quantize_int4_asym, quantize_int4_sym, rel_err

GROUP = 128
# H != I：in-major 与 out-major 在 in==out 时形状相同（方形层歧义），
# 这里取非方形维度使两种布局可无歧义识别；同时满足 out-major 的
# 组打包要求（in/G 是 8 的倍数：1024/128=8、2048/128=16）
H = 1024
I = 2048
E = 4


def _cfg(method: str) -> QuantConfig:
    return QuantConfig(
        method=method,
        bits=4,
        weight_scale="group",
        act_scale=None,
        group_size=GROUP,
        zero_point=(method == "awq"),
        sym=(method != "awq"),
    )


def _parallel(tp: int = 1, rank: int = 0, ep: int = 1):
    """ep>1 通过 enable_ep 表达（EP 复用 TP ranks：ep_size=tp_size, ep_rank=tp_rank）。"""
    return MoEParallelConfig.from_engine(
        tp_size=tp,
        tp_rank=rank,
        enable_ep=ep > 1,
        fused_backend="triton",
    )


def _awq_expert_tensors(w_gate, w_up, w_down, *, out_major: bool = False):
    """[i, h]/[h, i] 的 gate/up/down（HF [out, in]）→ AWQ 风格三件套。"""
    out = {}
    for name, w in (("gate", w_gate), ("up", w_up), ("down", w_down)):
        q, s, z = quantize_int4_asym(w.t().contiguous())  # → [in, out]
        if out_major:
            out[name] = (
                pack_int4_cols(q).t().contiguous(),
                s.t().contiguous(),
                pack_int4_cols(z).t().contiguous(),
            )
        else:
            out[name] = (
                pack_int4_rows(q, interleave=True),
                s,
                pack_int4_rows(z, interleave=True),
            )
    return out


def _gptq_expert_tensors(w_gate, w_up, w_down):
    out = {}
    for name, w in (("gate", w_gate), ("up", w_up), ("down", w_down)):
        q, s = quantize_int4_sym(w.t().contiguous())  # → [in, out]
        out[name] = (pack_int4_cols(q), s, None)
    return out


def _load_expert(experts: Experts, eid: int, tensors: dict) -> None:
    for name in ("gate", "up", "down"):
        qw, sc, zz = tensors[name]
        proj = f"{name}_proj.weight"
        experts.weight_loader_int4(qw, proj, eid, "qweight")
        experts.weight_loader_int4(sc, proj, eid, "scales")
        if zz is not None:
            experts.weight_loader_int4(zz, proj, eid, "qzeros")


def deq(param_q, param_s, param_z, *, bias: float):
    """[E, K, N/8] packed → 反量化 [E, K, N]（逐 expert 调用 dense 参考）。"""
    outs = []
    for e in range(param_q.shape[0]):
        outs.append(
            dequantize_int4_reference(
                param_q[e], param_s[e], param_z[e] if param_z is not None else None,
                group_size=GROUP, zero_bias=bias,
            )
        )
    return outs


class TestExpertsInt4Shapes:
    def test_awq_shapes(self):
        e = Experts(E, H, I, _parallel(), quant=_cfg("awq"))
        i = I  # tp=1
        assert e.w13_qweight.shape == (E, H, 2 * i // 8)
        assert e.w13_scales.shape == (E, H // GROUP, 2 * i)
        assert e.w13_zeros is not None and e.w13_zeros.shape == (E, H // GROUP, 2 * i)
        assert e.w2_qweight.shape == (E, i, H // 8)
        assert e.w2_scales.shape == (E, i // GROUP, H)
        assert e.zero_bias == 0.0
        assert e.int4_asym is True

    def test_gptq_sym_shapes(self):
        e = Experts(E, H, I, _parallel(), quant=_cfg("gptq"))
        assert e.w13_zeros is None
        assert e.w2_zeros is None
        assert e.zero_bias == 8.0
        assert e.int4_asym is False

    def test_bf16_path_unchanged(self):
        e = Experts(E, H, I, _parallel(), quant=None)
        assert e.w13_weight.shape == (E, 2 * I, H)
        assert e.int4 is False


class TestExpertsInt4Load:
    def test_awq_per_expert_and_segments(self):
        torch.manual_seed(0)
        e = Experts(E, H, I, _parallel(), quant=_cfg("awq"))
        refs = {}
        for eid in range(E):
            w_gate = torch.randn(I, H, dtype=torch.float32) * 0.1
            w_up = torch.randn(I, H, dtype=torch.float32) * 0.1
            w_down = torch.randn(H, I, dtype=torch.float32) * 0.1
            refs[eid] = (w_gate, w_up, w_down)
            _load_expert(
                e, eid, _awq_expert_tensors(w_gate, w_up, w_down)
            )
        gu = deq(e.w13_qweight.data, e.w13_scales.data, e.w13_zeros.data, bias=0.0)
        dn = deq(e.w2_qweight.data, e.w2_scales.data, e.w2_zeros.data, bias=0.0)
        for eid, (w_gate, w_up, w_down) in refs.items():
            # 内部序 [up | gate]（沿用 cutlass 约定）；反量化为 [in, out]，
            # HF 权重是 [out, in] → 比较前转置
            assert rel_err(gu[eid][:, :I], w_up.t()) < 0.12, f"e{eid} up"
            assert rel_err(gu[eid][:, I:], w_gate.t()) < 0.12, f"e{eid} gate"
            assert rel_err(dn[eid], w_down.t()) < 0.12, f"e{eid} down"

    def test_gptq_sym_per_expert(self):
        torch.manual_seed(1)
        e = Experts(E, H, I, _parallel(), quant=_cfg("gptq"))
        refs = {}
        for eid in range(E):
            w_gate = torch.randn(I, H, dtype=torch.float32) * 0.1
            w_up = torch.randn(I, H, dtype=torch.float32) * 0.1
            w_down = torch.randn(H, I, dtype=torch.float32) * 0.1
            refs[eid] = (w_gate, w_up, w_down)
            _load_expert(e, eid, _gptq_expert_tensors(w_gate, w_up, w_down))
        gu = deq(e.w13_qweight.data, e.w13_scales.data, None, bias=8.0)
        dn = deq(e.w2_qweight.data, e.w2_scales.data, None, bias=8.0)
        for eid, (w_gate, w_up, w_down) in refs.items():
            assert rel_err(gu[eid][:, :I], w_up.t()) < 0.12, f"e{eid} up"
            assert rel_err(gu[eid][:, I:], w_gate.t()) < 0.12, f"e{eid} gate"
            assert rel_err(dn[eid], w_down.t()) < 0.12, f"e{eid} down"

    @pytest.mark.parametrize("rank", [0, 1])
    def test_awq_tp2_expert_slices(self, rank):
        """w13 切 out=2i、w2 切 in=i；本地反量化应等于对应 TP 分片。"""
        torch.manual_seed(2)
        e = Experts(E, H, I, _parallel(tp=2, rank=rank), quant=_cfg("awq"))
        i = I // 2
        for eid in range(E):
            w_gate = torch.randn(I, H, dtype=torch.float32) * 0.1
            w_up = torch.randn(I, H, dtype=torch.float32) * 0.1
            w_down = torch.randn(H, I, dtype=torch.float32) * 0.1
            _load_expert(e, eid, _awq_expert_tensors(w_gate, w_up, w_down))
            gu = dequantize_int4_reference(
                e.w13_qweight.data[eid], e.w13_scales.data[eid],
                e.w13_zeros.data[eid], group_size=GROUP,
            )
            # 反量化 [in, out_local] vs HF [out, in] 的对应分片 → 转置后比
            ref_up = w_up[rank * i : (rank + 1) * i].t()
            ref_gate = w_gate[rank * i : (rank + 1) * i].t()
            assert rel_err(gu[:, :i], ref_up) < 0.12
            assert rel_err(gu[:, i:], ref_gate) < 0.12
            dn = dequantize_int4_reference(
                e.w2_qweight.data[eid], e.w2_scales.data[eid],
                e.w2_zeros.data[eid], group_size=GROUP,
            )
            ref_down = w_down[:, rank * i : (rank + 1) * i].t()
            assert rel_err(dn, ref_down) < 0.12

    def test_ep_filters_remote_experts(self):
        """EP=2：rank1 只写本地 expert 2,3；远程 expert 参数保持零。"""
        torch.manual_seed(3)
        # enable_ep + tp_rank=1 → ep_size=2, ep_rank=1（持有 expert 2,3）
        e = Experts(E, H, I, _parallel(tp=2, rank=1, ep=2), quant=_cfg("awq"))
        assert e.num_experts == 2
        w_gate = torch.randn(I, H, dtype=torch.float32) * 0.1
        w_up = torch.randn(I, H, dtype=torch.float32) * 0.1
        w_down = torch.randn(H, I, dtype=torch.float32) * 0.1
        # expert 0 是远程 → 不写
        _load_expert(e, 0, _awq_expert_tensors(w_gate, w_up, w_down))
        assert e.w13_qweight.data.abs().sum() == 0
        # expert 3 是本地（local id 1）
        _load_expert(e, 3, _awq_expert_tensors(w_gate, w_up, w_down))
        assert e.w13_qweight.data[1].abs().sum() > 0
        assert e.w13_qweight.data[0].abs().sum() == 0

    def test_awq_out_major_checkpoint(self):
        """AutoAWQ 原始 out-major 布局也能正确归一化。"""
        torch.manual_seed(4)
        e = Experts(E, H, I, _parallel(), quant=_cfg("awq"))
        w_gate = torch.randn(I, H, dtype=torch.float32) * 0.1
        w_up = torch.randn(I, H, dtype=torch.float32) * 0.1
        w_down = torch.randn(H, I, dtype=torch.float32) * 0.1
        _load_expert(e, 0, _awq_expert_tensors(w_gate, w_up, w_down, out_major=True))
        gu = dequantize_int4_reference(
            e.w13_qweight.data[0], e.w13_scales.data[0],
            e.w13_zeros.data[0], group_size=GROUP,
        )
        assert rel_err(gu[:, :I], w_up.t()) < 0.12
        assert rel_err(gu[:, I:], w_gate.t()) < 0.12
