"""INT4 weight-only 的 CPU 单测（无 GPU）。

验证策略：自建 INT4 量化器 + 打包器（对齐 vLLM 参考语义）→ 经
``Int4Method`` 加载 → 参考反量化 → 与原权重比对。这样在无 checkpoint
的环境下也能验证「打包语义 ↔ 加载归一化 ↔ TP 切分」整条链的数学正确性。

覆盖两种格式 × 两种方向：
  AWQ in-major（含 N 交织）/ AWQ out-major（AutoAWQ 原始）/ GPTQ（uint4b8）

维度约定：in=1024/2048、out=512，使 in/G 为 8 的倍数——这是 out-major
布局下 qzeros 按组打包（8 组/int32）的必要条件，与真实模型（in≥1024）一致。
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.quant.config import QuantConfig
from src.quant.kernels.int4_gemm import (
    AWQ_INTERLEAVE,
    dequantize_int4_reference,
    pack_int4_cols,
    pack_int4_rows,
    qweight_to_in_major,
    qzeros_to_in_major,
    scales_to_in_major,
    unpack_int4_cols,
    unpack_int4_rows,
)
from src.quant.methods.int4 import Int4Method

GROUP = 128
IN = 1024
OUT = 512


# ====================================================================== #
# 参考量化器（对齐 vLLM quantize_weights 的语义）
# ====================================================================== #


def quantize_int4_asym(w: torch.Tensor, group: int = GROUP):
    """AWQ 风格非对称：s=(max-min)/15，z=round(|min|/s)。"""
    k, n = w.shape
    wg = w.view(k // group, group, n)
    wmax = wg.amax(dim=1)
    wmin = wg.amin(dim=1)
    s = (wmax - wmin).clamp(min=1e-5) / 15.0
    z = (wmin.abs() / s).round().clamp(0, 15)
    q = ((wg / s[:, None, :]).round() + z[:, None, :]).clamp(0, 15)
    return q.to(torch.uint8).view(k, n), s.float(), z.to(torch.uint8)


def quantize_int4_sym(w: torch.Tensor, group: int = GROUP):
    """GPTQ uint4b8 对称：s=max(|max|/7, |min|/8)，存值 = q + 8。"""
    k, n = w.shape
    wg = w.view(k // group, group, n)
    wmax = wg.amax(dim=1)
    wmin = wg.amin(dim=1)
    s = torch.maximum(wmax.abs() / 7.0, wmin.abs() / 8.0).clamp(min=1e-5)
    q = ((wg / s[:, None, :]).round() + 8).clamp(0, 15)
    return q.to(torch.uint8).view(k, n), s.float()


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (
        (a.float() - b.float()).abs().max()
        / b.float().abs().max().clamp(min=1e-6)
    ).item()


# ====================================================================== #
# 打包 / 解包
# ====================================================================== #


class TestPackUnpack:
    def test_rows_roundtrip(self):
        w = torch.randint(0, 16, (256, 512), dtype=torch.uint8)
        q = pack_int4_rows(w)
        assert q.shape == (256, 64) and q.dtype == torch.int32
        assert torch.equal(unpack_int4_rows(q, 512), w)

    def test_cols_roundtrip(self):
        w = torch.randint(0, 16, (256, 512), dtype=torch.uint8)
        q = pack_int4_cols(w)
        assert q.shape == (32, 512)
        assert torch.equal(unpack_int4_cols(q, 256), w)

    def test_nibble_order(self):
        """nibble j 必须对应 N 通道 8*n8+j（行打包约定）。"""
        w = torch.zeros(8, 8, dtype=torch.uint8)
        w[:, 3] = 7
        q = pack_int4_rows(w)
        assert q[0, 0].item() == 7 << (4 * 3)
        assert unpack_int4_rows(q, 8)[0, 3].item() == 7

    def test_awq_interleave_applied(self):
        """interleave=True 时 nibble i 对应输出通道 AWQ_INTERLEAVE[i]。"""
        w = torch.zeros(8, 8, dtype=torch.uint8)
        w[:, 1] = 9
        q = pack_int4_rows(w, interleave=True)
        assert q[0, 0].item() == 9 << (4 * list(AWQ_INTERLEAVE).index(1))


# ====================================================================== #
# 归一化（方向 + 交织）
# ====================================================================== #


class TestNormalization:
    @pytest.mark.parametrize("method", ["awq", "gptq"])
    def test_in_major(self, method):
        w = torch.randint(0, 16, (IN, OUT), dtype=torch.uint8)
        q = pack_int4_rows(w, interleave=(method == "awq"))
        out = qweight_to_in_major(
            q, in_full=IN, out_full=OUT, method=method, awq_interleave=True
        )
        assert out.shape == (IN, OUT // 8)
        if method == "gptq":
            assert torch.equal(out, q)
        else:
            # AWQ：交织被还原 → 与未交织的打包一致
            assert torch.equal(out, pack_int4_rows(w))

    def test_out_major_transposed(self):
        """[out, in/8] → 归一化为 [in, out/8]，内容与参考一致。"""
        w = torch.randint(0, 16, (IN, OUT), dtype=torch.uint8)
        loaded = pack_int4_cols(w).t().contiguous()  # [out, in/8]
        out = qweight_to_in_major(
            loaded, in_full=IN, out_full=OUT, method="awq", awq_interleave=False
        )
        assert torch.equal(out, pack_int4_rows(w))

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="unsupported AWQ/GPTQ qweight shape"):
            qweight_to_in_major(
                torch.zeros(7, 9, dtype=torch.int32),
                in_full=IN,
                out_full=OUT,
                method="awq",
            )

    def test_scales_both_orientations(self):
        s = torch.rand(IN // GROUP, OUT, dtype=torch.float16)  # [in/G, out]
        assert torch.equal(
            scales_to_in_major(s, in_full=IN, out_full=OUT, group_size=GROUP), s
        )
        assert torch.equal(
            scales_to_in_major(
                s.t().contiguous(), in_full=IN, out_full=OUT, group_size=GROUP
            ),
            s,
        )

    def test_qzeros_both_orientations(self):
        g = IN // GROUP
        logical = torch.randint(0, 16, (g, OUT), dtype=torch.uint8)  # [in/G, out]
        plain = pack_int4_rows(logical)                     # 无交织（内部目标）
        z_in = pack_int4_rows(logical, interleave=True)      # AWQ in-major（带交织）
        z_out = pack_int4_rows(logical.t().contiguous())     # [out, g/8]（打包组）
        assert z_out.shape == (OUT, g // 8)
        # in-major AWQ：交织被还原
        assert torch.equal(
            qzeros_to_in_major(z_in, in_full=IN, out_full=OUT, group_size=GROUP),
            plain,
        )
        # out-major：按组打包 → 转置重打包（无交织语义）
        assert torch.equal(
            qzeros_to_in_major(z_out, in_full=IN, out_full=OUT, group_size=GROUP),
            plain,
        )


# ====================================================================== #
# 参考反量化语义
# ====================================================================== #


class TestDequantReference:
    def test_asym(self):
        w = torch.randn(IN, OUT, dtype=torch.bfloat16) * 0.1
        q, s, z = quantize_int4_asym(w)
        deq = dequantize_int4_reference(pack_int4_rows(q), s, z, group_size=GROUP)
        assert rel_err(deq, w) < 0.12

    def test_sym_bias8(self):
        w = torch.randn(IN, OUT, dtype=torch.bfloat16) * 0.1
        q, s = quantize_int4_sym(w)
        deq = dequantize_int4_reference(
            pack_int4_rows(q), s, None, group_size=GROUP, zero_bias=8.0
        )
        assert rel_err(deq, w) < 0.12


# ====================================================================== #
# Int4Method：参数创建 + 加载
# ====================================================================== #


def _cfg(method: str) -> QuantConfig:
    """直接构造（不经 from_name：那是 CLI 路径，有意要求预量化 checkpoint）。"""
    return QuantConfig(
        method=method,
        bits=4,
        weight_scale="group",
        act_scale=None,
        group_size=GROUP,
        zero_point=(method == "awq"),
        sym=(method != "awq"),
    )


def _method(kind, out_full, in_full, *, tp=1, rank=0, merged=None, qkv_meta=None, method="awq"):
    shard_out = kind in ("column", "merged_column", "qkv")
    shard_in = kind == "row"
    return Int4Method(
        _cfg(method),
        kind=kind,
        out_size=out_full // tp if shard_out else out_full,
        in_size=in_full // tp if shard_in else in_full,
        tp_size=tp,
        tp_rank=rank,
        merged_sizes=merged,
        qkv_meta=qkv_meta,
    )


class _Layer(nn.Module):
    pass


def _awq_tensors(w: torch.Tensor, *, out_major: bool, interleave: bool = True):
    """[in, out] bf16 → AWQ 风格 checkpoint 张量 (qweight, scales, qzeros)。"""
    q, s, z = quantize_int4_asym(w)
    if out_major:
        return (
            pack_int4_cols(q).t().contiguous(),  # [out, in/8]
            s.t().contiguous(),                  # [out, in/G]
            pack_int4_cols(z).t().contiguous(),  # [out, in/G/8]
        )
    # AWQ 的 qzeros 与 qweight 共用同一组 nibble 顺序 → 同样施加交织
    return (
        pack_int4_rows(q, interleave=interleave),
        s,
        pack_int4_rows(z, interleave=interleave),
    )


def _gptq_tensors(w: torch.Tensor):
    q, s = quantize_int4_sym(w)
    return pack_int4_cols(q), s, None  # [in/8, out], [in/G, out]


def _load_dense(m, layer, qw, sc, zz, shard_id=0):
    layer.qweight.weight_loader(layer.qweight, qw, shard_id)
    layer.scales.weight_loader(layer.scales, sc, shard_id)
    if zz is not None:
        layer.qzeros.weight_loader(layer.qzeros, zz, shard_id)


class TestInt4MethodShapes:
    def test_column(self):
        m = _method("column", OUT, IN)
        layer = _Layer()
        m.create_weights(layer)
        assert layer.qweight.shape == (IN, OUT // 8)
        assert layer.qweight.dtype == torch.int32
        assert layer.scales.shape == (IN // GROUP, OUT)
        assert layer.scales.dtype == torch.float16
        assert layer.qzeros.shape == (IN // GROUP, OUT)
        assert layer.qzeros.dtype == torch.uint8

    def test_gptq_sym_has_no_zeros(self):
        m = _method("column", OUT, IN, method="gptq")
        layer = _Layer()
        m.create_weights(layer)
        assert layer.qzeros is None
        assert m.zero_bias == 8.0

    def test_awq_has_zeros(self):
        m = _method("column", OUT, IN, method="awq")
        layer = _Layer()
        m.create_weights(layer)
        assert layer.qzeros is not None
        assert m.zero_bias == 0.0


class TestInt4LoadDense:
    @pytest.mark.parametrize("out_major", [False, True])
    def test_awq_both_orientations_tp1(self, out_major):
        torch.manual_seed(0)
        w = torch.randn(IN, OUT, dtype=torch.bfloat16) * 0.1
        qw, sc, zz = _awq_tensors(w, out_major=out_major)
        m = _method("column", OUT, IN)
        layer = _Layer()
        m.create_weights(layer)
        _load_dense(m, layer, qw, sc, zz)
        deq = dequantize_int4_reference(
            layer.qweight.data, layer.scales.data, layer.qzeros.data, group_size=GROUP
        )
        assert rel_err(deq, w) < 0.12, f"out_major={out_major}"

    def test_gptq_sym_tp1(self):
        torch.manual_seed(1)
        w = torch.randn(IN, OUT, dtype=torch.bfloat16) * 0.1
        qw, sc, _ = _gptq_tensors(w)
        m = _method("column", OUT, IN, method="gptq")
        layer = _Layer()
        m.create_weights(layer)
        _load_dense(m, layer, qw, sc, None)
        deq = dequantize_int4_reference(
            layer.qweight.data,
            layer.scales.data,
            None,
            group_size=GROUP,
            zero_bias=8.0,
        )
        assert rel_err(deq, w) < 0.12

    def test_column_tp2_slices(self):
        torch.manual_seed(2)
        w = torch.randn(IN, OUT, dtype=torch.bfloat16) * 0.1
        qw, sc, zz = _awq_tensors(w, out_major=False)
        for rank in (0, 1):
            m = _method("column", OUT, IN, tp=2, rank=rank)
            layer = _Layer()
            m.create_weights(layer)
            _load_dense(m, layer, qw, sc, zz)
            deq = dequantize_int4_reference(
                layer.qweight.data, layer.scales.data, layer.qzeros.data,
                group_size=GROUP,
            )
            ref = w[:, rank * (OUT // 2) : (rank + 1) * (OUT // 2)]
            assert rel_err(deq, ref) < 0.12, f"rank={rank}"

    def test_row_tp2_slices(self):
        torch.manual_seed(3)
        w = torch.randn(IN, OUT, dtype=torch.bfloat16) * 0.1
        qw, sc, zz = _awq_tensors(w, out_major=False)
        for rank in (0, 1):
            m = _method("row", OUT, IN, tp=2, rank=rank)
            layer = _Layer()
            m.create_weights(layer)
            _load_dense(m, layer, qw, sc, zz)
            deq = dequantize_int4_reference(
                layer.qweight.data, layer.scales.data, layer.qzeros.data,
                group_size=GROUP,
            )
            ref = w[rank * (IN // 2) : (rank + 1) * (IN // 2)]
            assert rel_err(deq, ref) < 0.12, f"rank={rank}"

    def test_merged_two_segments(self):
        torch.manual_seed(4)
        gate = torch.randn(IN, OUT // 2, dtype=torch.bfloat16) * 0.1
        up = torch.randn(IN, OUT // 2, dtype=torch.bfloat16) * 0.1
        gq, gs, gz = _awq_tensors(gate, out_major=False)
        uq, us, uz = _awq_tensors(up, out_major=False)
        m = _method("merged_column", OUT, IN, merged=[OUT // 2, OUT // 2])
        layer = _Layer()
        m.create_weights(layer)
        _load_dense(m, layer, gq, gs, gz, shard_id=0)
        _load_dense(m, layer, uq, us, uz, shard_id=1)
        deq = dequantize_int4_reference(
            layer.qweight.data, layer.scales.data, layer.qzeros.data, group_size=GROUP
        )
        assert rel_err(deq[:, : OUT // 2], gate) < 0.12
        assert rel_err(deq[:, OUT // 2 :], up) < 0.12

    def test_qkv_three_segments(self):
        torch.manual_seed(5)
        meta = (8, 4, 64, 1)  # q=512, kv=256 → 本地 out = 1024
        parts = {
            "q": torch.randn(IN, 512, dtype=torch.bfloat16) * 0.1,
            "k": torch.randn(IN, 256, dtype=torch.bfloat16) * 0.1,
            "v": torch.randn(IN, 256, dtype=torch.bfloat16) * 0.1,
        }
        m = _method("qkv", 1024, IN, qkv_meta=meta)
        layer = _Layer()
        m.create_weights(layer)
        for sid, w in parts.items():
            qw, sc, zz = _awq_tensors(w, out_major=False)
            _load_dense(m, layer, qw, sc, zz, shard_id=sid)
        deq = dequantize_int4_reference(
            layer.qweight.data, layer.scales.data, layer.qzeros.data, group_size=GROUP
        )
        assert rel_err(deq[:, :512], parts["q"]) < 0.12
        assert rel_err(deq[:, 512:768], parts["k"]) < 0.12
        assert rel_err(deq[:, 768:], parts["v"]) < 0.12


class TestConfigGuards:
    def test_desc_act_rejected(self):
        cfg = QuantConfig(
            method="gptq", bits=4, weight_scale="group", act_scale=None,
            group_size=GROUP, desc_act=True, sym=True,
        )
        with pytest.raises(NotImplementedError, match="desc_act"):
            Int4Method(
                cfg, kind="column", out_size=256, in_size=256, tp_size=1, tp_rank=0
            )

    def test_group_must_divide_in(self):
        with pytest.raises(ValueError, match="group_size"):
            Int4Method(
                _cfg("awq"),
                kind="column", out_size=256, in_size=100, tp_size=1, tp_rank=0,
            )

    def test_awq_requires_checkpoint_via_cli_path(self):
        """from_name 是 CLI 路径：awq/gptq 必须搭配预量化 checkpoint。"""
        with pytest.raises(ValueError, match="pre-quantized checkpoint"):
            QuantConfig.from_name("awq", device_capability=(8, 6))

    def test_gptq_sym_from_hf_config_has_no_zeros(self):
        """真实 GPTQ checkpoint 只给 sym：必须据此判定零点语义。

        （若误用 zero_point（默认 True）判定，会要求不存在的 qzeros 张量。）
        """

        class _FakeHF:
            quantization_config = {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
                "desc_act": False,
                "sym": True,
            }

        cfg = QuantConfig.from_hf_config(_FakeHF(), device_capability=(8, 6))
        m = Int4Method(
            cfg, kind="column", out_size=OUT, in_size=IN, tp_size=1, tp_rank=0
        )
        assert m.has_zeros is False
        assert m.zero_bias == 8.0
        layer = _Layer()
        m.create_weights(layer)
        assert layer.qzeros is None

    def test_gptq_asym_from_hf_config_has_zeros(self):
        class _FakeHF:
            quantization_config = {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
                "sym": False,
            }

        cfg = QuantConfig.from_hf_config(_FakeHF(), device_capability=(8, 6))
        m = Int4Method(
            cfg, kind="column", out_size=OUT, in_size=IN, tp_size=1, tp_rank=0
        )
        assert m.has_zeros is True
        assert m.zero_bias == 0.0
