"""QuantConfig / EngineConfig 量化解析的 CPU 单元测试（无需 GPU）。

覆盖：
- HF quantization_config 四类格式（fp8 block / awq / gptq / int8）解析
- 显式指定（from_name）与 auto 语义
- 硬件门控（SM 矩阵）与 (0,0) 未知放行
- modules_to_not_convert 过滤
- EngineConfig 端到端解析（临时模型目录）
"""

from __future__ import annotations

import json

import pytest

from src.quant import QuantConfig, supports_device
from src.quant.config import VALID_METHODS

# ------------------------------------------------------------------ #
# HF quantization_config 解析
# ------------------------------------------------------------------ #

FP8_BLOCK = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}
FP8_TENSOR = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic"}
AWQ = {
    "bits": 4,
    "group_size": 128,
    "modules_to_not_convert": None,
    "quant_method": "awq",
    "version": "gemm",
    "zero_point": True,
}
GPTQ = {
    "quant_method": "gptq",
    "bits": 4,
    "group_size": 128,
    "desc_act": False,
    "sym": True,
    "checkpoint_format": "gptq",
    "exllama_config": {"version": 2},
}


class _FakeHF:
    """最小 HF config 载体（仅需 quantization_config 属性）。"""

    def __init__(self, qc):
        self.quantization_config = qc


class TestFromHfConfig:
    def test_fp8_block(self):
        cfg = QuantConfig.from_hf_config(
            _FakeHF(FP8_BLOCK), device_capability=(8, 9)
        )
        assert cfg is not None
        assert cfg.method == "fp8"
        assert cfg.weight_scale == "per_block_128x128"
        assert cfg.act_scale == "dynamic"
        assert cfg.block_size == (128, 128)
        assert cfg.is_w8a8 and not cfg.is_weight_only
        assert cfg.kv_cache_dtype is None

    def test_fp8_per_tensor(self):
        cfg = QuantConfig.from_hf_config(
            _FakeHF(FP8_TENSOR), device_capability=(8, 9)
        )
        assert cfg.weight_scale == "per_tensor"
        assert cfg.block_size is None

    def test_fp8_kv_dtype_override(self):
        import torch

        cfg = QuantConfig.from_hf_config(
            _FakeHF(FP8_BLOCK),
            device_capability=(8, 9),
            kv_cache_dtype=torch.float8_e4m3fn,
        )
        assert cfg.kv_cache_dtype is torch.float8_e4m3fn

    def test_awq(self):
        cfg = QuantConfig.from_hf_config(_FakeHF(AWQ), device_capability=(8, 6))
        assert cfg.method == "awq"
        assert cfg.bits == 4
        assert cfg.group_size == 128
        assert cfg.zero_point is True
        assert cfg.version == "gemm"
        assert cfg.act_scale is None
        assert cfg.is_weight_only

    def test_gptq(self):
        cfg = QuantConfig.from_hf_config(_FakeHF(GPTQ), device_capability=(8, 6))
        assert cfg.method == "gptq"
        assert cfg.sym is True
        assert cfg.desc_act is False
        assert cfg.group_size == 128
        assert cfg.is_weight_only

    def test_no_quantization_config(self):
        assert QuantConfig.from_hf_config(_FakeHF(None), device_capability=(8, 9)) is None

    def test_unsupported_method(self):
        with pytest.raises(ValueError, match="unsupported quant_method"):
            QuantConfig.from_hf_config(
                _FakeHF({"quant_method": "marlin"}), device_capability=(8, 9)
            )

    def test_unsupported_block_size(self):
        with pytest.raises(ValueError, match="weight_block_size"):
            QuantConfig.from_hf_config(
                _FakeHF({"quant_method": "fp8", "weight_block_size": [64, 64]}),
                device_capability=(8, 9),
            )

    def test_hardware_gate_fp8_on_sm86(self):
        with pytest.raises(ValueError, match="not supported on SM"):
            QuantConfig.from_hf_config(
                _FakeHF(FP8_BLOCK), device_capability=(8, 6)
            )

    def test_int4_ok_on_sm86(self):
        cfg = QuantConfig.from_hf_config(_FakeHF(AWQ), device_capability=(8, 6))
        assert cfg.method == "awq"


# ------------------------------------------------------------------ #
# 显式指定 / auto
# ------------------------------------------------------------------ #


class TestFromName:
    def test_none(self):
        assert QuantConfig.from_name("none", device_capability=(8, 9)) is None

    def test_auto_without_config(self):
        assert (
            QuantConfig.from_name("auto", device_capability=(8, 9)) is None
        )

    def test_auto_with_config(self):
        cfg = QuantConfig.from_name(
            "auto", hf_cfg=_FakeHF(FP8_BLOCK), device_capability=(8, 9)
        )
        assert cfg.method == "fp8"
        assert cfg.weight_scale == "per_block_128x128"

    def test_fp8_online(self):
        cfg = QuantConfig.from_name(
            "fp8", device_capability=(8, 9), quantize_at_load=True
        )
        assert cfg.method == "fp8"
        assert cfg.quantize_at_load is True
        assert cfg.weight_scale == "per_block_128x128"

    def test_int8_online(self):
        cfg = QuantConfig.from_name(
            "int8", device_capability=(8, 6), quantize_at_load=True
        )
        assert cfg.method == "int8"
        assert cfg.weight_scale == "per_channel"

    def test_int4_requires_checkpoint(self):
        with pytest.raises(ValueError, match="pre-quantized"):
            QuantConfig.from_name("awq", device_capability=(8, 6))

    def test_int4_with_matching_checkpoint(self):
        cfg = QuantConfig.from_name(
            "awq", hf_cfg=_FakeHF(AWQ), device_capability=(8, 6)
        )
        assert cfg.method == "awq"
        assert cfg.group_size == 128

    def test_int4_with_mismatched_checkpoint(self):
        with pytest.raises(ValueError, match="requested awq"):
            QuantConfig.from_name(
                "awq", hf_cfg=_FakeHF(FP8_BLOCK), device_capability=(8, 9)
            )

    def test_invalid_name(self):
        with pytest.raises(ValueError, match="invalid quantization"):
            QuantConfig.from_name("bf16", device_capability=(8, 9))

    def test_fp8_rejected_on_sm86(self):
        with pytest.raises(ValueError, match="not supported on SM"):
            QuantConfig.from_name("fp8", device_capability=(8, 6))


# ------------------------------------------------------------------ #
# 硬件矩阵与过滤
# ------------------------------------------------------------------ #


class TestSupportMatrix:
    @pytest.mark.parametrize(
        "method,cap,ok",
        [
            ("fp8", (8, 9), True),
            ("fp8", (9, 0), True),
            ("fp8", (8, 6), False),
            ("fp8", (10, 0), True),
            ("int8", (8, 6), True),
            ("int8", (7, 5), True),
            ("awq", (8, 6), True),
            ("gptq", (7, 5), True),
            ("fp8", (0, 0), True),  # 未知环境放行（CPU 单测）
        ],
    )
    def test_supports_device(self, method, cap, ok):
        assert supports_device(method, cap) is ok

    def test_valid_methods(self):
        assert VALID_METHODS == ("none", "auto", "fp8", "int8", "awq", "gptq")


class TestSkipModule:
    def test_skip_by_substring(self):
        cfg = QuantConfig.from_hf_config(
            _FakeHF({**FP8_BLOCK, "modules_to_not_convert": ["lm_head", "mlp.gate"]}),
            device_capability=(8, 9),
        )
        assert cfg.skip_module("model.layers.3.mlp.gate")
        assert cfg.skip_module("lm_head.weight")
        assert not cfg.skip_module("model.layers.3.mlp.experts.w13_weight")
        assert not cfg.skip_module("model.layers.3.self_attn.qkv_proj")

    def test_empty_list(self):
        cfg = QuantConfig.from_name("fp8", device_capability=(8, 9))
        assert not cfg.skip_module("anything")


# ------------------------------------------------------------------ #
# EngineConfig 集成（临时模型目录）
# ------------------------------------------------------------------ #

_MINIMAL_QWEN3 = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 256,
    "torch_dtype": "bfloat16",
}


def _write_model(tmp_path, **overrides):
    cfg = dict(_MINIMAL_QWEN3)
    cfg.update(overrides)
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return str(d)


class TestEngineConfigQuant:
    # EngineConfig 的量化解析逻辑应与本地 GPU 解耦：统一按 SM89 解析，
    # 真实硬件门控由 TestSupportMatrix / from_hf_config 的测试覆盖。
    @pytest.fixture(autouse=True)
    def _force_sm89(self, monkeypatch):
        from src.config import EngineConfig

        monkeypatch.setattr(
            EngineConfig,
            "_device_capability",
            staticmethod(lambda: (8, 9)),
        )

    def test_default_none(self, tmp_path):
        from src.config import EngineConfig

        ec = EngineConfig(model=_write_model(tmp_path))
        assert ec.quant_config is None

    def test_auto_picks_hf_config(self, tmp_path):
        from src.config import EngineConfig

        path = _write_model(tmp_path, quantization_config=FP8_BLOCK)
        ec = EngineConfig(model=path, quantization="auto")
        assert ec.quant_config is not None
        assert ec.quant_config.method == "fp8"
        assert ec.quant_config.weight_scale == "per_block_128x128"

    def test_explicit_fp8_online(self, tmp_path):
        from src.config import EngineConfig

        ec = EngineConfig(model=_write_model(tmp_path), quantization="fp8")
        assert ec.quant_config is not None
        assert ec.quant_config.method == "fp8"
        assert ec.quant_config.quantize_at_load is True

    def test_explicit_fp8_respects_checkpoint(self, tmp_path):
        from src.config import EngineConfig

        path = _write_model(tmp_path, quantization_config=FP8_BLOCK)
        ec = EngineConfig(model=path, quantization="fp8")
        # checkpoint 已是 fp8 block → 按其格式，不再标记在线量化
        assert ec.quant_config.weight_scale == "per_block_128x128"
        assert ec.quant_config.quantize_at_load is False

    def test_int4_without_checkpoint_raises(self, tmp_path):
        from src.config import EngineConfig

        with pytest.raises(ValueError, match="pre-quantized"):
            EngineConfig(model=_write_model(tmp_path), quantization="awq")

    def test_int4_with_checkpoint(self, tmp_path):
        from src.config import EngineConfig

        path = _write_model(tmp_path, quantization_config=AWQ)
        ec = EngineConfig(model=path, quantization="awq")
        assert ec.quant_config.method == "awq"
        assert ec.quant_config.group_size == 128

    def test_invalid_quantization_name(self, tmp_path):
        from src.config import EngineConfig

        with pytest.raises(ValueError, match="quantization must be"):
            EngineConfig(model=_write_model(tmp_path), quantization="nf4")

    def test_kv_cache_flag(self, tmp_path):
        """quant_kv_cache=True → kv_cache_dtype 解析为 fp8（SM 门控在 Worker）。"""
        import torch

        from src.config import EngineConfig

        path = _write_model(tmp_path, quantization_config=FP8_BLOCK)
        ec = EngineConfig(model=path, quantization="auto", quant_kv_cache=True)
        assert ec.quant_config.kv_cache_dtype is torch.float8_e4m3fn

    def test_kv_cache_dtype_parsed(self, tmp_path):
        """QuantConfig 层支持 kv_cache_dtype 透传。"""
        import torch

        from src.quant import QuantConfig

        cfg = QuantConfig.from_hf_config(
            _FakeHF(FP8_BLOCK),
            device_capability=(8, 9),
            kv_cache_dtype=torch.float8_e4m3fn,
        )
        assert cfg.kv_cache_dtype is torch.float8_e4m3fn

    def test_fp8_refused_on_real_device(self, tmp_path, monkeypatch):
        """真实硬件门控：本机 3060（SM86）上 fp8 必须显式拒绝（有 CUDA 才跑）。"""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("no CUDA on this machine")
        cap = torch.cuda.get_device_capability()
        if cap >= (8, 9):
            pytest.skip(f"device SM{cap[0]}{cap[1]} supports fp8 natively")
        from src.config import EngineConfig

        monkeypatch.undo()  # 恢复真实 _device_capability
        with pytest.raises(ValueError, match="not supported on SM"):
            EngineConfig(model=_write_model(tmp_path), quantization="fp8")
