"""INT8 模型级加载集成测试（CPU，无需 GPU）。

用 transformers 生成小模型 bf16 checkpoint，逐 Linear 量化打包成 int8
（per-channel 权重 + fp32 scale），走引擎 ``load_model`` 全链路，验证：

- ``load_weights`` 的名字路由（q_proj.weight → 融合层 qkv_proj.weight 等）；
- 融合层（qkv / gate_up）三段拼接；
- 反量化结果逼近原 bf16 权重（per-channel int8 误差范围内）；
- embed / lm_head 保持 bf16（int8 不量化它们）；
- SmoothQuant 的 ``input_scale`` 张量触发显式 NotImplementedError。
"""

from __future__ import annotations

import pytest
import torch

from src.quant.kernels.int8_gemm import (
    dequantize_per_channel,
    quantize_per_channel,
)

_HIDDEN = 256
_INTER = 512
_VOCAB = 256
_LAYERS = 2


@pytest.fixture(scope="module", autouse=True)
def _init_single_proc_dist():
    """模型构造依赖 TP/PP 进程组：单进程 gloo 初始化一次。"""
    import os

    import torch.distributed as dist

    from src.distributed import init_distributed_environment, initialize_model_parallel

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29521")
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="gloo",
            init_method="tcp://127.0.0.1:29521",
        )
    initialize_model_parallel(1, 1, 0, 0)
    yield


def _write_int8_checkpoint(tmp_path, *, smoothquant: bool = False) -> str:
    """transformers 模型 → 逐 Linear per-channel 量化成 int8 checkpoint。"""
    from glob import glob

    from safetensors.torch import load_file, save_file
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        hidden_size=_HIDDEN,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=_INTER,
        num_hidden_layers=_LAYERS,
        vocab_size=_VOCAB,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        head_dim=64,  # Qwen3Config 默认 128，显式收窄以对齐测试维度
        torch_dtype=torch.bfloat16,
    )
    model = Qwen3ForCausalLM(cfg)
    d = tmp_path / "int8"
    model.save_pretrained(d)

    targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    originals: dict[str, torch.Tensor] = {}
    for f in glob(str(d / "*.safetensors")):
        tensors = load_file(f)
        add: dict[str, torch.Tensor] = {}
        for name in list(tensors):
            if not name.endswith(".weight") or "." not in name:
                continue
            if not any(name.endswith(f"{t}.weight") for t in targets):
                continue
            w = tensors[name].float()
            originals[name] = tensors[name]
            q, s = quantize_per_channel(w)
            base = name[: -len(".weight")]
            add[base + ".weight"] = q
            add[base + ".weight_scale"] = s
            if smoothquant:
                add[base + ".input_scale"] = torch.ones(_HIDDEN, dtype=torch.float32)
            del tensors[name]
        tensors.update(add)
        save_file(tensors, f)
    save_file(originals, str(tmp_path / "originals.safetensors"))
    return str(d)


def _load_engine_model(path: str):
    from transformers import AutoConfig

    from src.compute.models.qwen3 import Qwen3ForCausalLM
    from src.quant.config import QuantConfig
    from utils.model_loader import default_dtype_context, load_model

    cfg = QuantConfig(
        method="int8",
        bits=8,
        weight_scale="per_channel",
        act_scale="dynamic",
    )
    hf_cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    with default_dtype_context(torch.bfloat16):
        model = Qwen3ForCausalLM(hf_cfg, quant=cfg)
    load_model(model, path)
    return model


class TestInt8ModelLoad:
    def test_params_and_shapes(self, tmp_path):
        path = _write_int8_checkpoint(tmp_path)
        model = _load_engine_model(path)
        p = dict(model.named_parameters())

        for name in (
            "model.layers.0.self_attn.qkv_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
        ):
            assert p[name].dtype == torch.int8, name
        # per-channel scale：[out]
        assert p["model.layers.0.self_attn.qkv_proj.weight_scale"].shape == (512,)
        assert p["model.layers.0.self_attn.o_proj.weight_scale"].shape == (256,)
        assert p["model.layers.0.mlp.down_proj.weight_scale"].shape == (256,)
        assert p["model.layers.0.self_attn.qkv_proj.weight"].shape == (512, 256)

        # embed / lm_head 保持 bf16
        assert p["model.embed_tokens.weight"].dtype == torch.bfloat16
        assert p["model.layers.0.input_layernorm.weight"].dtype == torch.bfloat16

    def test_dequant_error_within_int8_bound(self, tmp_path):
        """反量化结果逼近原 bf16 权重（验证路由与段拼接的数值正确性）。"""
        from safetensors.torch import load_file

        path = _write_int8_checkpoint(tmp_path)
        model = _load_engine_model(path)
        p = dict(model.named_parameters())
        sd = load_file(str(tmp_path / "originals.safetensors"))

        deq = dequantize_per_channel(
            p["model.layers.0.self_attn.qkv_proj.weight"].data,
            p["model.layers.0.self_attn.qkv_proj.weight_scale"].data,
        )
        ref_qkv = torch.cat(
            [
                sd["model.layers.0.self_attn.q_proj.weight"],
                sd["model.layers.0.self_attn.k_proj.weight"],
                sd["model.layers.0.self_attn.v_proj.weight"],
            ],
            dim=0,
        ).float()
        rel = (deq - ref_qkv).abs().max() / ref_qkv.abs().max()
        assert rel < 0.02, f"qkv rel={rel:.4f}"

        deq_gu = dequantize_per_channel(
            p["model.layers.0.mlp.gate_up_proj.weight"].data,
            p["model.layers.0.mlp.gate_up_proj.weight_scale"].data,
        )
        ref_gu = torch.cat(
            [
                sd["model.layers.0.mlp.gate_proj.weight"],
                sd["model.layers.0.mlp.up_proj.weight"],
            ],
            dim=0,
        ).float()
        rel = (deq_gu - ref_gu).abs().max() / ref_gu.abs().max()
        assert rel < 0.02, f"gate_up rel={rel:.4f}"

    def test_tied_lm_head_binding(self, tmp_path):
        path = _write_int8_checkpoint(tmp_path)
        model = _load_engine_model(path)
        p = dict(model.named_parameters())
        assert torch.equal(
            p["lm_head.weight"].data, p["model.embed_tokens.weight"].data[1:]
        )

    def test_smoothquant_input_scale_raises(self, tmp_path):
        path = _write_int8_checkpoint(tmp_path, smoothquant=True)
        with pytest.raises(NotImplementedError, match="input_scale"):
            _load_engine_model(path)
