"""模型级量化加载集成测试（CPU，无需 GPU）。

用 transformers 生成小模型 checkpoint（bf16 safetensors），再走引擎的
``load_model`` 完整链路（load_weights 路由 + Fp8FallbackCatcher +
finalize 在线量化），验证：
- 量化层参数变为 fp8 + scale 非零；
- embed/lm_head/router 等高精度模块不受影响；
- 权重绑定（tie_word_embeddings）仍生效；
- MoE 模型（含 shared expert / router）的量化加载。
"""

from __future__ import annotations

import os

import pytest
import torch

FP8 = torch.float8_e4m3fn

_HIDDEN = 256
_INTER = 512
_VOCAB = 256
_LAYERS = 2


@pytest.fixture(scope="module", autouse=True)
def _init_single_proc_dist():
    """模型构造依赖 TP/PP 进程组：单进程 gloo 初始化一次。"""
    import torch.distributed as dist

    from src.distributed import init_distributed_environment, initialize_model_parallel

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="gloo",
            init_method="tcp://127.0.0.1:29517",
        )
    initialize_model_parallel(1, 1, 0, 0)
    yield


def _write_dense_checkpoint(tmp_path, *, tied: bool = True) -> str:
    """transformers Qwen3ForCausalLM 生成 bf16 checkpoint。"""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        hidden_size=_HIDDEN,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=_INTER,
        num_hidden_layers=_LAYERS,
        vocab_size=_VOCAB,
        max_position_embeddings=512,
        tie_word_embeddings=tied,
        torch_dtype=torch.bfloat16,
    )
    model = Qwen3ForCausalLM(cfg)
    d = tmp_path / "dense"
    model.save_pretrained(d)
    return str(d)


def _write_moe_checkpoint(tmp_path) -> str:
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    cfg = Qwen3MoeConfig(
        hidden_size=_HIDDEN,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=_INTER,
        num_hidden_layers=_LAYERS,
        vocab_size=_VOCAB,
        max_position_embeddings=512,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=512,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        torch_dtype=torch.bfloat16,
    )
    model = Qwen3MoeForCausalLM(cfg)
    d = tmp_path / "moe"
    model.save_pretrained(d)
    # transformers 5.x 的 Qwen3Moe 实现不含 shared expert；真实 HF checkpoint
    # （Qwen3-30B-A3B）有，按真实命名补齐，保证 shared expert 加载路径被覆盖。
    from glob import glob

    from safetensors.torch import load_file, save_file

    gen = torch.Generator().manual_seed(7)
    for f in glob(str(d / "*.safetensors")):
        tensors = load_file(f)
        for i in range(_LAYERS):
            p = f"model.layers.{i}.mlp."
            tensors[p + "shared_expert.gate_proj.weight"] = torch.randn(
                512, _HIDDEN, dtype=torch.bfloat16, generator=gen
            )
            tensors[p + "shared_expert.up_proj.weight"] = torch.randn(
                512, _HIDDEN, dtype=torch.bfloat16, generator=gen
            )
            tensors[p + "shared_expert.down_proj.weight"] = torch.randn(
                _HIDDEN, 512, dtype=torch.bfloat16, generator=gen
            )
            tensors[p + "shared_expert_gate.weight"] = torch.randn(
                1, _HIDDEN, dtype=torch.bfloat16, generator=gen
            )
        save_file(tensors, f)
    return str(d)


def _load_engine_model(path: str, *, moe: bool):
    """引擎侧模型 + load_model 完整链路。"""
    from transformers import AutoConfig

    from src.compute.models.qwen3 import Qwen3ForCausalLM
    from src.compute.models.qwen3_moe import Qwen3MoeForCausalLM
    from src.moe.parallel import MoEParallelConfig
    from src.quant.config import QuantConfig
    from utils.model_loader import default_dtype_context, load_model

    quant = QuantConfig.from_name(
        "fp8", device_capability=(8, 9), quantize_at_load=True
    )
    hf_cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    with default_dtype_context(torch.bfloat16):
        if moe:
            parallel = MoEParallelConfig.from_engine(
                tp_size=1, tp_rank=0, enable_ep=False, fused_backend="triton"
            )
            model = Qwen3MoeForCausalLM(hf_cfg, parallel=parallel, quant=quant)
        else:
            model = Qwen3ForCausalLM(hf_cfg, quant=quant)
    load_model(model, path)
    return model


class TestDenseOnlineFp8:
    def test_quantized_params_and_untouched_modules(self, tmp_path):
        path = _write_dense_checkpoint(tmp_path)
        model = _load_engine_model(path, moe=False)
        p = dict(model.named_parameters())

        # 量化层：fp8 权重 + 非零 scale
        for name in (
            "model.layers.0.self_attn.qkv_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.1.mlp.gate_up_proj.weight",
        ):
            assert p[name].dtype == FP8, name
        for name in (
            "model.layers.0.self_attn.qkv_proj.weight_scale",
            "model.layers.0.mlp.down_proj.weight_scale",
        ):
            assert p[name].dtype == torch.float32, name
            assert p[name].abs().sum() > 0, name

        # 不量化模块保持 bf16
        assert p["model.embed_tokens.weight"].dtype == torch.bfloat16
        assert p["model.layers.0.input_layernorm.weight"].dtype == torch.bfloat16
        assert p["model.layers.0.self_attn.q_norm.weight"].dtype == torch.bfloat16

        # scale 网格形状：qkv [8, 2]（512/128 × 256/128）
        assert p["model.layers.0.self_attn.qkv_proj.weight_scale"].shape == (8, 2)
        # gate_up 融合后 2×intermediate=1024 行 → [8, 2]；down [2, 4]
        assert p["model.layers.0.mlp.gate_up_proj.weight_scale"].shape == (8, 2)
        assert p["model.layers.0.mlp.down_proj.weight_scale"].shape == (2, 4)

        # staging 已释放
        assert model.model.layers[0].mlp.gate_up_proj._quant_staging is None

    def test_tied_lm_head_binding(self, tmp_path):
        path = _write_dense_checkpoint(tmp_path, tied=True)
        model = _load_engine_model(path, moe=False)
        p = dict(model.named_parameters())
        # lm_head 与 embed 绑定（skip padding 行）
        assert torch.equal(
            p["lm_head.weight"].data,
            p["model.embed_tokens.weight"].data[1:],
        )


class TestMoeOnlineFp8:
    def test_experts_and_shared(self, tmp_path):
        path = _write_moe_checkpoint(tmp_path)
        model = _load_engine_model(path, moe=True)
        p = dict(model.named_parameters())

        # 专家：fp8 + scale
        w13 = "model.layers.1.mlp.experts.w13_weight"
        assert p[w13].dtype == FP8
        experts = model.model.layers[1].mlp.experts
        assert experts.w13_scale.abs().sum() > 0
        assert experts.w2_scale.abs().sum() > 0
        # w13 [E, 2i, h] = [4, 512, 256]；scale [4, 4, 2]
        assert p[w13].shape == (4, 512, 256)
        assert experts.w13_scale.shape == (4, 4, 2)

        # router / shared_expert_gate 不量化；shared gate_up/down 量化
        assert p["model.layers.1.mlp.gate.weight"].dtype == torch.bfloat16
        assert (
            p["model.layers.1.mlp.shared_expert_gate.weight"].dtype == torch.bfloat16
        )
        assert (
            p["model.layers.1.mlp.shared_expert_gate_up.weight"].dtype == FP8
        )
        assert p["model.layers.1.mlp.shared_expert_down.weight"].dtype == FP8

        # dense 层（layer 0 非 MoE 层时）不受影响——此处 step=1 全 MoE，
        # 验证第 0 层的 attention 仍量化
        assert p["model.layers.0.self_attn.qkv_proj.weight"].dtype == FP8
