"""INT4 模型级加载集成测试（CPU，无需 GPU）。

用 transformers 生成小模型 bf16 checkpoint，再按 AWQ 格式离线量化打包
（qweight/scales/qzeros 三件套），走引擎的 ``load_model`` 全链路，验证：

- ``load_weights`` 的名字路由正确（q_proj.qweight → 融合层 qkv_proj.qweight 等）；
- 融合层（qkv / gate_up）的三段拼接与 TP 切分语义正确；
- 反量化结果逼近原 bf16 权重（INT4 group=128 的量化误差范围内）。
"""

from __future__ import annotations

import pytest
import torch

from tests.unit.test_quant_int4 import (  # noqa: E402  复用参考量化器
    GROUP,
    _awq_tensors,
    quantize_int4_asym,
    rel_err,
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
        os.environ.setdefault("MASTER_PORT", "29519")
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="gloo",
            init_method="tcp://127.0.0.1:29519",
        )
    initialize_model_parallel(1, 1, 0, 0)
    yield


def _write_awq_checkpoint(tmp_path) -> str:
    """transformers 模型 → 逐 Linear 量化打包成 AWQ checkpoint。

    同时把量化前的 bf16 权重存到 ``originals.safetensors``，供测试做
    数值基准（不能另建一个模型当参考——随机初始化不同）。
    """
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
        head_dim=64,  # Qwen3Config 默认 128，这里显式收窄以对齐测试维度
        torch_dtype=torch.bfloat16,
    )
    model = Qwen3ForCausalLM(cfg)
    d = tmp_path / "awq"
    model.save_pretrained(d)

    targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    from src.quant.kernels.int4_gemm import pack_int4_rows

    originals: dict[str, torch.Tensor] = {}
    for f in glob(str(d / "*.safetensors")):
        tensors = load_file(f)
        add: dict[str, torch.Tensor] = {}
        for name in list(tensors):
            if not name.endswith(".weight") or "." not in name:
                continue
            if not any(name.endswith(f"{t}.weight") for t in targets):
                continue
            w = tensors[name].float()  # [out, in] → 量化需 [in, out]
            originals[name] = tensors[name]
            q, s, z = quantize_int4_asym(w.t().contiguous())
            base = name[: -len(".weight")]
            # AWQ 格式：qweight 与 qzeros 的输出通道都带 [0,2,4,6,1,3,5,7] 交织
            add[base + ".qweight"] = pack_int4_rows(q, interleave=True)
            add[base + ".scales"] = s.to(torch.float16)
            add[base + ".qzeros"] = pack_int4_rows(z, interleave=True)
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
        method="awq",
        bits=4,
        weight_scale="group",
        act_scale=None,
        group_size=GROUP,
        zero_point=True,
    )
    hf_cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    with default_dtype_context(torch.bfloat16):
        model = Qwen3ForCausalLM(hf_cfg, quant=cfg)
    load_model(model, path)
    return model


class TestInt4ModelLoad:
    def test_params_and_shapes(self, tmp_path):
        path = _write_awq_checkpoint(tmp_path)
        model = _load_engine_model(path)
        p = dict(model.named_parameters())

        # 量化层：int32 qweight + fp16 scales + uint8 qzeros
        for name in (
            "model.layers.0.self_attn.qkv_proj.qweight",
            "model.layers.0.self_attn.o_proj.qweight",
            "model.layers.0.mlp.gate_up_proj.qweight",
            "model.layers.1.mlp.down_proj.qweight",
        ):
            assert p[name].dtype == torch.int32, name
        assert p["model.layers.0.self_attn.qkv_proj.scales"].dtype == torch.float16
        assert p["model.layers.0.self_attn.qkv_proj.qzeros"].dtype == torch.uint8

        # 融合层形状：qkv 本地 out = q(256) + k(128) + v(128) = 512
        assert p["model.layers.0.self_attn.qkv_proj.qweight"].shape == (_HIDDEN, 512 // 8)
        assert p["model.layers.0.self_attn.qkv_proj.scales"].shape == (2, 512)
        # gate_up: 2 × inter = 1024
        assert p["model.layers.0.mlp.gate_up_proj.qweight"].shape == (_HIDDEN, 1024 // 8)
        # down: in=inter=512, out=hidden=256
        assert p["model.layers.0.mlp.down_proj.qweight"].shape == (_INTER, _HIDDEN // 8)

        # embed / lm_head 保持 bf16（AWQ 不量化它们）
        assert p["model.embed_tokens.weight"].dtype == torch.bfloat16
        assert p["model.layers.0.input_layernorm.weight"].dtype == torch.bfloat16

    def test_dequant_error_within_int4_bound(self, tmp_path):
        """反量化结果逼近原 bf16 权重（验证路由与段拼接的数值正确性）。

        基准取 writer 保存的原始 bf16 权重——不能另建模型当参考
        （随机初始化不同，比较结果无意义）。
        """
        from safetensors.torch import load_file

        from src.quant.kernels.int4_gemm import dequantize_int4_reference

        path = _write_awq_checkpoint(tmp_path)
        model = _load_engine_model(path)
        p = dict(model.named_parameters())
        sd = load_file(str(tmp_path / "originals.safetensors"))

        # qkv：本地布局 [q | k | v]，HF 侧来自三个独立权重
        deq = dequantize_int4_reference(
            p["model.layers.0.self_attn.qkv_proj.qweight"].data,
            p["model.layers.0.self_attn.qkv_proj.scales"].data,
            p["model.layers.0.self_attn.qkv_proj.qzeros"].data,
            group_size=GROUP,
        )  # [in, out]
        ref_q = sd["model.layers.0.self_attn.q_proj.weight"].float()
        ref_k = sd["model.layers.0.self_attn.k_proj.weight"].float()
        ref_v = sd["model.layers.0.self_attn.v_proj.weight"].float()
        ref_qkv = torch.cat([ref_q, ref_k, ref_v], dim=0)  # [512, in]
        assert rel_err(deq.t(), ref_qkv) < 0.12

        # gate_up：HF 侧 [inter, hidden] × 2
        deq_gu = dequantize_int4_reference(
            p["model.layers.0.mlp.gate_up_proj.qweight"].data,
            p["model.layers.0.mlp.gate_up_proj.scales"].data,
            p["model.layers.0.mlp.gate_up_proj.qzeros"].data,
            group_size=GROUP,
        )
        ref_gu = torch.cat(
            [sd["model.layers.0.mlp.gate_proj.weight"], sd["model.layers.0.mlp.up_proj.weight"]],
            dim=0,
        ).float()
        assert rel_err(deq_gu.t(), ref_gu) < 0.12

        # down：行并行，输入维即量化组维
        deq_d = dequantize_int4_reference(
            p["model.layers.0.mlp.down_proj.qweight"].data,
            p["model.layers.0.mlp.down_proj.scales"].data,
            p["model.layers.0.mlp.down_proj.qzeros"].data,
            group_size=GROUP,
        )
        assert rel_err(
            deq_d.t(), sd["model.layers.0.mlp.down_proj.weight"].float()
        ) < 0.12

    def test_tied_lm_head_binding(self, tmp_path):
        path = _write_awq_checkpoint(tmp_path)
        model = _load_engine_model(path)
        p = dict(model.named_parameters())
        assert torch.equal(
            p["lm_head.weight"].data, p["model.embed_tokens.weight"].data[1:]
        )
