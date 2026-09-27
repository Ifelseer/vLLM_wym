from __future__ import annotations

import re
from collections.abc import Iterable

import torch
from torch import nn
from transformers import PretrainedConfig

from ...distributed import PPMissingLayer, get_pp_group, get_pp_indices
from ...moe import SparseMoeBlock
from ...moe.parallel import MoEParallelConfig
from ..layers import RMSNorm, ParallelLMHead, VocabParallelEmbedding
from ..layers.attention import AttnInputs
from ..layers.linear import get_weight_loader
from .qwen3 import (
    Qwen3Attention,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3Model,
    _parse_layer_idx,
)


def _is_moe_layer(config: PretrainedConfig, layer_idx: int) -> bool:
    mlp_only = list(getattr(config, "mlp_only_layers", None) or [])
    if layer_idx in mlp_only:
        return False
    num_experts = int(getattr(config, "num_experts", 0) or 0)
    if num_experts <= 0:
        return False
    step = int(getattr(config, "decoder_sparse_step", 1) or 1)
    return (layer_idx + 1) % step == 0


def _use_qk_norm(config: PretrainedConfig) -> bool:
    mt = getattr(config, "model_type", "") or ""
    return mt != "qwen2_moe"


def _qkv_bias(config: PretrainedConfig) -> bool:
    if getattr(config, "attention_bias", None) is not None:
        return bool(config.attention_bias)
    return (getattr(config, "model_type", "") or "") == "qwen2_moe"


def _rope_theta(config: PretrainedConfig) -> float:
    rp = getattr(config, "rope_parameters", None)
    if rp is not None:
        theta = rp["rope_theta"] if isinstance(rp, dict) else getattr(rp, "rope_theta", None)
        if theta is not None:
            return float(theta)
    return float(getattr(config, "rope_theta", 1000000.0))


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        *,
        layer_id: int,
        global_layer_id: int,
        parallel: MoEParallelConfig,
    ):
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rope_theta=_rope_theta(config),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=_qkv_bias(config),
            head_dim=getattr(config, "head_dim", None),
            layer_id=layer_id,
            use_qk_norm=_use_qk_norm(config),
        )
        if _is_moe_layer(config, global_layer_id):
            self.mlp = SparseMoeBlock(config, parallel)
        else:
            self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, positions, hidden_states, residual, attn: AttnInputs):
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, attn)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class Qwen3MoeModel(Qwen3Model):
    def __init__(self, config: PretrainedConfig, parallel: MoEParallelConfig):
        nn.Module.__init__(self)
        pp = get_pp_group()
        self.start_layer, self.end_layer = get_pp_indices(
            config.num_hidden_layers, pp.group_rank, pp.size
        )
        self.embed_tokens = (
            VocabParallelEmbedding(config.vocab_size, config.hidden_size)
            if pp.is_first_rank
            else PPMissingLayer()
        )
        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(
                    config,
                    layer_id=local_i,
                    global_layer_id=self.start_layer + local_i,
                    parallel=parallel,
                )
                for local_i in range(self.end_layer - self.start_layer)
            ]
        )
        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if pp.is_last_rank
            else PPMissingLayer()
        )


class Qwen3MoeForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config: PretrainedConfig, parallel: MoEParallelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.parallel = parallel
        self.model = Qwen3MoeModel(config, parallel)
        self.lm_head = (
            ParallelLMHead(config.hidden_size, config.vocab_size, bias=False)
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )

    def _experts_module(self, weight_name: str):
        # load_weights 已把全局层号改成当前 PP stage 的局部下标。
        m = re.search(r"layers\.(\d+)\.mlp\.experts", weight_name)
        if not m:
            return None
        local_id = int(m.group(1))
        if local_id < 0 or local_id >= len(self.model.layers):
            return None
        return getattr(self.model.layers[local_id].mlp, "experts", None)

    def _load_expert_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        loaded: set[str],
        coverage: dict[str, set[tuple[int, str]]],
    ) -> bool:
        m = re.search(
            r"(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$",
            name,
        )
        if not m:
            return False
        prefix, expert_id_s, proj = m.group(1), m.group(2), m.group(3)
        expert_id = int(expert_id_s)
        experts = self._experts_module(name)
        if experts is None:
            return True
        if experts._local_id(expert_id) is None:
            return True
        coverage.setdefault(prefix, set()).add((expert_id, proj))
        if proj in ("gate_proj", "up_proj"):
            experts.weight_loader_w13(
                experts.w13_weight, loaded_weight, f"{proj}.weight", expert_id
            )
            loaded.add(f"{prefix}.w13_weight")
        else:
            experts.weight_loader_w2(
                experts.w2_weight, loaded_weight, f"{proj}.weight", expert_id
            )
            loaded.add(f"{prefix}.w2_weight")
        return True

    def _load_fused_expert_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        loaded: set[str],
        fused_coverage: dict[str, set[str]],
    ) -> bool:
        is_w13 = ".mlp.experts.gate_up_proj" in name and loaded_weight.dim() == 3
        is_w2 = (
            ".mlp.experts.down_proj" in name
            and loaded_weight.dim() == 3
            and not re.search(r"experts\.\d+\.", name)
        )
        if not (is_w13 or is_w2):
            return False
        marker = ".gate_up_proj" if is_w13 else ".down_proj"
        prefix = name.split(marker, 1)[0]
        experts = self._experts_module(name)
        if experts is None:
            return True
        param_name = f"{prefix}.{'w13_weight' if is_w13 else 'w2_weight'}"
        if is_w13:
            experts.load_fused_w13(loaded_weight)
            kind = "w13"
        else:
            experts.load_fused_w2(loaded_weight)
            kind = "w2"
        loaded.add(param_name)
        fused_coverage.setdefault(prefix, set()).add(kind)
        return True

    def _assert_expert_coverage(
        self,
        params_dict: dict[str, nn.Parameter],
        coverage: dict[str, set[tuple[int, str]]],
        fused_coverage: dict[str, set[str]],
    ) -> None:
        par = self.parallel
        start = par.ep_rank * (int(self.config.num_experts) // par.ep_size)
        end = start + int(self.config.num_experts) // par.ep_size
        for name in params_dict:
            if name.endswith(".experts.w13_weight"):
                prefix = name.removesuffix(".w13_weight")
                if "w13" in fused_coverage.get(prefix, set()):
                    continue
                expected = {
                    (eid, proj)
                    for eid in range(start, end)
                    for proj in ("gate_proj", "up_proj")
                }
            elif name.endswith(".experts.w2_weight"):
                prefix = name.removesuffix(".w2_weight")
                if "w2" in fused_coverage.get(prefix, set()):
                    continue
                expected = {(eid, "down_proj") for eid in range(start, end)}
            else:
                continue
            missing = sorted(expected - coverage.get(prefix, set()))
            if missing:
                preview = ", ".join(f"expert {eid} {proj}" for eid, proj in missing[:8])
                suffix = " ..." if len(missing) > 8 else ""
                raise RuntimeError(
                    f"Qwen3MoeForCausalLM: incomplete {name}: "
                    f"missing {len(missing)} source tensors: {preview}{suffix}"
                )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        stacked_attn = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        stacked_dense_mlp = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        stacked_shared = [
            ("shared_expert_gate_up", "shared_expert.gate_proj", 0),
            ("shared_expert_gate_up", "shared_expert.up_proj", 1),
        ]
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        expert_coverage: dict[str, set[tuple[int, str]]] = {}
        fused_coverage: dict[str, set[str]] = {}

        for name, w in weights:
            layer_idx = _parse_layer_idx(name)
            if layer_idx is not None:
                if not (self.model.start_layer <= layer_idx < self.model.end_layer):
                    continue
                name = name.replace(
                    f"model.layers.{layer_idx}.",
                    f"model.layers.{layer_idx - self.model.start_layer}.",
                    1,
                )
            if "e_score_correction_bias" in name:
                continue
            if self._load_fused_expert_weight(name, w, loaded, fused_coverage):
                continue
            if ".mlp.experts." in name and "shared_expert" not in name:
                if self._load_expert_weight(name, w, loaded, expert_coverage):
                    continue
            if "shared_expert.down_proj" in name:
                mapped = name.replace("shared_expert.down_proj", "shared_expert_down")
                if mapped in params:
                    get_weight_loader(params[mapped])(params[mapped], w)
                    loaded.add(mapped)
                    continue
            hit = False
            for mapping in (stacked_shared, stacked_attn):
                for param_name, weight_name, shard_id in mapping:
                    if weight_name not in name:
                        continue
                    our = name.replace(weight_name, param_name)
                    if our in params:
                        params[our].weight_loader(params[our], w, shard_id)
                        loaded.add(our)
                    hit = True
                    break
                if hit:
                    break
            if hit:
                continue
            if ".mlp.experts." not in name and ".mlp.gate." not in name:
                for param_name, weight_name, shard_id in stacked_dense_mlp:
                    if weight_name not in name:
                        continue
                    our = name.replace(weight_name, param_name)
                    if our in params:
                        params[our].weight_loader(params[our], w, shard_id)
                        loaded.add(our)
                    hit = True
                    break
                if hit:
                    continue
            if name in params:
                get_weight_loader(params[name])(params[name], w)
                loaded.add(name)

        unloaded = set(params) - loaded
        if "lm_head.weight" in unloaded and "model.embed_tokens.weight" in loaded:
            embed = params["model.embed_tokens.weight"].data
            params["lm_head.weight"].data.copy_(embed[1:])
            loaded.add("lm_head.weight")
            unloaded.discard("lm_head.weight")
        self._assert_expert_coverage(params, expert_coverage, fused_coverage)
        if unloaded:
            print(f"[WARN] {len(unloaded)} params NOT loaded: {sorted(unloaded)[:10]}...")
