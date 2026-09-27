from __future__ import annotations

import torch
from torch import nn
from transformers.models.qwen3 import Qwen3Config

from ..layers import (
    Attention,
    MergedColumnParallelLinear,
    ParallelLMHead,
    QKVParallelLinear,
    RMSNorm,
    RotaryEmbedding,
    RowParallelLinear,
    SiluAndMul,
    VocabParallelEmbedding,
)
from ..layers.attention import AttnInputs
from ..layers.linear import get_weight_loader
from ...distributed import (
    IntermediateTensors,
    PPMissingLayer,
    get_pp_group,
    get_pp_indices,
)


def _parse_layer_idx(name: str) -> int | None:
    """
        从 HF 权重名解析全局层号（"model.layers.15..." → 15）
        非层权重返回 None
    """
    prefix = "model.layers."
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    idx = rest.split(".", 1)[0]
    try:
        return int(idx)
    except ValueError:
        return None


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

        # 将 gate_proj 和 up_proj 融合为单一的 MergedColumnParallelLinear 层。
        # 在 TP 下沿输出维度进行切分，减少 kernel launch 开销。
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size], bias=False
        )

        # down_proj 采用行并行。
        # 由于 gate_up_proj 的输出已经是按 TP 切分的，因此 down_proj 的输入也是分片状态，
        # 需设置 input_is_parallel=True 避免底层触发不必要的 All-Gather 操作。
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False, input_is_parallel=True
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x = self.gate_up_proj(x)
        return self.down_proj(self.act_fn(x))


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float,
        max_position: int = 32768,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        layer_id: int = 0,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim or hidden_size // num_heads
        self.use_qk_norm = use_qk_norm

        # 融合 QKV 投影层。在 TP 下，该层自动按注意力头进行切分并暴露出当前 rank 实际分配到的本地头数。
        self.qkv_proj = QKVParallelLinear(
            hidden_size, self.head_dim, num_heads, num_kv_heads, bias=qkv_bias
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # 输出投影层，采用行并行。
        # 输入维度应为全局 Q 的总维度 (num_heads_total * head_dim)，
        # 且 Attention 的输出已经按 TP 切分，需设置 input_is_parallel=True。
        self.o_proj = RowParallelLinear(
            self.qkv_proj.num_heads_total * self.head_dim,
            hidden_size,
            input_is_parallel=True,
        )
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, self.head_dim, max_position, rope_theta
        )

        # 核心 Attention 计算模块。
        # 传入的 layer_id 是当前 PP 内的本地层索引，用于在分布式 KV Cache 中正确索引对应的缓存块。
        self.attn = Attention(
            self.num_heads, self.head_dim, self.num_kv_heads, layer_id=layer_id
        )
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, attn: AttnInputs):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = self._split_norm_rope(positions, qkv)
        return self.o_proj(self.attn(q, k, v, attn))

    def _split_norm_rope(self, positions, qkv):
        """
            处理 QKV 投影后的张量：拆分 -> QK 归一化 -> RoPE 
        """
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if self.q_norm is not None:
            q, _ = self.q_norm(q)
            k, _ = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_id: int = 0):
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rope_theta=float(config.rope_parameters["rope_theta"]),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            layer_id=layer_id,
        )
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual: torch.Tensor | None, attn: AttnInputs):
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, attn)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        pp = get_pp_group()
        # 获取当前 PP 负责处理的 Transformer 层区间 [start_layer, end_layer)
        self.start_layer, self.end_layer = get_pp_indices(
            config.num_hidden_layers, pp.group_rank, pp.size
        )
        
        # 词嵌入层：仅放置在 PP 的第一个 stage (is_first_rank)。
        # 非首个 stage 使用 PPMissingLayer 占位，以维持模型结构和参数命名的一致性。
        self.embed_tokens = (
            VocabParallelEmbedding(config.vocab_size, config.hidden_size)
            if pp.is_first_rank
            else PPMissingLayer()
        )
        # 实例化当前 stage 负责的 Transformer 层。
        # local_i 为当前 stage 内的相对层索引。
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_id=local_i)
                for local_i in range(self.end_layer - self.start_layer)
            ]
        )
        # 最终归一化层：仅放置在 PP 的最后一个 stage (is_last_rank)
        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if pp.is_last_rank
            else PPMissingLayer()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn: AttnInputs,
        intermediate_tensors: IntermediateTensors | None = None,
    ):
        if intermediate_tensors is None:
            # 首个 PP stage 从 input_ids 计算 Embedding
            h, residual = self.embed_tokens(input_ids), None
        else:
            # 中间 PP stage 接收上一个 stage 的 hidden_states 和 residual
            h = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        # 处理当前 PP 负责的 Transformer 层
        for layer in self.layers:
            h, residual = layer(positions, h, residual, attn)
        # 最后一个 PP stage 进行最终归一化
        if get_pp_group().is_last_rank:
            return self.norm(h, residual)[0]
        # 中间 PP stage 返回 IntermediateTensors 供后续 stage 使用
        return IntermediateTensors(hidden_states=h, residual=residual)


class Qwen3ForCausalLM(nn.Module):

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)
        # LM Head 放置在 PP 的最后一个 stage
        self.lm_head = (
            ParallelLMHead(config.hidden_size, config.vocab_size, bias=False)
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )

    def forward(self, 
        input_ids: torch.Tensor, 
        positions: torch.Tensor, 
        attn: AttnInputs, 
        intermediate_tensors: IntermediateTensors | None = None
    ) -> torch.Tensor | IntermediateTensors:

        return self.model(input_ids, positions, attn, intermediate_tensors)

    def compute_logits(self, hidden_states):
        """ 计算全量的 logits """
        return self.lm_head(hidden_states)

    def argmax(self, hidden_states):
        """贪心路径，直接计算全局 argmax token id 避免对全量 logits 进行 all-gather"""
        return self.lm_head.argmax(hidden_states)

    def load_weights(self, weights):
        """
            从 HuggingFace 格式加载权重，并自动处理 TP 切分与 PP 映射。
            
            主要处理逻辑:
            1. 过滤并映射层索引：根据当前 PP stage 重命名 layer index。
            2. 处理融合权重：将 HF 的独立 Q/K/V 和 Gate/Up 权重切分加载到融合矩阵中。
            3. 处理普通权重：直接加载未融合的权重。
            4. 权重绑定：处理 lm_head 和 embed_tokens 的权重共享逻辑。
        """


        # 定义需要特殊处理的融合层映射关系
        stacked = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params = dict(self.named_parameters())
        loaded = set()
        skipped = []

        for name, w in weights:
            # 1. PP 层索引过滤和重映射
            layer_idx = _parse_layer_idx(name)
            if layer_idx is not None:
                # 如果该层并不属于当前 PP stage，则跳过
                if not (self.model.start_layer <= layer_idx < self.model.end_layer):
                    continue
                # 如果该层属于当前 PP stage，则重命名层索引， 从0开始
                name = name.replace(
                    f"model.layers.{layer_idx}.",
                    f"model.layers.{layer_idx - self.model.start_layer}.",
                    1,
                )

            # 2. 处理融合权重
            for param_name, weight_name, shard_id in stacked:
                if weight_name not in name:
                    continue
                # 替换权重名，得到我们需要的权重名
                our = name.replace(weight_name, param_name)
                if our in params:
                    # 调用权重加载器，将权重加载到模型中
                    params[our].weight_loader(params[our], w, shard_id)
                    loaded.add(our)
                else:
                    skipped.append(name)
                break
            # 3. 处理普通权重
            else:
                if name in params:
                    get_weight_loader(params[name])(params[name], w)
                    loaded.add(name)
                else:
                    skipped.append(name)


        # 4. 权重绑定
        # 如果 lm_head 未被显式加载，则复用 embed_tokens 的权重。
        # 注意：VocabParallelEmbedding 内部会多分配一行用于越界占位 (padding)，
        # 因此在复制给 lm_head 时需要切片跳过第一行 (embed[1:]) 以保证 shape 匹配。
        unloaded = set(params) - loaded
        if "lm_head.weight" in unloaded and "model.embed_tokens.weight" in loaded:
            embed = params["model.embed_tokens.weight"].data
            params["lm_head.weight"].data.copy_(embed[1:])
            loaded.add("lm_head.weight")
            unloaded.discard("lm_head.weight")

        if unloaded:
            print(f"[WARN] {len(unloaded)} params NOT loaded: {sorted(unloaded)[:10]}...")
        if skipped:
            print(f"[INFO] {len(skipped)} HF weights skipped (not in model): {skipped[:5]}...")
