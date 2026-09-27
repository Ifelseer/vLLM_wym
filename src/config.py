from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from transformers import AutoConfig, PretrainedConfig


def _is_moe_config(cfg: PretrainedConfig) -> bool:
    arch = getattr(cfg, "architectures", None) or []
    name = type(cfg).__name__
    if any("Moe" in a or "MoE" in a for a in arch) or "Moe" in name or "MoE" in name:
        return True
    return int(getattr(cfg, "num_experts", 0) or 0) > 0


@dataclass
class EngineConfig:
    """引擎配置：调度 / 内存 / 算子 / 并行。

    并行语义（三者正交，分叉点不同）::

        dp_size  → 独立 Engine 副本数（``MPClient`` / ZMQ）；不进入 NCCL world
        tp_size  → 同 Engine 内张量并行度（NCCL all-reduce / gather）
        pp_size  → 同 Engine 内流水线 stage 数（NCCL P2P hidden+residual）

    ``enable_ep`` 不增加 world：Attention 仍走 TP，MoE 在同一批 TP rank 上按 expert 切分。
    单 Engine 的 NCCL ``world_size = tp_size * pp_size``。
    """

    model: str

    context_len: int = 4096
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    mix_prefill_decode: bool = True

    kvcache_block_size: int = 256
    gpu_memory_utilization: float = 0.9
    num_kvcache_blocks: int = 0
    prefix_backend: str = "hash"

    enforce_eager: bool = True
    torch_compile: bool = False
    compile_mode: str = "reduce-overhead"
    compile_dynamic: bool = True
    dtype: str = "auto"

    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    nccl_port: int = 29500
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    device_ids: list[int] | None = None
    nccl_ifname: str = ""
    pp_layer_counts: list[int] | None = None

    enable_ep: bool = False
    moe_backend: Literal["auto", "cutlass", "triton"] = "auto"

    hf_config: PretrainedConfig = field(init=False)
    is_moe: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.model = os.path.expanduser(self.model)
        if not os.path.isdir(self.model):
            raise FileNotFoundError(
                f"Model path not found: {self.model}. "
                "Use a local directory (e.g. ~/huggingface/Qwen3-0.6B)."
            )
        if self.kvcache_block_size % 256 != 0:
            raise ValueError("kvcache_block_size must be a multiple of 256 (FA2 paged)")
        if self.prefix_backend not in ("none", "hash", "radix"):
            raise ValueError(
                f"prefix_backend must be none|hash|radix, got {self.prefix_backend!r}"
            )
        if self.tp_size < 1 or self.pp_size < 1 or self.dp_size < 1:
            raise ValueError("tp_size / pp_size / dp_size must be >= 1")
        if self.moe_backend not in ("auto", "cutlass", "triton"):
            raise ValueError(
                f"moe_backend must be auto|cutlass|triton, got {self.moe_backend!r}"
            )
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.is_moe = _is_moe_config(self.hf_config)
        text = getattr(self.hf_config, "text_config", self.hf_config)
        self.context_len = min(
            self.context_len,
            getattr(text, "max_position_embeddings", self.context_len),
        )
        self._validate_parallel(text)
        # MoE 可以进 Decode CUDA Graph：token 数按 captured BS 垫齐后，
        # align / silu / sum / GEMM workspace 都按容量复用，不再逐步 cudaMalloc。
        # 若环境仍要强制 eager：VLLM_MOE_FORCE_EAGER=1。
        if self.is_moe and os.environ.get("VLLM_MOE_FORCE_EAGER", "0") in (
            "1",
            "true",
            "True",
        ):
            self.enforce_eager = True

    def _validate_parallel(self, text: PretrainedConfig) -> None:
        n_heads = int(text.num_attention_heads)
        n_kv = int(text.num_key_value_heads)
        inter = int(text.intermediate_size)
        if n_heads % self.tp_size != 0:
            raise ValueError(
                f"num_attention_heads ({n_heads}) must be divisible by tp_size ({self.tp_size})"
            )
        if n_kv >= self.tp_size:
            if n_kv % self.tp_size != 0:
                raise ValueError(
                    f"num_key_value_heads ({n_kv}) must be divisible by tp_size ({self.tp_size})"
                )
        elif self.tp_size % n_kv != 0:
            raise ValueError(
                f"tp_size ({self.tp_size}) must be divisible by num_key_value_heads ({n_kv})"
            )
        if inter % self.tp_size != 0:
            raise ValueError(
                f"intermediate_size ({inter}) must be divisible by tp_size ({self.tp_size})"
            )
        if self.pp_size > int(text.num_hidden_layers):
            raise ValueError(
                f"pp_size ({self.pp_size}) > num_hidden_layers ({text.num_hidden_layers})"
            )
        if not self.is_moe:
            if self.enable_ep:
                raise ValueError("enable_ep requires a MoE model")
            return
        moe_inter = getattr(text, "moe_intermediate_size", None)
        if moe_inter is None:
            raise ValueError("MoE config missing moe_intermediate_size")
        moe_inter = int(moe_inter)
        n_experts = int(getattr(text, "num_experts", 0) or 0)
        if n_experts <= 0:
            raise ValueError("MoE config has num_experts <= 0")
        shared_inter = int(getattr(text, "shared_expert_intermediate_size", 0) or 0)
        if self.enable_ep:
            if n_experts % self.tp_size != 0:
                raise ValueError(
                    f"enable_ep requires num_experts ({n_experts}) % tp_size "
                    f"({self.tp_size}) == 0"
                )
        else:
            if moe_inter % self.tp_size != 0:
                raise ValueError(
                    f"moe_intermediate_size ({moe_inter}) must be divisible by "
                    f"tp_size ({self.tp_size})"
                )
            if shared_inter and shared_inter % self.tp_size != 0:
                raise ValueError(
                    f"shared_expert_intermediate_size ({shared_inter}) must be "
                    f"divisible by tp_size ({self.tp_size})"
                )

    @property
    def world_size(self) -> int:
        """单 Engine 的 NCCL world（不含 DP 副本）。"""
        return self.tp_size * self.pp_size

    @property
    def num_gpus(self) -> int:
        """门面视角下的总 GPU 数：``dp_size * tp_size * pp_size``。"""
        return self.dp_size * self.tp_size * self.pp_size

    @property
    def block_size(self) -> int:
        return self.kvcache_block_size
