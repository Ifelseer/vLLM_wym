from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Literal

import torch
from transformers import AutoConfig, PretrainedConfig

from .quant import QuantConfig
from .quant.config import VALID_METHODS as _QUANT_VALID_METHODS


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

    # --- 量化 ---
    # none / auto / fp8 / int8 / awq / gptq：
    #   auto = 模型自带 quantization_config 就按它解析，否则不量化；
    #   fp8/int8 = 强制该方法（BF16 checkpoint 时加载时在线量化）；
    #   awq/gptq = 强制 INT4 weight-only（必须搭配预量化 checkpoint）。
    quantization: str = "none"
    # FP8 KV cache（仅 fp8 路径生效，SM89+；FA2 不支持，走 FlashInfer wrapper）
    quant_kv_cache: bool = False

    # --- 投机解码（speculative decoding）---
    # 0 = 关闭；k>0 = 每步草稿 k 个 token，一次 verify 写入 1+k 个位置。
    # v1 要求 max_num_seqs == 1（verify 按单序列实现，见 docs/dspark-plan.md）。
    spec_tokens: int = 0
    # 草稿来源：ngram（无模型，Phase 1 验证替身）/ dspark（Phase 3 接入）
    spec_draft: str = "ngram"
    # greedy tie 守卫（0 = 关闭）。verify 与 decode 用不同 attention 内核，
    # 目标分布出现精确并列（top1≈top2）时 argmax 可能因 ~1ulp 差异翻转，导致
    # 与 baseline 分叉。>0 时：若 emit 位置的 top1-top2 间隔小于该值，就用
    # **decode 内核**重算该位置（与 baseline 完全同路径），保证 greedy 逐位一致。
    # 代价：只在 tie 步多一次 1-token 前向（罕见）。
    spec_tie_eps: float = 0.0

    hf_config: PretrainedConfig = field(init=False)
    is_moe: bool = field(init=False, default=False)
    quant_config: QuantConfig | None = field(init=False, default=None)

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
        if self.quantization not in _QUANT_VALID_METHODS:
            raise ValueError(
                f"quantization must be none|auto|fp8|int8|awq|gptq, "
                f"got {self.quantization!r}"
            )
        if self.spec_tokens < 0:
            raise ValueError(f"spec_tokens must be >= 0, got {self.spec_tokens}")
        if self.spec_tokens > 0:
            # v1 限制：verify 是单序列路径（树形验证需要每序列一棵树 + 拼平
            # cu_seqlens），多序列留到 v2。显式报错好过静默忽略。
            if self.max_num_seqs != 1:
                raise ValueError(
                    "spec_tokens > 0 requires max_num_seqs == 1 in v1, "
                    f"got max_num_seqs={self.max_num_seqs}"
                )
            if self.spec_draft not in ("ngram", "dspark"):
                raise ValueError(
                    f"spec_draft must be ngram|dspark, got {self.spec_draft!r}"
                )
        if self.spec_tie_eps < 0:
            raise ValueError(f"spec_tie_eps must be >= 0, got {self.spec_tie_eps}")
        if self.spec_tie_eps > 0:
            # tie 守卫要在 verify 之后追加一次 decode 前向；PP 下各 stage 的
            # verify 分支不对称（非末段提前 send），会死锁。v1 只支持 pp=1。
            if self.pp_size > 1:
                raise ValueError("spec_tie_eps > 0 requires pp_size == 1 in v1")
            if self.spec_tokens <= 0:
                raise ValueError("spec_tie_eps > 0 requires spec_tokens > 0")
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.is_moe = _is_moe_config(self.hf_config)
        self.quant_config = self._resolve_quant()
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

    def _resolve_quant(self) -> QuantConfig | None:
        """解析量化配置（EngineConfig.quantization + HF quantization_config）。

        规则：
        - ``none``            → 不量化。
        - ``auto``            → 模型自带 quantization_config 就按它解析。
        - ``awq`` / ``gptq``  → 必须搭配同方法的预量化 checkpoint。
        - ``fp8`` / ``int8``  → checkpoint 已是同方法则按其格式；
                                否则对 BF16 checkpoint 在线量化（加载时）。
        """
        if self.quantization == "none":
            return None
        capability = self._device_capability()
        hf_qc = getattr(self.hf_config, "quantization_config", None)
        kv_dtype = (
            torch.float8_e4m3fn
            if self.quant_kv_cache and self.quantization in ("auto", "fp8")
            else None
        )
        if self.quantization == "auto":
            return QuantConfig.from_hf_config(
                self.hf_config,
                device_capability=capability,
                kv_cache_dtype=kv_dtype,
            )
        if self.quantization in ("awq", "gptq"):
            if hf_qc is None:
                raise ValueError(
                    f"quantization={self.quantization} requires a pre-quantized "
                    f"checkpoint (HF quantization_config); none found in {self.model}"
                )
            d = hf_qc.to_dict() if hasattr(hf_qc, "to_dict") else dict(hf_qc)
            if d.get("quant_method") != self.quantization:
                raise ValueError(
                    f"quantization={self.quantization} but checkpoint is "
                    f"{d.get('quant_method')!r}; use quantization=auto or match "
                    "the checkpoint"
                )
            return QuantConfig.from_hf_config(
                self.hf_config, device_capability=capability
            )
        # fp8 / int8：checkpoint 已是同方法 → 按 checkpoint 格式（尊重其 scale
        # 模式与 modules_to_not_convert）；否则加载时在线量化。
        if hf_qc is not None:
            d = hf_qc.to_dict() if hasattr(hf_qc, "to_dict") else dict(hf_qc)
            if d.get("quant_method") == self.quantization:
                cfg = QuantConfig.from_hf_config(
                    self.hf_config, device_capability=capability
                )
                if kv_dtype is not None and cfg.kv_cache_dtype is None:
                    cfg = replace(cfg, kv_cache_dtype=kv_dtype)
                return cfg
        return QuantConfig.from_name(
            self.quantization,
            device_capability=capability,
            kv_cache_dtype=kv_dtype,
            quantize_at_load=True,
        )

    @staticmethod
    def _device_capability() -> tuple[int, int]:
        """当前默认 GPU 的 (major, minor)；无 CUDA 时返回 (0, 0)（未知）。"""
        if torch.cuda.is_available():
            try:
                return torch.cuda.get_device_capability()
            except Exception:  # pragma: no cover - 驱动异常等
                pass
        return (0, 0)

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
