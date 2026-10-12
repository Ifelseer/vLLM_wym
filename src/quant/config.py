"""QuantConfig：量化配置与 HF quantization_config 解析。

支持四类量化方法（method）::

    fp8   W8A8（权重 + 激活），e4m3，scale 模式 per_tensor / per_block_128x128
    int8  W8A8，per-token 动态激活 × per-channel 静态权重
    awq   INT4 weight-only（group 量化，zero_point，version=gemm）
    gptq  INT4 weight-only（group 量化，sym，支持 desc_act act-order）

构造来源：
1. HF config.json 的 ``quantization_config``（社区预量化 checkpoint）——
   :meth:`QuantConfig.from_hf_config`。
2. 命令行 / config.yaml 显式指定（如 ``quantization: fp8``，配合
   ``quantize_at_load`` 把 BF16 checkpoint 加载时在线量化）——
   :meth:`QuantConfig.from_name`。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# 命令行 / config.yaml 可填的取值
VALID_METHODS = ("none", "auto", "fp8", "int8", "awq", "gptq")

# FP8 e4m3 动态范围上界（448），在线量化用
FP8_E4M3_MAX = 448.0


@dataclass(frozen=True)
class QuantConfig:
    """量化配置（不可变；构造后按 hash 缓存，供层工厂复用）。"""

    method: str
    bits: int = 8
    fmt: str = "e4m3"
    # fp8: "per_tensor" | "per_block_128x128"
    # int8: "per_channel"
    # awq/gptq: "group"
    weight_scale: str = "per_tensor"
    # fp8/int8: "dynamic" | "static"；awq/gptq（weight-only）: None
    act_scale: str | None = "dynamic"
    group_size: int | None = None
    desc_act: bool = False
    sym: bool = False
    zero_point: bool = True
    version: str = "gemm"
    # FP8 KV cache 的 dtype（None = KV 不量化）；仅 fp8 方法允许设置
    kv_cache_dtype: torch.dtype | None = None
    # HF quantization_config.modules_to_not_convert（如 lm_head / router gate）
    modules_to_not_convert: tuple[str, ...] = ()
    # BF16/FP16 checkpoint 加载时在线量化（仅 fp8 / int8 允许；int4 需校准）
    quantize_at_load: bool = False

    # ------------------------------------------------------------------ #
    # 派生属性
    # ------------------------------------------------------------------ #

    @property
    def is_weight_only(self) -> bool:
        """awq/gptq：只量化权重，激活保持高精度。"""
        return self.method in ("awq", "gptq")

    @property
    def is_w8a8(self) -> bool:
        """fp8/int8：权重与激活都量化。"""
        return self.method in ("fp8", "int8")

    @property
    def block_size(self) -> tuple[int, int] | None:
        """per-block scale 的 [block_out, block_in]；非 block 模式返回 None。"""
        if self.weight_scale == "per_block_128x128":
            return (128, 128)
        return None

    def skip_module(self, prefix: str) -> bool:
        """prefix 是否命中 modules_to_not_convert。

        点分段感知匹配：skip 项的分段序列须为 prefix 分段序列的连续子序列。
        这样 ``mlp.gate``（router）不会误伤 ``mlp.gate_up_proj``（dense MLP），
        而 ``lm_head`` 仍可命中 ``lm_head.weight``。
        """
        if not self.modules_to_not_convert:
            return False
        parts = prefix.split(".")
        for m in self.modules_to_not_convert:
            if not m:
                continue
            mp = m.split(".")
            n = len(mp)
            if n > len(parts):
                continue
            if any(parts[i : i + n] == mp for i in range(len(parts) - n + 1)):
                return True
        return False

    def supports_device(self, capability: tuple[int, int]) -> bool:
        return supports_device(self.method, capability)

    # ------------------------------------------------------------------ #
    # 构造：HF quantization_config
    # ------------------------------------------------------------------ #

    @classmethod
    def from_hf_config(
        cls,
        hf_cfg,
        *,
        device_capability: tuple[int, int],
        kv_cache_dtype: torch.dtype | None = None,
    ) -> QuantConfig | None:
        """从 HF PretrainedConfig 解析 quantization_config。

        Returns:
            QuantConfig；模型无 quantization_config 时返回 None。
        Raises:
            ValueError: 量化格式不受支持，或硬件不兼容。
        """
        qc = getattr(hf_cfg, "quantization_config", None)
        if qc is None:
            return None
        # transformers 可能给出 dict 或 QuantizationConfigMixin 对象
        d = qc.to_dict() if hasattr(qc, "to_dict") else dict(qc)
        method = d.get("quant_method")
        if method is None:
            raise ValueError(f"quantization_config missing quant_method: {d}")

        common = dict(
            modules_to_not_convert=tuple(d.get("modules_to_not_convert") or ()),
        )

        if method == "fp8":
            cfg = cls(
                method="fp8",
                bits=8,
                fmt=d.get("fmt", "e4m3"),
                weight_scale=cls._fp8_scale_mode(d),
                act_scale=d.get("activation_scheme", "dynamic"),
                kv_cache_dtype=kv_cache_dtype,
                **common,
            )
        elif method == "int8":
            cfg = cls(
                method="int8",
                bits=8,
                weight_scale="per_channel",
                act_scale=d.get("activation_scheme", "dynamic"),
                **common,
            )
        elif method == "awq":
            cfg = cls(
                method="awq",
                bits=int(d.get("bits", 4)),
                weight_scale="group",
                act_scale=None,
                group_size=int(d.get("group_size", 128)),
                zero_point=bool(d.get("zero_point", True)),
                version=str(d.get("version", "gemm")),
                **common,
            )
        elif method == "gptq":
            cfg = cls(
                method="gptq",
                bits=int(d.get("bits", 4)),
                weight_scale="group",
                act_scale=None,
                group_size=int(d.get("group_size", 128)),
                desc_act=bool(d.get("desc_act", False)),
                sym=bool(d.get("sym", False)),
                **common,
            )
        else:
            raise ValueError(
                f"unsupported quant_method {method!r} "
                "(supported: fp8 / int8 / awq / gptq)"
            )

        if not cfg.supports_device(device_capability):
            raise ValueError(
                f"quant method {cfg.method!r} not supported on SM "
                f"{device_capability[0]}{device_capability[1]}; "
                "use a different method or device"
            )
        return cfg

    @staticmethod
    def _fp8_scale_mode(d: dict) -> str:
        """weight_block_size=[128,128] → per_block_128x128；缺省 → per_tensor。"""
        block = d.get("weight_block_size")
        if block is None:
            return "per_tensor"
        block = tuple(int(x) for x in block)
        if block == (128, 128):
            return "per_block_128x128"
        raise ValueError(
            f"unsupported fp8 weight_block_size {block}; "
            "only per-tensor and 128x128 block are supported"
        )

    # ------------------------------------------------------------------ #
    # 构造：显式指定（CLI / config.yaml）
    # ------------------------------------------------------------------ #

    @classmethod
    def from_name(
        cls,
        name: str,
        *,
        hf_cfg=None,
        device_capability: tuple[int, int],
        kv_cache_dtype: torch.dtype | None = None,
        quantize_at_load: bool = False,
    ) -> QuantConfig | None:
        """按显式名称构造。

        ``none``  → None（不量化）。
        ``auto``  → 模型自带 quantization_config 就按它解析，否则 None。
        ``fp8`` 等 → 强制该方法（BF16 checkpoint 时 ``quantize_at_load``
                     在加载时在线量化；仅 fp8/int8 允许）。
        """
        if name not in VALID_METHODS:
            raise ValueError(
                f"invalid quantization {name!r}; expected one of {VALID_METHODS}"
            )
        if name == "none":
            return None
        if name == "auto":
            return cls.from_hf_config(
                hf_cfg,
                device_capability=device_capability,
                kv_cache_dtype=kv_cache_dtype,
            )
        if quantize_at_load and name in ("awq", "gptq"):
            raise ValueError(
                f"{name} cannot be produced by load-time quantization "
                "(needs calibrated checkpoints); use a pre-quantized model"
            )
        if name in ("awq", "gptq"):
            # INT4 无法从 BF16 在线量化：必须提供带 quantization_config 的
            # hf_cfg，且 checkpoint 的 quant_method 与请求一致。
            qc = getattr(hf_cfg, "quantization_config", None) if hf_cfg else None
            if qc is None:
                raise ValueError(
                    f"{name} requires a pre-quantized checkpoint "
                    "(HF quantization_config); none provided"
                )
            d = qc.to_dict() if hasattr(qc, "to_dict") else dict(qc)
            if d.get("quant_method") != name:
                raise ValueError(
                    f"requested {name} but checkpoint is "
                    f"{d.get('quant_method')!r}; pass the matching method or 'auto'"
                )
            return cls.from_hf_config(
                hf_cfg, device_capability=device_capability
            )
        if name == "fp8":
            cfg = cls(
                method="fp8",
                bits=8,
                fmt="e4m3",
                # 在线量化默认 per-block（社区 FP8 checkpoint 主流格式）；
                # 预量化路径由 from_hf_config 按 checkpoint 真实格式覆盖。
                weight_scale="per_block_128x128",
                act_scale="dynamic",
                kv_cache_dtype=kv_cache_dtype,
                quantize_at_load=quantize_at_load,
                # 在线量化默认跳过：lm_head（logits 精度）/ embedding（无收益）/
                # MoE router（精度敏感）
                modules_to_not_convert=("lm_head", "model.embed_tokens", "mlp.gate"),
            )
        elif name == "int8":
            cfg = cls(
                method="int8",
                bits=8,
                weight_scale="per_channel",
                act_scale="dynamic",
                quantize_at_load=quantize_at_load,
                modules_to_not_convert=("lm_head", "model.embed_tokens", "mlp.gate"),
            )
        elif name == "awq":
            cfg = cls(
                method="awq",
                bits=4,
                weight_scale="group",
                act_scale=None,
                group_size=128,
                zero_point=True,
                version="gemm",
            )
        else:  # gptq
            cfg = cls(
                method="gptq",
                bits=4,
                weight_scale="group",
                act_scale=None,
                group_size=128,
                sym=True,
            )
        if not cfg.supports_device(device_capability):
            raise ValueError(
                f"quant method {cfg.method!r} not supported on SM "
                f"{device_capability[0]}{device_capability[1]}; "
                "use a different method or device"
            )
        return cfg


def supports_device(method: str, capability: tuple[int, int]) -> bool:
    """量化方法 × GPU 架构支持矩阵。

    - fp8：SM89+（Ada / Hopper / Blackwell 原生 e4m3 tensor core）。
      SM86（3060）等无原生 FP8，显式拒绝而非静默降级。
    - int8：SM61+（Pascal int8 DP4A；Volta+ cuBLASLt int8 GEMM）。
    - awq/gptq：任意架构（Triton fused dequant-GEMM）。
    - capability == (0, 0)（无 CUDA / 未知，如 CPU 环境）→ 放行，
      真实硬件门控推迟到 Worker 初始化（那里能拿到确切 device capability）。
    """
    if capability == (0, 0):
        return True
    major, minor = capability
    sm = major * 10 + minor
    if method == "fp8":
        return sm >= 89
    if method in ("int8", "awq", "gptq"):
        return sm >= 61
    raise ValueError(f"unknown quant method {method!r}")
