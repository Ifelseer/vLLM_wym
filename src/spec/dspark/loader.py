"""加载官方 DSpark 草稿 checkpoint，并做依赖/规格检查。

用法（Phase 3）::

    draft = load_dspark_draft("~/huggingface/dspark_qwen3_8b_block7",
                              device="cuda:0", dtype=torch.bfloat16)

为什么不用 `AutoModel.from_pretrained`：官方 checkpoint 的 `architectures` 是
`Qwen3DSparkModel`，而该实现**不在 transformers 里**（config 里的
`transformers_version: 5.10.2` 只是保存时版本，不参与校验）。所以必须用本包
vendored 的类直接 `from_pretrained`。
"""

from __future__ import annotations

from pathlib import Path

import torch

from .modeling import Qwen3DSparkModel

#: vendored modeling.py 需要的 transformers 符号（缺一不可）
REQUIRED_TRANSFORMERS_SYMBOLS = (
    "Qwen3MLP",
    "Qwen3RMSNorm",
    "Qwen3RotaryEmbedding",
    "eager_attention_forward",
    "rotate_half",
    "ALL_ATTENTION_FUNCTIONS",
    "GradientCheckpointingLayer",
    "FlashAttentionKwargs",
)

#: 官方 checkpoint 必须带的 config 字段
REQUIRED_CONFIG_FIELDS = (
    "block_size",
    "target_layer_ids",
    "mask_token_id",
    "markov_rank",
    "vocab_size",
    "num_hidden_layers",
)


def check_transformers_symbols() -> list[str]:
    """返回缺失的符号列表（空 = 全部可用）。

    为什么单独做检查：不同 transformers 版本对 Qwen3 组件的导出位置有差异，
    提前报"缺哪个"比在 `import modeling` 时抛一句 AttributeError 好排查得多。
    """
    import transformers.models.qwen3.modeling_qwen3 as qwen3

    missing = []
    for name in REQUIRED_TRANSFORMERS_SYMBOLS:
        if hasattr(qwen3, name):
            continue
        # GradientCheckpointingLayer / FlashAttentionKwargs 在部分版本里从
        # modeling_layers / modeling_utils 导出，未 re-export 到 modeling_qwen3
        fallback = {
            "GradientCheckpointingLayer": "transformers.modeling_layers",
            "FlashAttentionKwargs": "transformers.modeling_utils",
        }.get(name)
        found = False
        if fallback is not None:
            try:
                module = __import__(fallback, fromlist=[name])
                found = hasattr(module, name)
            except Exception:  # pragma: no cover - 版本差异
                found = False
        if not found:
            missing.append(name)
    return missing


def warn_if_degraded_transformers() -> str | None:
    """transformers < 5 时草稿的数值与上游不一致，acceptance 会明显偏低。

    实测（2026-10-25）：同一份 vendored 代码、同一份 ctx/anchor，

    - transformers 5.10：hidden 与 tokens 与上游**完全相等**；
    - transformers 4.57：block_hidden 相对误差 33%，从第 2 个草稿位置起分叉。

    原因是草稿模型仍借用 transformers 的 Qwen3 组件（RotaryEmbedding /
    sdpa_attention_forward），这两个在 4.x→5.x 之间行为变了。**不影响正确性**
    （草稿永远被目标模型验证），只影响接受率（实测 accept_len 5.53 vs 6.26）。

    返回告警文本（未触发时为 None），由调用方打印。
    """
    import transformers

    major = int(transformers.__version__.split(".")[0])
    if major >= 5:
        return None
    return (
        f"[WARN] transformers {transformers.__version__} < 5：DSpark 草稿的数值"
        "与上游实现不一致（实测 block_hidden rel 33%），accept_len 会明显偏低"
        "（约 5.5 vs 6.26）。草稿仍被验证，输出正确性不受影响；要拿到论文量级的"
        "接受率需 transformers>=5，或改用自带的 RoPE/attention 实现"
        "（见 docs/dspark-spec.md §7）。"
    )


def assert_transformers_ready() -> None:
    import transformers

    missing = check_transformers_symbols()
    if missing:
        raise ImportError(
            f"transformers {transformers.__version__} 缺少 DSpark 草稿所需的符号: "
            f"{missing}。升级 transformers（Qwen3 组件自 4.51 起提供）或把缺失的"
            "符号加入 src/spec/dspark/modeling.py 的 import 兼容层。"
        )


def load_draft_config(path: str | Path, **kwargs):
    """只读 config（不加载权重），用于打印/校验规格。"""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(str(path), trust_remote_code=False, **kwargs)


def check_config_leakage(config) -> list[str]:
    """返回**缺失**的必需字段。"""
    return [f for f in REQUIRED_CONFIG_FIELDS if not hasattr(config, f)]


def load_dspark_draft(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
) -> Qwen3DSparkModel:
    """加载草稿模型（eval + 关闭梯度）。

    ``attn_implementation`` 默认 ``sdpa``：上游 eval 也是 sdpa。`flex_attention`
    只用于训练（需要 block mask），推理不要开。
    """
    assert_transformers_ready()
    warning = warn_if_degraded_transformers()
    if warning is not None:
        print(warning, flush=True)
    kwargs = {
        "attn_implementation": attn_implementation,
        "low_cpu_mem_usage": True,
    }
    try:
        model = Qwen3DSparkModel.from_pretrained(str(path), dtype=dtype, **kwargs)
    except TypeError:
        # transformers < 4.56 用 torch_dtype
        model = Qwen3DSparkModel.from_pretrained(
            str(path), torch_dtype=dtype, **kwargs
        )
    model = model.to(device=device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


__all__ = [
    "REQUIRED_CONFIG_FIELDS",
    "REQUIRED_TRANSFORMERS_SYMBOLS",
    "assert_transformers_ready",
    "check_config_leakage",
    "check_transformers_symbols",
    "load_draft_config",
    "load_dspark_draft",
]
