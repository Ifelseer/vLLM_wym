"""src/quant：量化框架（配置识别 → 方法注册表分派 → kernel 执行）。

三层结构::

    quant/config.py     QuantConfig：量化配置（由 EngineConfig.quantization
                        + HF quantization_config 共同构造），格式解析与硬件校验
    quant/methods/      QuantMethod 实现（fp8 / int4 / int8），按注册表分派
    quant/kernels/      底层 kernel（Triton / torch 原语封装）

设计借鉴 vLLM 的 ``QuantizationConfig.from_config`` + ``get_quant_method``
方法分派：线性层只做薄委托，量化逻辑全部下沉本模块；control/data 两面零改动。

依赖方向：``compute/*`` 与 ``moe/*`` 可依赖本模块；本模块只依赖 ``torch``
与 ``src/config.py`` 的类型，不反向依赖 compute 具体实现。
"""

from __future__ import annotations

from .config import QuantConfig, supports_device

__all__ = [
    "QuantConfig",
    "supports_device",
    "get_quant_method",
    "quant_for",
    "finalize_model",
]


def quant_for(quant: QuantConfig | None, prefix: str) -> QuantConfig | None:
    """按参数名前缀过滤：命中 modules_to_not_convert 时返回 None（不量化）。

    模型构造时对每个子模块调用，例如::

        q = quant_for(quant, "self_attn.qkv_proj")
        self.qkv_proj = QKVParallelLinear(..., quant=q)
    """
    if quant is None or quant.skip_module(prefix):
        return None
    return quant


def finalize_model(model) -> None:
    """加载完成后收尾：对所有带 quant_method 的模块调用 finalize。

    在线量化（quantize_at_load）路径下，各层的 BF16 staging buffer 在此统一
    量化为 fp8 权重 + scale；预量化 checkpoint 路径为 no-op。
    在 ``utils.model_loader.load_model`` 的 ``load_weights`` 之后调用一次。
    """
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if method is not None and hasattr(method, "finalize"):
            method.finalize(module)


def get_quant_method(
    quant: QuantConfig | None,
    *,
    kind: str,
    out_size: int,
    in_size: int,
    tp_size: int,
    tp_rank: int,
    prefix: str = "",
    merged_sizes: list[int] | None = None,
    qkv_meta: tuple[int, int, int, int] | None = None,
):
    """按 QuantConfig 构造某一层的 QuantMethod。

    Args:
        quant: 量化配置；None 或 method == "none" 时返回 None（调用方走原路径）。
        kind: 层类型——``"replicated"`` / ``"column"`` / ``"row"`` /
            ``"merged_column"`` / ``"qkv"``。
        out_size / in_size: 本 TP rank 负责的输出 / 输入维度（切分后）。
        tp_size / tp_rank: 张量并行信息（scale 切分需要）。
        prefix: 层在模型中的参数名前缀（用于 ``modules_to_not_convert`` 过滤）。
        merged_sizes: merged_column 的各段输出尺寸（gate/up 等）。
        qkv_meta: qkv 层的 (heads_total, kv_total, head_dim, replicas)。

    Returns:
        QuantMethod 实例或 None。
    """
    if quant is None or quant.method == "none":
        return None
    if quant.skip_module(prefix):
        return None

    from .methods import make_method

    return make_method(
        quant,
        kind=kind,
        out_size=out_size,
        in_size=in_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        prefix=prefix,
        merged_sizes=merged_sizes,
        qkv_meta=qkv_meta,
    )
