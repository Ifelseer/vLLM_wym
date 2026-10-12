"""QuantMethod 注册表：method 名 → 实现模块（懒加载）。

``awq`` / ``gptq`` 共用 ``int4.py``（同为 INT4 weight-only，仅权重布局与
scale 语义不同），由模块内按 ``quant.method`` 二次分派。
"""

from __future__ import annotations

import importlib

from ..config import QuantConfig

# quant method → 实现模块名
_MODULE_MAP = {
    "fp8": "fp8",
    "int8": "int8",
    "awq": "int4",
    "gptq": "int4",
}


def make_method(
    quant: QuantConfig,
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
    """按 quant.method 懒加载对应模块并构造 QuantMethod。"""
    module_name = _MODULE_MAP.get(quant.method)
    if module_name is None:
        raise ValueError(f"unknown quant method {quant.method!r}")
    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except ModuleNotFoundError as exc:
        raise NotImplementedError(
            f"quant method {quant.method!r} is not implemented yet "
            f"(missing src/quant/methods/{module_name}.py)"
        ) from exc
    return module.create(
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
