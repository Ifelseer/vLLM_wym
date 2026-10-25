"""vLLM_wym public API: ``from src import Engine, EngineConfig, SamplingParams``."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# 必须在任何 transformers import 之前：transformers 5.x 会引用 torch 2.8+ 才有的
# dtype 常量（我们环境是 torch 2.6），详见 src/_compat.py
from ._compat import install_transformers_shims

install_transformers_shims()

__all__ = ["Engine", "EngineConfig", "SamplingParams"]

if TYPE_CHECKING:
    from .config import EngineConfig
    from .control.engine import Engine
    from .sampling_params import SamplingParams

_LAZY = {
    "Engine": (".control.engine", "Engine"),
    "EngineConfig": (".config", "EngineConfig"),
    "SamplingParams": (".sampling_params", "SamplingParams"),
}


def __getattr__(name: str) -> Any:
    """懒加载：避免 ``import src`` 时立即拉起 torch/tokenizer 等重依赖。"""
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module, __name__), attr)
