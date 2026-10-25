"""第三方版本的兼容垫片（必须在任何 transformers import 之前执行）。

## 为什么需要

DSpark 草稿模型借用 transformers 的 Qwen3 组件（`Qwen3RotaryEmbedding` /
`sdpa_attention_forward` 等）。实测（`docs/dspark-spec.md` §6.4）：同一份权重、
同一份输入，**transformers 4.57 与 5.10 的块前向相差 33%**，导致草稿接受率从
6.54 掉到 5.53。所以项目对 DSpark 要求 `transformers >= 5.0`。

而 transformers 5.x 在 import 期会引用 `torch.float8_e8m0fnu`——这个 dtype 是
torch 2.8+ 才有的，我们的环境是 **torch 2.6.0a0（NGC）**，直接 import 会 AttributeError。

## 为什么可以安全地占位

`float8_e8m0fnu` 在我们这条路径上**只作为一个 dtype 常量被引用**（transformers 的
量化配置映射表），从来没有真的分配过这种张量——我们的量化走 `src/quant` 自研实现，
不用 transformers 的 quantization 分支。所以把它映射到已有的 fp8 dtype 只是让 import
通过，不影响任何计算。

配套的环境要求（`transformers==5.10.2`、`huggingface_hub==1.33.0`）见
`docs/dspark-spec.md` §6.4；torch / flash-attn / flashinfer 都不需要动。

## 用法

`src/__init__.py` 在**第一行**调用本模块，因此任何 `from src import ...` 都会先装好
垫片。若某个入口绕过了 `src`（例如直接 import transformers 的脚本），在 `transformers`
之前显式 `import src._compat` 即可，或用 sitecustomize。
"""

from __future__ import annotations

import torch

__all__ = ["install_transformers_shims"]


def install_transformers_shims() -> None:
    """补齐 torch 2.8+ 才有的 dtype 常量（幂等，可重复调用）。"""
    if not hasattr(torch, "float8_e8m0fnu"):
        # 占位：只在 transformers 的 dtype 映射表里被引用，不会被实际分配
        torch.float8_e8m0fnu = torch.float8_e4m3fn  # type: ignore[attr-defined]
