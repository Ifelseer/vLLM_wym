# Vendored from deepseek-ai/DeepSpec (MIT License)
#   deepspec/utils/sampling.py::sample_tokens   @ main（2026-10-25）
#   deepspec/utils/metrics.py::add_metric       @ main（训练期打点，推理 no-op）
# 本地改动：只保留本引擎用到的两个函数；`add_metric` 降级为 no-op。
"""上游 `deepspec.utils` 的最小替代，避免把整个训练框架依赖进来。"""

from __future__ import annotations

from typing import Any

import torch


def sample_tokens(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """``[B, S, V]`` logits → ``[B, S]`` token id（temperature=0 即 argmax）。"""
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    flat_logits = logits.reshape(-1, vocab_size) / temperature
    probs = torch.softmax(flat_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).reshape(bsz, seq_len)


def add_metric(name: str, value: Any, *, den: Any = None, tag: str = "") -> None:
    """训练期指标打点。推理路径不需要，保留签名以免改动上游调用点。"""
    del name, value, den, tag
    return None


__all__ = ["add_metric", "sample_tokens"]
