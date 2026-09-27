"""Parallel layout checks."""

from __future__ import annotations

import os
from pathlib import Path

from transformers import AutoConfig


def _text(model: str | os.PathLike[str]):
    c = AutoConfig.from_pretrained(str(model), trust_remote_code=True)
    return getattr(c, "text_config", c)


def layout_ok(
    model: str | os.PathLike[str],
    *,
    tp: int = 1,
    enable_ep: bool = False,
) -> tuple[bool, str]:
    t = _text(model)
    n_heads = int(t.num_attention_heads)
    n_kv = int(t.num_key_value_heads)
    inter = int(t.intermediate_size)
    if n_heads % tp != 0:
        return False, f"heads={n_heads} % tp={tp} != 0"
    if n_kv >= tp:
        if n_kv % tp != 0:
            return False, f"kv={n_kv} % tp={tp} != 0"
    elif tp % n_kv != 0:
        return False, f"tp={tp} % kv={n_kv} != 0"
    if inter % tp != 0:
        return False, f"intermediate={inter} % tp={tp} != 0"
    moe_inter = getattr(t, "moe_intermediate_size", None)
    n_experts = int(getattr(t, "num_experts", 0) or 0)
    if moe_inter is not None and n_experts > 0:
        if enable_ep:
            if n_experts % tp != 0:
                return False, f"experts={n_experts} % tp={tp} != 0"
        else:
            if int(moe_inter) % tp != 0:
                return False, f"moe_inter={moe_inter} % tp={tp} != 0"
    return True, "ok"
