"""Path helpers for tests."""

from __future__ import annotations

from ..model_hub import DEFAULT_CACHE_DIR, is_local_model, resolve_model_path

__all__ = [
    "DEFAULT_CACHE_DIR",
    "HUGGINGFACE_HUB_CACHE",
    "is_local_model",
    "resolve_hf_home",
    "resolve_model_path",
]


def resolve_hf_home():
    return DEFAULT_CACHE_DIR


HUGGINGFACE_HUB_CACHE = DEFAULT_CACHE_DIR
