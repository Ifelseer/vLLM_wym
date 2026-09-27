"""Local HF model paths."""

from __future__ import annotations

from pathlib import Path

from src.utils.paths import resolve_hf_home

HF_HOME = resolve_hf_home()


def _model(*candidates: Path) -> Path:
    for p in candidates:
        if p.is_dir() and any(p.glob("*.safetensors")):
            return p
    return candidates[0]


DENSE_MODEL = _model(HF_HOME / "Qwen3-0.6B")
MOE_MODEL = _model(
    HF_HOME / "Qwen1.5-MoE-A2.7B",
    HF_HOME / "Qwen1.5-MoE-A2.7B-Chat",
)
MOE_30B = _model(HF_HOME / "Qwen3-30B-A3B", HF_HOME / "Qwen3-30B-A3B-Instruct")


def require_model(path: Path):
    import pytest

    ok = path.is_dir() and any(path.glob("*.safetensors"))
    return pytest.mark.skipif(not ok, reason=f"missing model: {path}")
