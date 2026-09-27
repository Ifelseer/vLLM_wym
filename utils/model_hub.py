"""vLLM_wym 模型路径与下载注册表。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

DEFAULT_CACHE_DIR = Path.home() / "huggingface"
RTX_4090_VRAM_GB = 24.0

__all__ = [
    "DEFAULT_CACHE_DIR",
    "ModelSpec",
    "QWEN3_REGISTRY",
    "RTX_4090_VRAM_GB",
    "download_model",
    "download_models",
    "gpu_memory_gb",
    "is_downloaded",
    "is_local_model",
    "resolve_model_path",
    "select_models_for_vram",
]


@dataclass(frozen=True)
class ModelSpec:
    local_name: str
    repo_id: str
    fp16_gb: float


QWEN3_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec("Qwen3-0.6B", "Qwen/Qwen3-0.6B", 1.2),
    ModelSpec("Qwen3-1.7B", "Qwen/Qwen3-1.7B", 3.4),
    ModelSpec("Qwen3-4B", "Qwen/Qwen3-4B", 8.0),
)


def is_local_model(path: Path) -> bool:
    return path.is_dir() and (
        (path / "config.json").is_file() or any(path.glob("*.safetensors"))
    )


def resolve_model_path(model: str | os.PathLike[str]) -> Path:
    """显式本地目录优先，否则解析为 ~/huggingface/<name>。"""
    path = Path(model).expanduser()
    if is_local_model(path):
        return path
    return DEFAULT_CACHE_DIR / path.name


def is_downloaded(spec: ModelSpec, cache_dir: Path | None = None) -> bool:
    return is_local_model((cache_dir or DEFAULT_CACHE_DIR) / spec.local_name)


def download_model(spec: ModelSpec, cache_dir: Path | None = None) -> Path:
    root = cache_dir or DEFAULT_CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    target = root / spec.local_name
    if is_downloaded(spec, root):
        print(f"[skip] {spec.local_name} already at {target}")
        return target
    print(f"[download] {spec.repo_id} -> {target}")
    snapshot_download(
        spec.repo_id,
        local_dir=str(target),
        token=os.environ.get("HF_TOKEN"),
        endpoint=os.environ.get("HF_ENDPOINT") or None,
    )
    return target


def download_models(
    specs: list[ModelSpec] | None = None,
    cache_dir: Path | None = None,
) -> list[Path]:
    items = specs or list(QWEN3_REGISTRY)
    return [download_model(s, cache_dir) for s in items]


def gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def select_models_for_vram(
    vram_gb: float | None = None,
    reserve_gb: float = 4.0,
    registry: tuple[ModelSpec, ...] = QWEN3_REGISTRY,
) -> list[ModelSpec]:
    """返回单模型可装入显存的注册表条目。"""
    budget = (vram_gb if vram_gb is not None else gpu_memory_gb() or RTX_4090_VRAM_GB) - reserve_gb
    return [s for s in registry if s.fp16_gb <= budget]
