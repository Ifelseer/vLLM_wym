"""vLLM_wym 模型路径与下载注册表。

注册表分两部分：

- ``QWEN3_REGISTRY``：BF16 基线模型（回归基线 / 在线量化的输入）。
- ``QUANT_REGISTRY``：社区**预量化** checkpoint（Phase 0 核验，检索日
  2026-10-12），按 ``quant`` 字段标注格式；``size_gb`` 据此估算实际权重
  显存（bf16 的 1/2 或 1/4），``select_models_for_vram`` 使用有效值而非
  ``fp16_gb`` 做预算。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

DEFAULT_CACHE_DIR = Path.home() / "huggingface"
RTX_4090_VRAM_GB = 24.0

# 格式 → 权重字节数相对 BF16 的比例（不含 scale/激活/KV）
_QUANT_BYTES_FACTOR = {
    "none": 1.0,
    "fp8": 0.5,
    "int8": 0.5,
    "awq": 0.25,
    "gptq": 0.25,
}

__all__ = [
    "DEFAULT_CACHE_DIR",
    "ModelSpec",
    "QWEN3_REGISTRY",
    "QUANT_REGISTRY",
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
    # 预量化格式：none / fp8 / int8 / awq / gptq（none = BF16 基线）
    quant: str = "none"
    # 备注（如 scale 模式、来源核验日期）
    note: str = ""

    @property
    def size_gb(self) -> float:
        """该 checkpoint 的**有效权重显存**（GB，不含激活/KV/scale）。"""
        return self.fp16_gb * _QUANT_BYTES_FACTOR.get(self.quant, 1.0)


QWEN3_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec("Qwen3-0.6B", "Qwen/Qwen3-0.6B", 1.2),
    ModelSpec("Qwen3-1.7B", "Qwen/Qwen3-1.7B", 3.4),
    ModelSpec("Qwen3-4B", "Qwen/Qwen3-4B", 8.0),
)

# 社区预量化 checkpoint（Phase 0 核验存在性与 scale 格式，检索日 2026-10-12）
QUANT_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec(
        "Qwen3-0.6B-FP8", "Qwen/Qwen3-0.6B-FP8", 1.2, quant="fp8",
        note="block 128x128 + dynamic act; modules_to_not_convert: lm_head/norm/router",
    ),
    ModelSpec(
        "Qwen3-30B-A3B-FP8", "Qwen/Qwen3-30B-A3B-FP8", 61.0, quant="fp8",
        note="MoE；SM89 走 Triton block-scale grouped GEMM（cutlass 仅 SM90）",
    ),
    ModelSpec(
        "Qwen3-32B-AWQ", "Qwen/Qwen3-32B-AWQ", 64.0, quant="awq",
        note="W4A16 group=128 zero_point=true version=gemm；24GB 单卡可跑",
    ),
    ModelSpec(
        "Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4",
        "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4",
        28.6,
        quant="gptq",
        note="sym=true desc_act=false exllama v2；MoE 专家 INT4 已验证",
    ),
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
    registry: tuple[ModelSpec, ...] | None = None,
) -> list[ModelSpec]:
    """返回可装入显存的注册表条目（按**有效权重显存** ``size_gb`` 预算）。

    默认同时考虑 BF16 基线与预量化 checkpoint——同一模型的不同格式
    会给出不同的可装性结论（如 Qwen3-32B 的 AWQ 版可进 24GB）。
    """
    items = registry if registry is not None else (QWEN3_REGISTRY + QUANT_REGISTRY)
    budget = (
        vram_gb if vram_gb is not None else gpu_memory_gb() or RTX_4090_VRAM_GB
    ) - reserve_gb
    return [s for s in items if s.size_gb <= budget]
