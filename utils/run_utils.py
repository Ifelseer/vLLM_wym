"""Shared helpers for the run.py entrypoints."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "cuda_sync",
    "drain_engine",
    "dummy_token_ids",
    "handle_hub_cli",
    "load_config",
    "merge_cli",
    "print_bench_report",
    "warmup_engine",
]

_WARMUP_ITERS = 3


def load_config(path: Path, defaults: dict) -> dict:
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    return {k: cfg.get(k, v) for k, v in defaults.items()}


def merge_cli(cfg: dict, args: Any, keys: tuple[str, ...]) -> None:
    for key in keys:
        if (v := getattr(args, key, None)) is not None:
            cfg[key] = v


def handle_hub_cli(args: Any) -> bool:
    """Handle ``--list`` / ``--download``; return True if the command was handled."""
    if not getattr(args, "list", False) and not getattr(args, "download", False):
        return False

    from utils.model_hub import (
        DEFAULT_CACHE_DIR,
        QWEN3_REGISTRY,
        download_models,
        gpu_memory_gb,
        select_models_for_vram,
    )

    if args.list:
        vram = gpu_memory_gb() or 24.0
        selected = select_models_for_vram(vram_gb=vram)
        print(f"Cache: {DEFAULT_CACHE_DIR}")
        print(f"VRAM:  {vram:.1f} GB")
        print(f"All:   {', '.join(s.local_name for s in QWEN3_REGISTRY)}")
        print(f"Fit:   {', '.join(s.local_name for s in selected)}")
        return True

    print(f"Downloading to {DEFAULT_CACHE_DIR} ...")
    specs = getattr(args, "download_specs", None)
    download_models(list(specs) if specs is not None else None)
    print("Done.")
    return True


def cuda_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def dummy_token_ids(n: int, salt: int = 0) -> list[int]:
    n = max(1, n)
    return [((i * 7919 + 10007 + salt * 97) % 100000) + 100 for i in range(n)]


def drain_engine(is_finished: Callable[[], bool], step: Callable[[], None]) -> None:
    while not is_finished():
        step()


def warmup_engine(
    prompt_lens: list[int],
    *,
    add_request: Callable[[list[int], Any], None],
    drain: Callable[[], None],
    sync: Callable[[], None],
    sampling_params: Any,
    iters: int = _WARMUP_ITERS,
) -> float:
    sync()
    t0 = time.perf_counter()
    for it in range(iters):
        for n in prompt_lens:
            add_request(dummy_token_ids(n, salt=it * 31 + n), sampling_params)
        drain()
    sync()
    return time.perf_counter() - t0


def print_bench_report(
    ttft_s: float | None,
    n: int,
    e2e_s: float,
    *,
    decode_s: float | None = None,
    warmup_s: float | None = None,
) -> None:
    print()
    if warmup_s is not None:
        print(f"Warmup: {warmup_s:.2f}s  (compile + CUDA Graph capture, excluded)")
    if ttft_s is not None:
        print(f"TTFT:   {ttft_s * 1000:.1f} ms  (steady prefill + first token)")
    n_dec = max(0, n - 1)
    if decode_s is not None and decode_s > 0 and n_dec > 0:
        print(f"Decode: {n_dec} tokens in {decode_s:.3f}s ({n_dec / decode_s:.1f} tok/s)")
    elif n_dec == 0:
        print("Decode: (no tokens after the first; raise max_new_tokens)")
    if e2e_s > 0:
        print(f"E2E:    {n} tokens in {e2e_s:.3f}s ({n / e2e_s:.1f} tok/s)")
