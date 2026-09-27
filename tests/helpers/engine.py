"""Engine one-shot helpers for tests / bench."""

from __future__ import annotations

import gc
import os
from contextlib import contextmanager
from typing import Any, Iterator

import torch

from src import Engine, EngineConfig, SamplingParams

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def free_cuda() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def cfg(model: str | os.PathLike, **kwargs: Any) -> EngineConfig:
    defaults: dict[str, Any] = dict(
        model=str(model),
        enforce_eager=True,
        max_num_seqs=4,
        context_len=1024,
        num_kvcache_blocks=32,
        gpu_memory_utilization=0.75,
        prefix_backend="none",
    )
    defaults.update(kwargs)
    return EngineConfig(**defaults)


@contextmanager
def engine_session(config: EngineConfig) -> Iterator[Engine]:
    free_cuda()
    eng = Engine(config)
    try:
        yield eng
    finally:
        eng.destroy()
        free_cuda()


def greedy_tokens(
    config: EngineConfig,
    prompt: str | list[int],
    *,
    n: int = 8,
) -> list[int]:
    with engine_session(config) as eng:
        seq = eng.add_request(
            prompt,
            SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True),
        )
        while not seq.is_finished:
            eng.step()
        return list(seq.token_ids[seq.num_prompt_tokens :])
