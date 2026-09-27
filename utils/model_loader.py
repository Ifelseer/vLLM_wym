from __future__ import annotations

import concurrent.futures
import glob
import os
from collections.abc import Iterable
from contextlib import contextmanager

import torch
from safetensors.torch import load_file
from torch import nn
from tqdm import tqdm

__all__ = [
    "default_dtype_context",
    "default_weight_loader",
    "load_model",
    "safetensors_weights_iterator",
    "skip_param_init",
]


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight)


@contextmanager
def skip_param_init():
    """Skip random init during module construction; weights are loaded immediately after."""
    orig_linear = nn.Linear.reset_parameters
    orig_embed = nn.Embedding.reset_parameters
    nn.Linear.reset_parameters = lambda self: None
    nn.Embedding.reset_parameters = lambda self: None
    try:
        yield
    finally:
        nn.Linear.reset_parameters = orig_linear
        nn.Embedding.reset_parameters = orig_embed


@contextmanager
def default_dtype_context(dtype: torch.dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def _load_shard_cpu(path: str) -> list[tuple[str, torch.Tensor]]:
    return list(load_file(path, device="cpu").items())


def safetensors_weights_iterator(hf_folder: str) -> Iterable[tuple[str, torch.Tensor]]:
    files = sorted(glob.glob(os.path.join(hf_folder, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {hf_folder}")

    if len(files) == 1:
        for st_file in tqdm(files, desc="Loading weights"):
            yield from _load_shard_cpu(st_file)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_load_shard_cpu, files[0])
        for i, _st_file in enumerate(tqdm(files, desc="Loading weights")):
            items = fut.result()
            if i + 1 < len(files):
                fut = pool.submit(_load_shard_cpu, files[i + 1])
            yield from items


def load_model(model: nn.Module, hf_folder: str) -> nn.Module:
    model.load_weights(safetensors_weights_iterator(hf_folder))
    return model
