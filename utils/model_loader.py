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
    "Fp8FallbackCatcher",
    "filter_unloaded_scales",
]


def filter_unloaded_scales(model, unloaded: set[str]) -> set[str]:
    """在线量化（quantize_at_load）路径下，``weight_scale`` 参数由
    ``finalize_model`` 统一生成，不算「未加载」，避免误导性 WARN。
    预量化路径不过滤（缺 scale 张量即真异常）。
    """
    online = any(
        getattr(m, "quant_method", None) is not None
        and getattr(m.quant_method.config, "quantize_at_load", False)
        for m in model.modules()
    )
    if not online:
        return unloaded
    return {n for n in unloaded if not n.endswith("weight_scale")}


class Fp8FallbackCatcher:
    """量化 checkpoint 中「不量化模块」（lm_head / embed_tokens）的 fp8 权重处理。

    社区 FP8 checkpoint（如 Qwen3-30B-A3B-FP8）可能对 embedding/lm_head 也存
    fp8 权重 + ``*.weight_scale``；本引擎这两个模块保持高精度，因此在
    ``load_weights`` 中把 fp8 权重暂存，待对应 scale 到达后反量化写回。

    用法（模型 load_weights 内）::

        catcher = Fp8FallbackCatcher(params)   # params = dict(named_parameters())
        for name, w in weights:
            ...
            if catcher.is_pending_name(name):
                catcher.feed(name, w, load_fn=get_weight_loader)
                continue
    """

    def __init__(self, params: dict) -> None:
        self._params = params
        self._fp8: dict[str, torch.Tensor] = {}
        self._scale: dict[str, torch.Tensor] = {}
        self.written: list[str] = []  # 已反量化写回的参数名

    @staticmethod
    def is_pending_name(name: str) -> bool:
        """是否是可能归本 catcher 管的名字（fp8 权重或其 scale）。

        注意 HF FP8 checkpoint 的 scale 后缀是 ``_scale``（如
        ``...embed_tokens.weight_scale``），不是 ``.weight_scale``。
        """
        return name.endswith(".weight") or name.endswith("weight_scale")

    def feed(self, name: str, w: torch.Tensor, *, load_fn) -> bool:
        """喂入一个 (name, tensor)；返回 True 表示已消费（调用方应 continue）。

        load_fn: 取 ``get_weight_loader(param)``，反量化完成后经它写回
        （保留 embedding 的 padding 行 / TP 切分逻辑）。
        """
        if name.endswith("weight_scale"):
            stem = name[: -len("_scale")]
            self._scale[stem] = w.float()
            pend = self._fp8.pop(stem, None)
            if pend is not None:
                self._write(stem, pend, self._scale.pop(stem), load_fn)
            return True
        # name.endswith(".weight")
        param = self._params.get(name)
        if param is None or param.dtype == w.dtype:
            return False  # 不归本 catcher 管（走调用方原路径）
        if w.dtype != torch.float8_e4m3fn:
            return False
        s = self._scale.pop(name, None)
        if s is not None:
            self._write(name, w, s, load_fn)
        else:
            self._fp8[name] = w  # 等 scale 到达
        return True

    def _write(self, stem: str, w_fp8: torch.Tensor, scale: torch.Tensor, load_fn) -> None:
        from src.quant.kernels.fp8_gemm import dequantize_fp8_per_block

        param = self._params[stem]
        deq = dequantize_fp8_per_block(w_fp8, scale, param.dtype)
        load_fn(param, deq)
        self.written.append(stem)

    def unresolved(self) -> list[str]:
        """加载结束后仍缺 scale 的 fp8 权重名（非空说明 checkpoint 异常）。"""
        return sorted(self._fp8)


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
    # 量化收尾：在线量化路径把 staging 统一转 fp8 + scale（无量化时为 no-op）
    from src.quant import finalize_model

    finalize_model(model)
    return model
