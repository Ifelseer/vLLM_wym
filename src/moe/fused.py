"""Fused MoE forward: FlashInfer cutlass (SM89+) or Triton GEMM.

FP8 block-scale 路径（``w13_scale`` 非 None）：
- FlashInfer cutlass 的 block-scale FP8 MoE 仅 SM90+ 支持（vLLM 同款门控），
  SM89 上强制走 Triton grouped GEMM（block scale 版，见 kernels/triton.py）；
- cutlass 的 per-tensor FP8（quant_scales）路径保留给后续 per-tensor checkpoint。
"""

from __future__ import annotations

import torch

from .kernels import clear_cuda_state
from .kernels import triton as triton_moe

_IDS: dict[tuple, torch.Tensor] = {}
_SCALES: dict[tuple, torch.Tensor] = {}
_OUT: dict[tuple, torch.Tensor] = {}


def clear_fused_state() -> None:
    _IDS.clear()
    _SCALES.clear()
    _OUT.clear()
    triton_moe.clear_state()
    clear_cuda_state()


def _scratch(cache, key, *, shape, dtype, device):
    buf = cache.get(key)
    if buf is None or buf.shape != shape or buf.dtype != dtype or buf.device != device:
        buf = torch.empty(shape, dtype=dtype, device=device)
        cache[key] = buf
    return buf


def _cutlass(
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
) -> torch.Tensor:
    from flashinfer import cutlass_fused_moe
    from flashinfer.tllm_enums import ActivationType

    t, k = hidden_states.shape[0], topk_ids.shape[1]
    h = hidden_states.shape[1]
    dev = hidden_states.device
    di = dev.index or 0
    out = _scratch(
        _OUT, (di, t, h, hidden_states.dtype), shape=(t, h), dtype=hidden_states.dtype, device=dev
    )
    ids = _scratch(_IDS, (di, t, k), shape=(t, k), dtype=torch.int32, device=dev)
    scales = _scratch(
        _SCALES, (di, t, k), shape=(t, k), dtype=torch.float32, device=dev
    )
    ids.copy_(topk_ids.to(torch.int32))
    scales.copy_(topk_weights.float())
    result = cutlass_fused_moe(
        hidden_states,
        ids,
        scales,
        w13_weight,
        w2_weight,
        hidden_states.dtype,
        quant_scales=None,
        output=out,
        ep_size=1,
        ep_rank=0,
        activation_type=ActivationType.Swiglu,
    )
    return result[0] if isinstance(result, (list, tuple)) else result


def fused_moe_forward(
    *,
    backend: str,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    quantized = w13_scale is not None
    if quantized:
        # FP8 block-scale：cutlass 仅 SM90 支持 block scale → 统一 Triton
        if backend == "cutlass":
            major, minor = torch.cuda.get_device_capability(hidden_states.device)
            if major * 10 + minor >= 90:
                raise NotImplementedError(
                    "fp8 block-scale MoE via cutlass (SM90+) is not wired yet; "
                    "use --moe-backend triton"
                )
            # SM89 及以下：Triton block-scale grouped GEMM
        return triton_moe.forward(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
        )
    if backend == "cutlass":
        return _cutlass(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13_weight=w13_weight,
            w2_weight=w2_weight,
        )
    if backend == "triton":
        return triton_moe.forward(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13_weight=w13_weight,
            w2_weight=w2_weight,
        )
    raise ValueError(f"unknown MoE backend: {backend!r}")
