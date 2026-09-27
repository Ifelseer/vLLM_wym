"""JIT CUDA aux kernels for MoE: align / silu / moe_sum.

Source lives next to this file (``moe_kernels.cu``). Main GEMMs stay Triton/cutlass.

Buffers are pooled by shape so decode CUDA Graph capture sees a fixed
allocation count (same approach as vLLM: no fresh ``cudaMalloc`` per step).
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

_CSRC = Path(__file__).resolve().parent / "moe_kernels.cu"

_ALIGN: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_SILU: dict[tuple, torch.Tensor] = {}
_SUM: dict[tuple, torch.Tensor] = {}


def clear_state() -> None:
    _ALIGN.clear()
    _SILU.clear()
    _SUM.clear()


def align_capacity(n: int, num_experts: int, block_size: int) -> int:
    """Upper bound on ``sorted_ids`` length for fixed ``n = M * top_k``."""
    if n <= 0:
        return 0
    cap = n + num_experts * (block_size - 1)
    if n < num_experts:
        cap = min(n * block_size, cap)
    return int(cap)


def _pool(cache: dict, key: tuple, *, shape, dtype, device) -> torch.Tensor:
    buf = cache.get(key)
    if buf is None or buf.shape != shape or buf.dtype != dtype or buf.device != device:
        buf = torch.empty(shape, dtype=dtype, device=device)
        cache[key] = buf
    return buf


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    if os.environ.get("VLLM_MOE_CUDA", "1") in ("0", "false", "False"):
        return False
    try:
        _ext()
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

    if not (os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")):
        for cand in ("/usr/local/cuda-12.8", "/usr/local/cuda"):
            if Path(cand).exists():
                os.environ["CUDA_HOME"] = cand
                break
    for cand in ("/usr/local/cuda-12.8/bin", "/usr/local/cuda/bin"):
        if Path(cand, "nvcc").exists():
            path = os.environ.get("PATH", "")
            if cand not in path.split(os.pathsep):
                os.environ["PATH"] = cand + os.pathsep + path
            break

    _which = shutil.which

    def _no_ccache(cmd, *args, **kwargs):
        if cmd == "ccache":
            return None
        return _which(cmd, *args, **kwargs)

    shutil.which = _no_ccache  # type: ignore[assignment]
    try:
        return load(
            name="vllm_moe_cuda",
            sources=[str(_CSRC)],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-gencode=arch=compute_80,code=sm_80",
                "-gencode=arch=compute_86,code=sm_86",
                "-gencode=arch=compute_89,code=sm_89",
                "-gencode=arch=compute_120a,code=sm_120a",
            ],
            verbose=bool(os.environ.get("VLLM_MOE_CUDA_VERBOSE")),
            with_cuda=True,
        )
    finally:
        shutil.which = _which  # type: ignore[assignment]


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """``[up|gate]`` → ``up * silu(gate)``."""
    out_shape = (*x.shape[:-1], x.shape[-1] // 2)
    if x.is_cuda and cuda_available():
        key = (x.device.index or 0, *out_shape, x.dtype)
        out = _pool(_SILU, key, shape=out_shape, dtype=x.dtype, device=x.device)
        _ext().mul_and_silu(out, x.contiguous())
        return out
    up, gate = x.chunk(2, dim=-1)
    return up * F.silu(gate)


def moe_sum(mid: torch.Tensor, M: int, top_k: int) -> torch.Tensor:
    """Reduce ``[M*top_k, H]`` → ``[M, H]``."""
    H = mid.shape[-1]
    if M == 0:
        return mid.new_empty(0, H)
    key = (mid.device.index or 0, M, H, mid.dtype)
    out = _pool(_SUM, key, shape=(M, H), dtype=mid.dtype, device=mid.device)
    if mid.is_cuda and cuda_available():
        _ext().moe_sum(mid.view(M, top_k, H), out)
        return out
    return mid.view(M, top_k, H).sum(dim=1, out=out)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack tokens per expert. Returns ``(sorted_ids, expert_ids, num_post)``.

    ``sorted_ids`` / ``expert_ids`` lengths are the *capacity* for this
    ``(n, E, block_size)``, not the live ``num_post``. Capacity is fixed when
    decode pads to a captured batch size, which is what CUDA Graph needs.
    """
    flat = topk_ids.reshape(-1)
    n = flat.numel()
    device = topk_ids.device
    if n == 0:
        z = torch.zeros(0, dtype=torch.int32, device=device)
        return z, z, torch.zeros(1, dtype=torch.int32, device=device)

    if device.type == "cuda" and cuda_available():
        cap = align_capacity(n, num_experts, block_size)
        n_blocks = (cap + block_size - 1) // block_size
        key = (device.index or 0, n, num_experts, block_size)
        bufs = _ALIGN.get(key)
        if (
            bufs is None
            or bufs[0].numel() != cap
            or bufs[1].numel() != n_blocks
            or bufs[0].device != device
        ):
            sorted_ids = torch.empty(cap, dtype=torch.int32, device=device)
            expert_ids = torch.empty(n_blocks, dtype=torch.int32, device=device)
            num_post = torch.empty(1, dtype=torch.int32, device=device)
            _ALIGN[key] = (sorted_ids, expert_ids, num_post)
        else:
            sorted_ids, expert_ids, num_post = bufs
        ids = flat if flat.dtype in (torch.int32, torch.int64) else flat.to(torch.int32)
        _ext().moe_align_block_size(
            ids.contiguous(),
            int(num_experts),
            int(block_size),
            sorted_ids,
            expert_ids,
            num_post,
        )
        return sorted_ids, expert_ids, num_post

    return _align_torch(flat, block_size, num_experts)


def _align_torch(
    flat: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch fallback when the CUDA extension is unavailable."""
    device = flat.device
    flat = flat.to(torch.int32).contiguous()
    n = flat.numel()
    valid = (flat >= 0) & (flat < num_experts)
    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    if bool(valid.any()):
        counts = torch.bincount(flat[valid], minlength=num_experts).to(torch.int32)
    padded = ((counts + block_size - 1) // block_size) * block_size
    starts = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    starts[0] = 0
    torch.cumsum(padded, dim=0, out=starts[1:])
    num_post = starts[-1:].contiguous()
    cap = align_capacity(n, num_experts, block_size)
    sorted_ids = torch.full((cap,), n, dtype=torch.int32, device=device)
    n_blocks = (cap + block_size - 1) // block_size
    if bool(valid.any()):
        order = torch.argsort(flat, stable=True)
        keep = (flat[order] >= 0) & (flat[order] < num_experts)
        order = order[keep]
        experts = flat[order]
        compact = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        compact[0] = 0
        torch.cumsum(counts, dim=0, out=compact[1:])
        arange = torch.arange(order.numel(), device=device, dtype=torch.int32)
        dest = starts[experts] + (arange - compact[experts])
        sorted_ids[dest.long()] = order.to(torch.int32)
    token_pos = torch.arange(n_blocks, device=device, dtype=torch.int32) * block_size
    expert_ids = torch.searchsorted(starts[1:].contiguous(), token_pos, right=True).to(
        torch.int32
    )
    expert_ids = torch.where(token_pos < num_post, expert_ids, expert_ids.new_full((), -1))
    return sorted_ids, expert_ids, num_post
