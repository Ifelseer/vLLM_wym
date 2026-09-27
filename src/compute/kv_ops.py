from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    key_offsets = idx * key_stride + offsets
    value_offsets = idx * value_stride + offsets
    key = tl.load(key_ptr + key_offsets, mask=mask, other=0.0)
    value = tl.load(value_ptr + value_offsets, mask=mask, other=0.0)
    cache_offsets = slot * D + offsets
    tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, value, mask=mask)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()

    n, num_heads, head_dim = key.shape
    d = num_heads * head_dim
    assert slot_mapping.numel() == n

    block = triton.next_power_of_2(d)
    _store_kvcache_kernel[(n,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        d,
        block,
    )
