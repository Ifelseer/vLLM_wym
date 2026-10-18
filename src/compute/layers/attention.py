from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
import torch.nn as nn

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    _HAS_FLASH_ATTN = True
except ImportError:  # pragma: no cover - 开发机（Windows/CPU）无 flash_attn
    _HAS_FLASH_ATTN = False

from ..kv_ops import store_kvcache, store_kvcache_fp8


def _require_flash_attn() -> None:
    if not _HAS_FLASH_ATTN:
        raise RuntimeError(
            "flash-attn is required by vllm::attention; "
            "install flash-attn on a CUDA machine"
        )

class AttnInputs(NamedTuple):
    slot_mapping: torch.Tensor
    is_prefill: bool
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    block_tables: Optional[torch.Tensor] = None
    cache_seqlens: Optional[torch.Tensor] = None


@dataclass
class AttnMetadata:
    kv_cache: torch.Tensor
    # FP8 KV cache（None = BF16 KV，走 FA2 原路径）
    fp8_kv: object = None
    # decode CUDA Graph replay 中：True 时用 CG wrapper（plan 已在录图外完成）
    fp8_kv_cg: bool = False


_ATTN_META: Optional[AttnMetadata] = None


@contextmanager
def attention_context(meta: AttnMetadata):
    global _ATTN_META
    prev = _ATTN_META
    _ATTN_META = meta
    try:
        yield
    finally:
        _ATTN_META = prev


@torch.library.custom_op("vllm::attention", mutates_args=())
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slot_mapping: torch.Tensor,
    is_prefill: bool,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_tables: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    layer_id: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    meta = _ATTN_META
    assert meta is not None, "attention_context() not active"
    pool = meta.kv_cache
    k_cache = pool[0, layer_id]
    v_cache = pool[1, layer_id]
    if meta.fp8_kv is not None:
        return _attention_fp8(
            meta, q, k, v, k_cache, v_cache, slot_mapping, is_prefill,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            block_tables, layer_id, num_qo_heads, num_kv_heads, head_dim,
        )
    _require_flash_attn()
    store_kvcache(
        k,
        v,
        k_cache.reshape(-1, num_kv_heads, head_dim),
        v_cache.reshape(-1, num_kv_heads, head_dim),
        slot_mapping,
    )
    q = q.contiguous().view(-1, num_qo_heads, head_dim)
    if is_prefill:
        if block_tables is not None:
            k, v = k_cache, v_cache
        else:
            k = k.contiguous().view(-1, num_kv_heads, head_dim)
            v = v.contiguous().view(-1, num_kv_heads, head_dim)
        o = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=True,
            block_table=block_tables,
        )
    else:
        o = flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_tables,
            causal=True,
        )
    return o.view(-1, num_qo_heads * head_dim)


@attention.register_fake
def _attention_fake(
    q, k, v, slot_mapping, is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
    max_seqlen_k, block_tables, cache_seqlens, layer_id, num_qo_heads,
    num_kv_heads, head_dim,
):
    return torch.empty(
        (q.shape[0], num_qo_heads * head_dim), dtype=q.dtype, device=q.device
    )


def _attention_fp8(
    meta: AttnMetadata,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    is_prefill: bool,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_tables: Optional[torch.Tensor],
    layer_id: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """FP8 KV cache 通路：量化写入 + FlashInfer paged attention。

    - 首次写入该层时按 amax 校准固定 scale（之后冻结，保证增量 KV 一致）；
    - prefill 无前缀缓存的首 chunk 仍走 FA2 varlen（直接读当前批 bf16 k/v，
      不碰 cache）；有缓存时走 FlashInfer paged prefill（kv fp8 + scale）；
    - decode 走 FlashInfer paged decode（plan 已在 ModelRunner 录图外完成）。
    """
    mgr = meta.fp8_kv
    k_scale, v_scale = mgr.scales(layer_id)
    mgr.maybe_calibrate(layer_id, k, v)
    k_scale, v_scale = mgr.scales(layer_id)  # 校准后重新取（值可能已更新）

    nkv_hd = (num_kv_heads, head_dim)
    store_kvcache_fp8(
        k,
        v,
        k_cache.reshape(-1, *nkv_hd),
        v_cache.reshape(-1, *nkv_hd),
        slot_mapping,
        k_scale,
        v_scale,
    )
    q = q.contiguous().view(-1, num_qo_heads, head_dim)
    if is_prefill:
        if block_tables is not None:
            # 有前缀缓存：全部 KV（缓存 + 本 chunk）从 fp8 cache 读
            o = mgr.run_prefill(q, layer_id)
        else:
            # 首 chunk：直接读当前批 bf16 k/v（FA2，无需 scale）
            _require_flash_attn()
            kk = k.contiguous().view(-1, *nkv_hd)
            vv = v.contiguous().view(-1, *nkv_hd)
            from flash_attn import flash_attn_varlen_func

            o = flash_attn_varlen_func(
                q,
                kk,
                vv,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
            )
    else:
        if meta.fp8_kv_cg:
            o = mgr.run_decode_cg(q, layer_id)
        else:
            o = mgr.run_decode(q, layer_id)
    return o.view(-1, num_qo_heads * head_dim)


class Attention(nn.Module):
    def __init__(
        self, num_heads: int, head_dim: int, num_kv_heads: int, layer_id: int = 0
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.layer_id = layer_id

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn: AttnInputs
    ) -> torch.Tensor:
        return torch.ops.vllm.attention(
            q,
            k,
            v,
            attn.slot_mapping,
            attn.is_prefill,
            attn.cu_seqlens_q,
            attn.cu_seqlens_k,
            attn.max_seqlen_q,
            attn.max_seqlen_k,
            attn.block_tables,
            attn.cache_seqlens,
            self.layer_id,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )
