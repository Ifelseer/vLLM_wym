"""FP8 KV cache 的 FlashInfer attention 通路（SM89+）。

背景：FA2（``flash_attn_with_kvcache`` / ``flash_attn_varlen_func``）不支持
FP8 KV cache（无 k_scale/v_scale）；FlashInfer 0.6.15 的 paged attention
wrapper 原生支持 ``kv_data_type=float8_e4m3fn`` + ``k_scale/v_scale``。
因此 KV FP8 时 attention 切到 FlashInfer wrapper，KV 布局不变
（``[num_blocks, page_size, n_kv, head_dim]`` NHD，与 wrapper 的 4-D
paged cache 视图完全一致）。

设计要点：

1. **scale 语义**：per-layer 固定标量（首次写入时按该层 k/v 的 amax 校准，
   留 2x 头部余量后冻结）。增量 KV 的正确性要求同一 slot 的写/读 scale 一致，
   per-step 动态 scale 不适用；固定 scale 同时保证 CUDA Graph replay 安全
   （capture 前完成校准，captured 区域不再写 scale）。
2. **CSR 转换**：引擎内部用 FA2 风格 ``block_tables [bs, mb]`` +
   ``cache_seqlens``，FlashInfer 需要 ``(indptr, indices, last_page_len)``。
   转换全 GPU 完成、零 host 同步（indices 用预分配缓冲，indptr 之外的条目
   不会被 kernel 读取）。
3. **plan 在录图外**：FlashInfer 的 ``plan`` 内部有 D2H 拷贝，不能进
   CUDA Graph；decode 录图时 ``capturing=True`` 跳过 plan（warmup 已执行过），
   正常 replay 前在 ``_run_decode`` 里显式 plan。
4. **CG wrapper 按 batch size 各建一个**：``CUDAGraphBatchDecode...`` 的
   batch size 在构造时固定（由缓冲长度决定），与引擎按 bs 列表录图对齐。
"""

from __future__ import annotations

import torch

FP8_KV_DTYPE = torch.float8_e4m3fn
_FP8_MAX = 448.0
# FlashInfer 推荐 workspace（split-K 中间结果）；decode/prefill 共用
_WORKSPACE_BYTES = 128 * 1024 * 1024
# scale 校准的安全余量：只用一半动态范围，抗激活离群值
_SCALE_HEADROOM = 2.0
_SCALE_FLOOR = 1e-4


def block_tables_to_csr(
    block_tables: torch.Tensor,
    seqlens: torch.Tensor,
    page_size: int,
    out_indptr: torch.Tensor,
    out_indices: torch.Tensor,
    out_last_len: torch.Tensor,
) -> None:
    """FA2 block_tables + seqlens → FlashInfer CSR（零同步，写预分配缓冲）。

    indices 为**紧凑 packing**（seq i 的页连续存放于 indptr[i]:indptr[i+1]），
    由位置散射写入；无效条目散射到缓冲区末尾的 trash 槽（不被读取）。

    Args:
        block_tables: [bs, mb] int32 GPU（FA2 页表）。
        seqlens: [bs] int32 GPU（每序列 KV 长度）。
        page_size: 页大小（= kvcache_block_size，256）。
        out_indptr: [>=bs+1] int32 缓冲（写 [0..bs]）。
        out_indices: [>=bs*mb+1] int32 缓冲（末尾留 1 个 trash 槽）。
        out_last_len: [>=bs] int32 缓冲。
    """
    bs, mb = block_tables.shape
    dev = block_tables.device
    bt = block_tables.to(torch.int32)
    n_blocks = ((seqlens + page_size - 1) // page_size).clamp(min=1, max=mb)
    indptr = out_indptr[: bs + 1]
    indptr[0] = 0
    torch.cumsum(n_blocks.to(torch.int32), dim=0, out=indptr[1:])
    out_last_len[:bs] = seqlens.to(torch.int32) - (n_blocks - 1) * page_size
    # 紧凑 packing：pos[i,j] = indptr[i] + j（无效 j 指向 trash 槽）
    cols = torch.arange(mb, device=dev)
    pos = indptr[:bs, None].to(torch.int64) + cols[None, :]
    valid = cols[None, :] < n_blocks[:, None]
    trash = out_indices.numel() - 1
    pos_safe = torch.where(valid, pos, torch.full_like(pos, trash))
    src = torch.where(valid, bt, torch.zeros_like(bt))
    out_indices.scatter_(0, pos_safe.reshape(-1), src.reshape(-1))
    out_indices[trash] = 0


class Fp8KVManager:
    """FP8 KV cache 的 wrapper 生命周期 / plan / scale 校准。"""

    def __init__(
        self,
        *,
        num_layers: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        max_batch_size: int,
        max_blocks_per_seq: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        use_cuda_graph: bool,
    ) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.num_layers = num_layers
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_batch_size = max_batch_size
        self.dtype = dtype
        self.use_cuda_graph = use_cuda_graph
        self.capturing = False  # 录图中跳过 plan

        self._workspace = torch.empty(
            _WORKSPACE_BYTES, dtype=torch.uint8, device=device
        )
        # decode：非 CG 变体（bs > max_bs 或 eager 时用）
        self._decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self._workspace, kv_layout="NHD"
        )
        # decode：CG 变体按 padded bs 懒创建（batch size 构造时固定）
        self._cg_decode: dict[int, object] = {}
        # prefill：paged 版（有前缀缓存时）；无缓存的首 chunk 仍走 FA2 varlen
        self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self._workspace, kv_layout="NHD"
        )
        # CSR 预分配缓冲（decode 用；prefill 每批独立张量）
        # indices 末尾留 1 个 trash 槽（无效条目散射到此，不被 kernel 读取）
        self._indptr = torch.empty(
            max_batch_size + 1, dtype=torch.int32, device=device
        )
        self._indices = torch.empty(
            max_batch_size * max_blocks_per_seq + 1, dtype=torch.int32, device=device
        )
        self._last_len = torch.empty(
            max_batch_size, dtype=torch.int32, device=device
        )
        # per-layer 固定 scale（0-dim；校准后冻结）。两套表示：
        # - Tensor：store_kvcache_fp8 在 GPU 内核里读；
        # - Python float：FlashInfer wrapper 的 k_scale/v_scale 形参要求标量
        #   （内部 ``sm_scale *= k_scale`` 后按 float 下发 kernel，传 Tensor
        #   会被 TIR 拒绝），且 decode 热路径/录图内不能做 .item() 同步。
        self._k_scale = torch.ones(num_layers, dtype=torch.float32, device=device)
        self._v_scale = torch.ones(num_layers, dtype=torch.float32, device=device)
        self._k_scale_val = [1.0] * num_layers
        self._v_scale_val = [1.0] * num_layers
        self._calibrated = [False] * num_layers
        self._cg_decode_cur = None
        self._kv_cache = None

    # ------------------------------------------------------------------ #
    # scale 校准（首次写入该层时调用一次；之后冻结）
    # ------------------------------------------------------------------ #

    def maybe_calibrate(self, layer_id: int, k: torch.Tensor, v: torch.Tensor) -> None:
        if self._calibrated[layer_id]:
            return
        amax = torch.maximum(k.abs().amax(), v.abs().amax())
        scale = (amax / _FP8_MAX * _SCALE_HEADROOM).clamp(min=_SCALE_FLOOR)
        self._k_scale[layer_id] = scale
        self._v_scale[layer_id] = scale
        # 一次性 D2H：校准只发生在录图前的最小前向里
        val = float(scale.item())
        self._k_scale_val[layer_id] = val
        self._v_scale_val[layer_id] = val
        self._calibrated[layer_id] = True

    def scales(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """GPU Tensor 形式（供 store_kvcache_fp8 内核读取）。"""
        return self._k_scale[layer_id], self._v_scale[layer_id]

    def scales_float(self, layer_id: int) -> tuple[float, float]:
        """Python float 形式（供 FlashInfer wrapper 的 run 使用）。"""
        return self._k_scale_val[layer_id], self._v_scale_val[layer_id]

    # ------------------------------------------------------------------ #
    # plan（录图外调用）
    # ------------------------------------------------------------------ #

    def plan_decode(
        self,
        bs: int,
        block_tables: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cg: bool | None = None,
    ) -> None:
        """plan decode wrapper。

        Args:
            cg: None = 按构造时的 use_cuda_graph；True = 强制 CG wrapper
                （replay 前）；False = 非 CG wrapper（eager decode / 校准）。
        """
        if bs <= 0:
            return
        bt = block_tables[:bs].to(torch.int32)
        cs = cache_seqlens[:bs].to(torch.int32)
        block_tables_to_csr(
            bt, cs, self.page_size, self._indptr, self._indices, self._last_len
        )
        use_cg = self.use_cuda_graph if cg is None else cg
        if use_cg:
            wrapper = self._cg_decode_wrapper(bs)
            # CG wrapper 的 indices 缓冲按 bs 分配，plan 会校验源长度并拷入；
            # 管理器缓冲是 max_bs 全长的，必须按 bs 切片。
            indices = self._indices[: bs * self._max_blocks() + 1]
        else:
            wrapper = self._decode_wrapper
            indices = self._indices
        wrapper.plan(
            self._indptr[: bs + 1],
            indices,
            self._last_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            q_data_type=self.dtype,
            kv_data_type=FP8_KV_DTYPE,
            o_data_type=self.dtype,
        )

    def plan_prefill(
        self,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> None:
        """有前缀缓存的 prefill chunk：kv 长度取 cu_seqlens_k 差分。"""
        bs = cu_seqlens_q.numel() - 1
        if bs <= 0:
            return
        kv_lens = (cu_seqlens_k[1 : bs + 1] - cu_seqlens_k[:bs]).to(torch.int32)
        indptr = torch.empty(bs + 1, dtype=torch.int32, device=kv_lens.device)
        _, mb = block_tables.shape
        indices = torch.empty(bs * mb + 1, dtype=torch.int32, device=kv_lens.device)
        last_len = torch.empty(bs, dtype=torch.int32, device=kv_lens.device)
        block_tables_to_csr(
            block_tables.to(torch.int32), kv_lens, self.page_size,
            indptr, indices, last_len,
        )
        self._prefill_wrapper.plan(
            cu_seqlens_q.to(torch.int32),
            indptr,
            indices,
            last_len,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            causal=True,
            q_data_type=self.dtype,
            kv_data_type=FP8_KV_DTYPE,
            o_data_type=self.dtype,
        )

    # ------------------------------------------------------------------ #
    # run（attention custom op 内调用；plan 已完成）
    # ------------------------------------------------------------------ #

    def run_decode(self, q: torch.Tensor, layer_id: int) -> torch.Tensor:
        """q: [bs, num_qo_heads, head_dim] 激活 dtype → 同形状输出。"""
        k_scale, v_scale = self.scales_float(layer_id)
        return self._decode_wrapper.run(
            q, self._kv_slice(layer_id), k_scale=k_scale, v_scale=v_scale
        )

    def run_decode_cg(self, q: torch.Tensor, layer_id: int) -> torch.Tensor:
        """CG replay 中的 decode：用最近一次 plan 的 CG wrapper。"""
        assert self._cg_decode_cur is not None, "plan_decode() must run before replay"
        k_scale, v_scale = self.scales_float(layer_id)
        return self._cg_decode_cur.run(
            q, self._kv_slice(layer_id), k_scale=k_scale, v_scale=v_scale
        )

    def run_prefill(self, q: torch.Tensor, layer_id: int) -> torch.Tensor:
        """q: [total_tokens, num_qo_heads, head_dim]（有前缀缓存）。"""
        k_scale, v_scale = self.scales_float(layer_id)
        return self._prefill_wrapper.run(
            q, self._kv_slice(layer_id), k_scale=k_scale, v_scale=v_scale
        )

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #

    def set_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """整个 KV 池（ModelRunner 持有引用；每层切 [0/1, layer_id]）。"""
        self._kv_cache = kv_cache

    def _kv_slice(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        pool = self._kv_cache
        # [2, L, blocks, page, nkv, hd] → 两个 4-D [blocks, page, nkv, hd]
        return pool[0, layer_id], pool[1, layer_id]

    def _cg_decode_wrapper(self, bs: int):
        """按 bs 懒创建 CUDAGraph decode wrapper（batch size 构造时固定）。"""
        wrapper = self._cg_decode.get(bs)
        if wrapper is None:
            from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

            wrapper = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
                self._workspace,
                torch.empty(bs + 1, dtype=torch.int32, device=self._workspace.device),
                torch.empty(
                    bs * self._max_blocks() + 1, dtype=torch.int32,
                    device=self._workspace.device,
                ),
                torch.empty(bs, dtype=torch.int32, device=self._workspace.device),
                kv_layout="NHD",
            )
            self._cg_decode[bs] = wrapper
        self._cg_decode_cur = wrapper
        return wrapper

    def _max_blocks(self) -> int:
        return self._indices.numel() // max(self.max_batch_size, 1)

    def destroy(self) -> None:
        self._cg_decode.clear()
        self._decode_wrapper = None
        self._prefill_wrapper = None
        self._workspace = None
