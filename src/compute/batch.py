from __future__ import annotations

import numpy as np
import torch

from ..data.batch import SeqSchedule
from ..data.sequence import slots_for_range


def _pinned(
    shape: int | tuple[int, ...],
    dtype: torch.dtype,
    fill: int | float | None = None,
) -> torch.Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    if fill is None:
        return torch.empty(shape, dtype=dtype, pin_memory=True)
    return torch.full(shape, fill, dtype=dtype, pin_memory=True)


def _h2d(dst: torch.Tensor, src: torch.Tensor, n: int) -> None:
    dst[:n].copy_(src[:n], non_blocking=True)


def _gpu(
    shape: int | tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    fill: int | float = 0,
) -> torch.Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    return torch.full(shape, fill, dtype=dtype, device=device)


class Batch:
    def __init__(
        self,
        *,
        max_bs: int,
        max_tokens: int,
        max_blocks: int,
        vocab_size: int,
        device: torch.device,
        block_size: int,
    ) -> None:
        self.max_bs = max_bs
        self.max_tokens = max_tokens
        self.max_blocks = max_blocks
        self.vocab_size = vocab_size
        self.device = device
        self.block_size = block_size

        self.input_ids = _gpu(max_tokens, torch.int64, device)
        self.positions = _gpu(max_tokens, torch.int64, device)
        self.slot_mapping = _gpu(max_tokens, torch.int32, device, fill=-1)
        self.cache_seqlens = _gpu(max_bs, torch.int32, device)
        self.cu_seqlens_q = _gpu(max_bs + 1, torch.int32, device)
        self.cu_seqlens_k = _gpu(max_bs + 1, torch.int32, device)
        self.block_tables = _gpu((max_bs, max_blocks), torch.int32, device, fill=-1)

        self.temperatures = _gpu(max_bs, torch.float32, device)
        self.top_ps = _gpu(max_bs, torch.float32, device)
        self.top_ks = _gpu(max_bs, torch.int32, device)
        self.min_ps = _gpu(max_bs, torch.float32, device)

        self.all_greedy = True
        self.any_greedy = False
        self.use_top_k = False
        self.use_top_p = False
        self.use_min_p = False
        self.max_top_k = 0

        self._dec_ids = _pinned(max_bs, torch.int64)
        self._dec_pos = _pinned(max_bs, torch.int64)
        self._dec_slot = _pinned(max_bs, torch.int32, fill=-1)
        self._dec_csl = _pinned(max_bs, torch.int32)

        self._pre_ids = _pinned(max_tokens, torch.int64)
        self._pre_pos = _pinned(max_tokens, torch.int64)
        self._pre_slot = _pinned(max_tokens, torch.int32, fill=-1)
        self._pre_cuq = _pinned(max_bs + 1, torch.int32)
        self._pre_cuk = _pinned(max_bs + 1, torch.int32)

        self._h_temp = _pinned(max_bs, torch.float32)
        self._h_top_p = _pinned(max_bs, torch.float32)
        self._h_top_k = _pinned(max_bs, torch.int32)
        self._h_min_p = _pinned(max_bs, torch.float32)

        self._bt_pin = _pinned((max_bs, max_blocks), torch.int32, fill=-1)
        self._bt_cpu = self._bt_pin.numpy()
        self._row_fp: list[tuple[int, int, int]] = [(-1, -1, -1)] * max_bs
        self._gpu_bt_ids: tuple[int, ...] = ()
        self._samp_key: tuple | None = None

        self._dec_ids_np = self._dec_ids.numpy()
        self._dec_pos_np = self._dec_pos.numpy()
        self._dec_slot_np = self._dec_slot.numpy()
        self._dec_csl_np = self._dec_csl.numpy()
        self._pre_ids_np = self._pre_ids.numpy()
        self._pre_pos_np = self._pre_pos.numpy()
        self._pre_slot_np = self._pre_slot.numpy()
        self._pre_cuq_np = self._pre_cuq.numpy()
        self._pre_cuk_np = self._pre_cuk.numpy()
        self._h_temp_np = self._h_temp.numpy()
        self._h_top_p_np = self._h_top_p.numpy()
        self._h_top_k_np = self._h_top_k.numpy()
        self._h_min_p_np = self._h_min_p.numpy()

    def build_decode(
        self, seqs: list[SeqSchedule], *, pad_to: int | None = None
    ) -> None:
        bs = len(seqs)
        ids_np, pos_np = self._dec_ids_np, self._dec_pos_np
        slot_np, csl_np = self._dec_slot_np, self._dec_csl_np
        bsz = self.block_size
        for i, s in enumerate(seqs):
            pos = s.num_cached_tokens
            bt = s.block_table
            ids_np[i] = s.token_ids[-1]
            pos_np[i] = pos
            slot_np[i] = bt[pos // bsz] * bsz + pos % bsz
            csl_np[i] = pos + 1
        _h2d(self.input_ids, self._dec_ids, bs)
        _h2d(self.positions, self._dec_pos, bs)
        _h2d(self.slot_mapping, self._dec_slot, bs)
        _h2d(self.cache_seqlens, self._dec_csl, bs)

        sid = tuple(s.seq_id for s in seqs)
        self._sync_block_tables(seqs, sid)
        self._sync_sampling(seqs)
        if pad_to is not None and pad_to > bs:
            self.slot_mapping[bs:pad_to].fill_(-1)
            self.cache_seqlens[bs:pad_to].zero_()
            self.block_tables[bs:pad_to].fill_(-1)

    def build_prefill(self, seqs: list[SeqSchedule]) -> tuple[int, int, int, bool]:
        ids = self._pre_ids_np
        positions = self._pre_pos_np
        slots = self._pre_slot_np
        cu_q = self._pre_cuq_np
        cu_k = self._pre_cuk_np
        cu_q[0] = 0
        cu_k[0] = 0
        max_q = 0
        max_k = 0
        n_tokens = 0
        for i, s in enumerate(seqs):
            start = s.num_cached_tokens
            n = len(s.token_ids)
            end = start + n
            ids[n_tokens : n_tokens + n] = s.token_ids
            positions[n_tokens : n_tokens + n] = np.arange(start, end)
            slots[n_tokens : n_tokens + n] = slots_for_range(
                s.block_table, start, end, self.block_size
            )
            cu_q[i + 1] = cu_q[i] + n
            cu_k[i + 1] = cu_k[i] + end
            max_q = max(max_q, n)
            max_k = max(max_k, end)
            n_tokens += n

        bs = len(seqs)
        _h2d(self.input_ids, self._pre_ids, n_tokens)
        _h2d(self.positions, self._pre_pos, n_tokens)
        _h2d(self.slot_mapping, self._pre_slot, n_tokens)
        n_cu = bs + 1
        _h2d(self.cu_seqlens_q, self._pre_cuq, n_cu)
        _h2d(self.cu_seqlens_k, self._pre_cuk, n_cu)

        has_cache = int(cu_k[bs]) > int(cu_q[bs])
        sid = tuple(s.seq_id for s in seqs)
        if has_cache:
            self._sync_block_tables(seqs, sid)
        else:
            self._gpu_bt_ids = ()
        self._sync_sampling(seqs)
        return n_tokens, max_q, max_k, has_cache

    def _sync_block_tables(
        self, seqs: list[SeqSchedule], sid: tuple[int, ...]
    ) -> None:
        bs = len(seqs)
        same = sid == self._gpu_bt_ids
        dirty: list[int] = []
        fp = self._row_fp
        cpu = self._bt_cpu
        for i, s in enumerate(seqs):
            bt = s.block_table
            k = len(bt)
            first, last, prev_k = fp[i]
            if not same:
                prev_k = -1
            elif prev_k == k and (k == 0 or (bt[0] == first and bt[-1] == last)):
                continue
            if (
                prev_k == k - 1
                and k > 0
                and (k == 1 or (first == bt[0] and last == bt[k - 2]))
            ):
                cpu[i, k - 1] = bt[-1]
            else:
                cpu[i, :k] = bt
                cpu[i, k:] = -1
            fp[i] = (bt[0] if k else -1, bt[-1] if k else -1, k)
            dirty.append(i)
        self._gpu_bt_ids = sid
        if not dirty:
            return
        if len(dirty) > 8 or not same:
            self.block_tables[:bs].copy_(self._bt_pin[:bs], non_blocking=True)
            return
        for i in dirty:
            self.block_tables[i].copy_(self._bt_pin[i], non_blocking=True)

    def _sync_sampling(self, seqs: list[SeqSchedule]) -> None:
        key = tuple(
            (s.seq_id, s.temperature, s.top_p, s.top_k, s.min_p) for s in seqs
        )
        if key == self._samp_key:
            return
        self._samp_key = key
        bs = len(seqs)
        vocab = self.vocab_size
        temps = self._h_temp_np
        top_ps = self._h_top_p_np
        top_ks = self._h_top_k_np
        min_ps = self._h_min_p_np
        for i, s in enumerate(seqs):
            temps[i] = s.temperature
            top_ps[i] = 1.0 if s.top_p is None else s.top_p
            top_ks[i] = (
                vocab if (s.top_k is None or s.top_k <= 0) else min(s.top_k, vocab)
            )
            min_ps[i] = 0.0 if s.min_p is None else s.min_p
        _h2d(self.temperatures, self._h_temp, bs)
        _h2d(self.top_ps, self._h_top_p, bs)
        _h2d(self.top_ks, self._h_top_k, bs)
        _h2d(self.min_ps, self._h_min_p, bs)

        t = temps[:bs]
        k = top_ks[:bs]
        p = top_ps[:bs]
        m = min_ps[:bs]
        self.all_greedy = bool(np.all(t <= 1e-5))
        self.any_greedy = bool(np.any(t <= 1e-5))
        self.use_top_k = bool(np.any(k < vocab))
        self.use_top_p = bool(np.any(p < 1.0))
        self.use_min_p = bool(np.any(m > 0.0))
        self.max_top_k = int(k.max()) if self.use_top_k else 0
