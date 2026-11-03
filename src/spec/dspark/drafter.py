"""DSpark 草稿器：把官方 ``draft_ops.forward_dspark_draft_block`` 的协议搬到本引擎。

核心只有一件事——**草稿的 KV 缓存怎么滚**（规则从上游 `base_evaluator.py` 读出，
见 docs/dspark-spec.md §6.3）：

```
propose(start, anchor, pending_ctx):
    past 里已有 (start - seg) 条 context
    → 追加 seg（上一步验证通过的 context 特征）+ block 个候选
    → crop(start)            # 只留 context，块内的候选 KV 用完即弃
update(ctx, start_new):
    pending_ctx = ctx        # 下个 propose 时追加
```

为什么用 `crop` 而不是"先把 block 写进临时缓存"：上游就是这么做的，块内那 block 个
位置的 KV 对后续没有任何用处（每步的候选都由 target 的 hidden + anchor 重新算），
所以直接复用同一块 KV 再裁掉，比维护两套缓存简单且省显存。

线程/序列安全：状态按 ``seq_id`` 隔离。**账本不一致时（preempt 后重建、或有人改了
num_cached_tokens）宁可本步不起草**，也不要用一份过期 context 去猜——错误草稿会被
验证拒掉，但浪费的是每步一次完整的块前向。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers.cache_utils import DynamicCache

from .modeling import Qwen3DSparkModel
from .padded_cache import PaddedKVCache
from .paged_cache import PagedDraftKV


@dataclass
class _DraftState:
    """一条序列的草稿侧状态。"""

    #: 草稿模型的 KV：只存"已提交 token 的 context"，长度恒等于已提交长度
    past: DynamicCache = field(default_factory=DynamicCache)
    #: 下一步要追加的 context 段（上一步验证通过的 accepted+1 个位置的 hidden）
    pending_ctx: torch.Tensor | None = None
    #: 上一步记录的已提交长度（用于检测账本漂移）
    start: int = 0


class DSparkDrafter:
    """在 worker 侧运行：持有草稿模型 + 每序列的草稿 KV。"""

    def __init__(
        self,
        model: Qwen3DSparkModel,
        *,
        device: torch.device,
        temperature: float = 0.0,
        max_proposal: int = 0,
        max_ctx_len: int = 4096,
        max_rows: int = 8,
        use_paged: bool = False,
        page_size: int = 16,
        num_pages: int = 0,
        use_paged_fa: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        # 块宽由 checkpoint 决定（官方 8B 是 7）：一次前向固定产出这么多位置
        self.block_size = int(model.block_size)
        # 实际拿几个去验证。**必须 ≤ 调度器预留的 k**，否则 verify 会写到没
        # 分配块的位置上（越界写 KV）。0 表示不截断（= block_size）。
        self.max_proposal = min(
            self.block_size, int(max_proposal) if max_proposal > 0 else self.block_size
        )
        self.mask_token_id = int(model.mask_token_id)
        self.temperature = float(temperature)
        self._state: dict[int, _DraftState] = {}
        # 批量化：seq_id → 行号（长期持有），以及共享的 padded KV
        self._row_of: dict[int, int] = {}
        self._padded: PaddedKVCache | None = None
        #: 已把 `DynamicCache` 的 prompt K/V 搬进 padded 行的序列（ZOS 切换用）
        self._padded_seeded: set[int] = set()
        self.max_ctx_len = int(max_ctx_len)
        self.max_rows = int(max_rows)
        #: 页式（读路径一）：容量 O(实际 token)，稠密是 O(行数 × max_len)
        self.use_paged = bool(use_paged)
        self.page_size = int(page_size)
        self.num_pages = int(num_pages)
        #: 读路径二（FA paged）；未开页式时无意义
        self.use_paged_fa = bool(use_paged_fa) and self.use_paged
        self._paged: PagedDraftKV | None = None

    # ---------------------------------------------------------------- #
    # 状态管理
    # ---------------------------------------------------------------- #

    def reset(self, seq_id: int) -> None:
        self._state.pop(seq_id, None)
        self._padded_seeded.discard(seq_id)

    def clear(self) -> None:
        self._state.clear()
        self._padded_seeded.clear()

    def ctx_len(self, seq_id: int) -> int:
        st = self._state.get(seq_id)
        return 0 if st is None else st.past.get_seq_length()

    def update(self, seq_id: int, ctx: torch.Tensor, *, start: int) -> None:
        """记录本步验证通过的 context 段（``accepted+1`` 个位置的 hidden）。"""
        st = self._state.get(seq_id)
        if st is None:
            return
        st.pending_ctx = ctx
        st.start = start

    # ---------------------------------------------------------------- #
    # 起草
    # ---------------------------------------------------------------- #

    @torch.inference_mode()
    def propose(
        self,
        seq_id: int,
        *,
        anchor_token_id: int,
        start: int,
        first_ctx: torch.Tensor | None = None,
    ) -> tuple[list[int], torch.Tensor | None]:
        """产出至多 ``block_size`` 个候选。

        Args:
            seq_id: 序列 id（状态隔离用）。
            anchor_token_id: 位置 ``start`` 上的 token（块的第 0 个位置喂它）。
            start: 已提交长度（= 引擎的 ``num_cached_tokens``）。
            first_ctx: 仅首步需要——整段 prompt 的 context 特征
                ``[1, prompt_len, len(target_layer_ids)*hidden]``。

        Returns:
            ``(tokens, draft_probs)``。``draft_probs`` 是 ``[k, vocab]`` 的归一化
            分布（温度采样时给拒绝采样用）；greedy 下返回 ``None``（等价于
            one-hot，见 `src/spec/verify.py::accept_rejection`）。

        Returns ``([], None)`` 的情形：账本漂移、缺 context、或 ``start`` 超界。
        调用方应退化为普通 decode，而不是报错。
        """
        if self.use_paged:
            return self._propose_paged(
                seq_id,
                anchor_token_id=anchor_token_id,
                start=start,
                first_ctx=first_ctx,
            )

        st = self._state.get(seq_id)
        if st is None:
            if first_ctx is None or start <= 0:
                return [], None
            st = _DraftState(past=DynamicCache(), pending_ctx=first_ctx, start=start)
            self._state[seq_id] = st
        elif st.start != start:
            # 账本漂移（preempt 重建、或外部改了 cached 长度）：丢弃状态，
            # 本步不起草。重建需要全量 aux hidden，v1 不做。
            self._state.pop(seq_id, None)
            return [], None

        if st.pending_ctx is None:
            return [], None
        ctx = st.pending_ctx
        if ctx.dim() == 2:  # 容忍 [n, F]：引擎侧传的是二维切片
            ctx = ctx.unsqueeze(0)

        block = self.block_size
        cached = st.past.get_seq_length()
        seg = start - cached
        if seg <= 0:
            # 没有新的 context 可追加：pending_ctx 与缓存不一致，保守放弃
            self._state.pop(seq_id, None)
            return [], None

        ids = torch.full(
            (1, block), self.mask_token_id, dtype=torch.long, device=self.device
        )
        ids[0, 0] = int(anchor_token_id)
        # 位置区间 [cached, start + block)：seg 个 context + block 个候选
        positions = torch.arange(
            cached, start + block, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        hidden = self.model._forward_backbone(
            position_ids=positions,
            noise_embedding=self.model.embed_tokens(ids),
            target_hidden_states=ctx,
            attention_mask=None,
            past_key_values=st.past,
            use_cache=True,
            is_causal=False,
        )
        # 只留 context：块内 block 个位置的 KV 用完即弃（上游同款）
        st.past.crop(start)
        st.pending_ctx = None

        block_hidden = hidden[:, :block, :]
        base_logits = self.model.compute_logits(block_hidden)
        tokens, corrected = self.model.sample_draft_tokens(
            base_logits,
            first_prev_token_ids=torch.tensor(
                [int(anchor_token_id)], dtype=torch.long, device=self.device
            ),
            temperature=self.temperature,
            hidden_states=block_hidden,
        )

        probs = None
        if self.temperature > 1e-5:
            # 温度采样才需要真实草稿分布；greedy 下 q 是 one-hot，验证侧按
            # draft_probs=None 处理即可（省一次 [k, vocab] 的 softmax）
            probs = torch.softmax(corrected.float() / self.temperature, dim=-1)

        # 截断到调度器预留的 k：块宽由 checkpoint 固定（7），但引擎可能只想验证
        # 前 k 个。多给会让 verify 写到未分配块的位置上。
        keep = self.max_proposal
        if keep < block:
            tokens = tokens[:, :keep]
            if probs is not None:
                probs = probs[:, :keep]

        return tokens[0].tolist(), (None if probs is None else probs[0])

    @torch.inference_mode()
    def _propose_paged(
        self,
        seq_id: int,
        *,
        anchor_token_id: int,
        start: int,
        first_ctx: torch.Tensor | None = None,
    ) -> tuple[list[int], torch.Tensor | None]:
        """单序列起草的页式版本（``use_paged=True`` 时走这里）。

        与稠密版的差别只在 KV 容器：长度取 `cache.row_len(row)`（页表维护），
        其余（位置区间 `[cached, start+block)`、只留 context 的 crop）完全一致。
        """
        cache = self._ensure_paged()
        row = self._row(seq_id)
        cached = cache.row_len(row)

        st = self._state.get(seq_id)
        if st is None:
            if first_ctx is None or start <= 0:
                return [], None
            st = _DraftState(past=DynamicCache(), pending_ctx=first_ctx, start=start)
            self._state[seq_id] = st
        elif st.start != start:
            # 账本漂移：丢弃状态与本行的页，本步不起草
            self._state.pop(seq_id, None)
            cache.reset_row(row)
            return [], None

        if st.pending_ctx is None:
            return [], None
        ctx = st.pending_ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0)
        seg = int(ctx.shape[-2])
        if cached + seg != start:
            self.reset_seq(seq_id)
            return [], None

        block = self.block_size
        ids = torch.full(
            (1, block), self.mask_token_id, dtype=torch.long, device=self.device
        )
        ids[0, 0] = int(anchor_token_id)
        positions = torch.arange(
            cached, start + block, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        cache.begin_step([row], [seg], seg_max=seg, block=block)
        if len(cache.active_rows) != 1:
            # 页池不足：本步不起草（只损收益，不影响正确性）
            return [], None
        # 读路径二不需要 additive mask（FA 按 cache_seqlens 处理变长）
        mask = (
            None
            if getattr(cache, "is_paged_fa", False)
            else cache.mask(q_len=block)
        )
        hidden = self.model._forward_backbone(
            position_ids=positions,
            noise_embedding=self.model.embed_tokens(ids),
            target_hidden_states=ctx,
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
            is_causal=False,
        )
        cache.crop()
        st.pending_ctx = None

        block_hidden = hidden[:, :block, :]
        base_logits = self.model.compute_logits(block_hidden)
        tokens, corrected = self.model.sample_draft_tokens(
            base_logits,
            first_prev_token_ids=torch.tensor(
                [int(anchor_token_id)], dtype=torch.long, device=self.device
            ),
            temperature=self.temperature,
            hidden_states=block_hidden,
        )
        probs = None
        if self.temperature > 1e-5:
            probs = torch.softmax(corrected.float() / self.temperature, dim=-1)
        keep = self.max_proposal
        if keep < block:
            tokens = tokens[:, :keep]
            if probs is not None:
                probs = probs[:, :keep]
        return tokens[0].tolist(), (None if probs is None else probs[0])

    # ---------------------------------------------------------------- #
    # 批量起草（B 条序列一次前向，权重只读一遍）
    # ---------------------------------------------------------------- #

    def _row(self, seq_id: int) -> int:
        row = self._row_of.get(seq_id)
        if row is not None:
            return row
        used = set(self._row_of.values())
        free = [i for i in range(self.max_rows) if i not in used]
        if not free:
            # 行池满：回收最老的序列。被回收的序列之后拿不到 first_ctx，会退化成
            # "本步不起草"——只损失收益，不影响正确性（不会越界、不会算错）。
            victim = next(iter(self._row_of))
            self.reset_seq(victim)
            used = set(self._row_of.values())
            free = [i for i in range(self.max_rows) if i not in used]
        row = free[0]
        self._row_of[seq_id] = row
        return row

    def _ensure_paged(self) -> PagedDraftKV:
        """页池容量：`num_pages=0` 时按 `max(64, max_rows × 16)` 推导。

        这个推导值决定了「总共能装多少 token」= `num_pages × page_size`；
        默认 16 页/行 × 16 token/页 = 每行 256 token 的预算，超出时该序列本步不起草。
        显存约为 `num_pages × page_size × heads × dim × 2 × layers × 2B`。
        """
        if self._paged is None:
            cfg = self.model.config
            pages = self.num_pages or max(64, self.max_rows * 16)
            self._paged = PagedDraftKV(
                num_layers=int(cfg.num_hidden_layers),
                num_kv_heads=int(cfg.num_key_value_heads),
                head_dim=int(
                    getattr(cfg, "head_dim", None)
                    or cfg.hidden_size // cfg.num_attention_heads
                ),
                num_rows=max(1, self.max_rows),
                num_pages=int(pages),
                page_size=self.page_size,
                device=self.device,
                dtype=next(self.model.parameters()).dtype,
                use_fa=self.use_paged_fa,
            )
        return self._paged

    def _ensure_cache(self):
        """当前生效的草稿 KV（页式或稠密）。两者接口同签名，调用点无需分支。"""
        return self._ensure_paged() if self.use_paged else self._ensure_padded()

    def _ensure_padded(self) -> PaddedKVCache:
        if self._padded is None:
            cfg = self.model.config
            self._padded = PaddedKVCache(
                num_layers=int(cfg.num_hidden_layers),
                num_kv_heads=int(cfg.num_key_value_heads),
                head_dim=int(getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads),
                num_rows=max(1, self.max_rows),
                max_len=self.max_ctx_len,
                device=self.device,
                dtype=next(self.model.parameters()).dtype,
            )
        return self._padded

    @torch.inference_mode()
    def propose_batch_ids(
        self,
        seq_ids: list[int],
        anchors: torch.Tensor,
        starts: torch.Tensor,
        ctxs: list[torch.Tensor],
        segs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """批量化起草的**设备侧入参**版本（ZOS 用）。

        与 ``propose_batch`` 的区别只有入参来源：``anchors`` / ``starts`` 是 ``[B]``
        的设备张量（上一拍的采样结果与推进后的起始位置），CPU 并不知道它们的值。
        位置编码因此必须用 broadcast 现算，不能再用 Python int。

        为什么不做行长度对齐（``seed_row``）：ZOS 下每拍都走 ``crop``，每行的有效长度
        是**增量维护**的；只有序列首次进入流水线时才需要 seed，而那一拍是同步提交的
        （CPU 状态准确），走 ``propose_batch`` 的 int 路径即可。

        Returns:
            ``(tokens [B, block] int64, probs [B, block, vocab] | None)``
        """
        b = len(seq_ids)
        assert anchors.numel() == b and starts.numel() == b
        if segs is not None:
            # ZOS：ctxs 已经是 pad 到 m 的整块，真实长度由设备 seg 给出
            device_segs, seg_list = segs, None
            seg_max = int(ctxs[0].shape[-2])
        else:
            device_segs = None
            seg_list = [int(c.shape[-2]) if c.dim() == 3 else int(c.shape[0]) for c in ctxs]
            seg_max = max(seg_list) if seg_list else 0
        block = self.block_size
        dtype = next(self.model.parameters()).dtype
        feat = int(ctxs[0].shape[-1])
        cache = self._ensure_padded()
        rows = [self._row(sid) for sid in seq_ids]

        ctx_pad = torch.zeros(b, seg_max, feat, dtype=dtype, device=self.device)
        for i, c in enumerate(ctxs):
            c3 = c if c.dim() == 3 else c.unsqueeze(0)
            take = seg_max if seg_list is None else seg_list[i]
            ctx_pad[i, :take] = c3[0][:take].to(dtype)

        ids = torch.full(
            (b, block), self.mask_token_id, dtype=torch.long, device=self.device
        )
        ids[:, 0] = anchors.to(torch.long)

        # 位置必须按**真实 token** 排：ctx 段 [start-seg, start) + 块 [start, start+block)。
        # 用 padded 的 kv 偏移算会让短行的块位置整体后移 seg_max-seg。
        start_col = starts.to(torch.long).view(b, 1)
        if device_segs is None:
            seg_col = torch.tensor(
                seg_list, dtype=torch.long, device=self.device
            ).view(b, 1)
        else:
            seg_col = device_segs.to(torch.long).view(b, 1)
        ctx_pos = (start_col - seg_col) + torch.arange(
            seg_max, device=self.device
        ).view(1, -1)
        blk_pos = start_col + torch.arange(block, device=self.device).view(1, -1)
        pos_rows = torch.cat([ctx_pos, blk_pos], dim=1)

        cache.begin_step(
            rows, seg_list, seg_max, block, device_segs=device_segs
        )
        mask = cache.mask(q_len=block, dtype=dtype)
        hidden = self.model._forward_backbone(
            position_ids=pos_rows,
            noise_embedding=self.model.embed_tokens(ids),
            target_hidden_states=ctx_pad,
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
            is_causal=False,
        )
        cache.crop()

        block_hidden = hidden[:, :block, :]
        base_logits = self.model.compute_logits(block_hidden)
        tokens, corrected = self.model.sample_draft_tokens(
            base_logits,
            first_prev_token_ids=anchors.to(torch.long),
            temperature=self.temperature,
            hidden_states=block_hidden,
        )
        probs = None
        if self.temperature > 1e-5:
            probs = torch.softmax(corrected.float() / self.temperature, dim=-1)
        return tokens, probs

    def update_full(
        self,
        seq_id: int,
        ctx_block: torch.Tensor,
        seg: torch.Tensor,
        start: int = -1,
    ) -> None:
        """ZOS：存**整块** ctx（`[1, m, F]`）与设备侧 seg（`1+j`），不做切片。

        普通路径的 `update` 用 host 的 `accepted` 切片（`aux[:accepted+1]`），ZOS 下
        这个值只在设备上，所以改成"存满 + 传设备 seg"，由 `propose_batch_ids` 用
        `device_segs` 把多余位置 mask 掉。
        """
        st = self._state.get(seq_id)
        if st is None:
            st = _DraftState(past=DynamicCache(), pending_ctx=None, start=0)
            self._state[seq_id] = st
        st.pending_ctx = ctx_block
        st.pending_seg = seg
        if start >= 0:
            st.start = start

    def take_pending(self, seq_id: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """取走该序列待用的 (ctx 块, 设备 seg)，并清空。"""
        st = self._state.get(seq_id)
        if st is None:
            return None, None
        ctx, seg = st.pending_ctx, st.pending_seg
        st.pending_ctx, st.pending_seg = None, None
        return ctx, seg

    def seed_padded_from_past(self, seq_id: int) -> None:
        """ZOS：把首拍 `propose()` 的 prompt K/V 初始化到 padded 行（只做一次）。"""
        if seq_id in self._padded_seeded:
            return
        st = self._state.get(seq_id)
        if st is None:
            return
        row = self._row(seq_id)
        if self._ensure_cache().seed_from_dynamic(row, st.past) > 0:
            self._padded_seeded.add(seq_id)

    def reset_seq(self, seq_id: int) -> None:
        """释放一条序列的草稿状态与 KV 行。"""
        self.reset(seq_id)
        row = self._row_of.pop(seq_id, None)
        if row is not None:
            # 两种容器都可能在生命周期内被创建过，都要回收
            if self._padded is not None:
                self._padded.reset_row(row)
            if self._paged is not None:
                self._paged.reset_row(row)

    @torch.inference_mode()
    def propose_batch(
        self,
        items: list[tuple[int, int, int, torch.Tensor | None]],
    ) -> list[tuple[list[int], torch.Tensor | None]]:
        """一次前向为多条序列起草。``items`` = ``(seq_id, anchor, start, first_ctx)``。

        只有"有 pending_ctx 且账本一致"的序列参与批量；其余返回空（调用方退化为
        普通 decode）。参与者不足 2 条时直接走单序列 ``propose``，避免为 B=1 付
        掩码/填充的复杂度。
        """
        results: list[tuple[list[int], torch.Tensor | None]] = [([], None)] * len(items)

        # 1) 结算状态：谁能参与
        parts: list[int] = []          # items 下标
        segs: list[int] = []
        for i, (seq_id, _anchor, start, first_ctx) in enumerate(items):
            st = self._state.get(seq_id)
            if st is None:
                if first_ctx is None or start <= 0:
                    continue
                st = _DraftState(past=DynamicCache(), pending_ctx=first_ctx, start=start)
                self._state[seq_id] = st
            elif st.start != start:
                self.reset_seq(seq_id)
                continue
            ctx = st.pending_ctx
            if ctx is None:
                continue
            parts.append(i)
            segs.append(int(ctx.shape[-2]) if ctx.dim() == 3 else int(ctx.shape[0]))

        if len(parts) < 2:
            # 退化：逐序列单跑（复用已验证的路径）
            out: list[tuple[list[int], torch.Tensor | None]] = []
            for seq_id, anchor, start, first_ctx in items:
                out.append(
                    self.propose(
                        seq_id, anchor_token_id=anchor, start=start, first_ctx=first_ctx
                    )
                )
            return out

        # 2) 组批：ctx pad 到 seg_max，位置按各自 start 排
        cache = self._ensure_cache()
        rows = [self._row(items[i][0]) for i in parts]
        seg_max = max(segs)
        block = self.block_size
        feat = int(self._state[items[parts[0]][0]].pending_ctx.shape[-1])
        ctx_pad = torch.zeros(
            len(parts), seg_max, feat,
            dtype=next(self.model.parameters()).dtype, device=self.device,
        )
        ids = torch.full(
            (len(parts), block), self.mask_token_id, dtype=torch.long, device=self.device
        )
        anchors = []
        pos_rows = []
        for k, i in enumerate(parts):
            seq_id, anchor, start, _ = items[i]
            st = self._state[seq_id]
            ctx = st.pending_ctx
            ctx = ctx if ctx.dim() == 3 else ctx.unsqueeze(0)
            ctx_pad[k, : segs[k]] = ctx[0]
            ids[k, 0] = int(anchor)
            anchors.append(int(anchor))
            cached = int(start) - segs[k]
            # 位置必须按"真实 token"排：ctx 段占 [cached, start)，块占 [start, start+block)。
            # **不能用 padded 的 kv 偏移**——那会把短行的块位置整体后移 seg_max-seg。
            row_pos = torch.zeros(
                seg_max + block, dtype=torch.long, device=self.device
            )
            row_pos[: segs[k]] = torch.arange(
                cached, cached + segs[k], device=self.device
            )
            if seg_max > segs[k]:
                row_pos[segs[k] : seg_max] = cached  # pad 被 mask，值无所谓
            row_pos[seg_max:] = torch.arange(start, start + block, device=self.device)
            pos_rows.append(row_pos)

        for k, r in enumerate(rows):
            # 行已有的 context 长度必须等于 start - seg；不一致说明账本漂移
            want = int(items[parts[k]][2]) - segs[k]
            if cache.row_len(r) != want:
                cache.seed_row(r, want)

        kv_len = cache.begin_step(rows, segs, seg_max, block)
        if self.use_paged and len(cache.active_rows) < len(rows):
            # 页池不足时 begin_step 会剔除部分行；行号一旦错位就会串数据，
            # 所以整批退回逐序列（每条各自降级为"不起草"）。
            return [
                self.propose(sid, anchor_token_id=a, start=st, first_ctx=fc)
                for sid, a, st, fc in items
            ]
        # q 只有 block 个（块内非因果），kv 才是 seg_max+block —— 掩码按 q 长度给。
        # 读路径二不需要它（FA 用 cache_seqlens 处理变长）。
        mask = (
            None
            if getattr(cache, "is_paged_fa", False)
            else cache.mask(q_len=block)
        )
        hidden = self.model._forward_backbone(
            position_ids=torch.stack(pos_rows),
            noise_embedding=self.model.embed_tokens(ids),
            target_hidden_states=ctx_pad,
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
            is_causal=False,
        )
        cache.crop()
        for i in parts:
            self._state[items[i][0]].pending_ctx = None

        block_hidden = hidden[:, :block, :]
        base_logits = self.model.compute_logits(block_hidden)
        tokens, corrected = self.model.sample_draft_tokens(
            base_logits,
            first_prev_token_ids=torch.tensor(anchors, dtype=torch.long, device=self.device),
            temperature=self.temperature,
            hidden_states=block_hidden,
        )
        probs = None
        if self.temperature > 1e-5:
            probs = torch.softmax(corrected.float() / self.temperature, dim=-1)
        keep = self.max_proposal
        for k, i in enumerate(parts):
            tk = tokens[k].tolist()[:keep]
            pk = None if probs is None else probs[k][:keep]
            results[i] = (tk, pk)
        return results
