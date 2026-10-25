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

    # ---------------------------------------------------------------- #
    # 状态管理
    # ---------------------------------------------------------------- #

    def reset(self, seq_id: int) -> None:
        self._state.pop(seq_id, None)

    def clear(self) -> None:
        self._state.clear()

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
