"""DSpark 草稿器的 KV 滚账协议（纯 CPU、随机权重）。

这个文件只测一件事：**草稿 KV 的长度必须恒等于"已提交长度"**。

- propose 时先追加「上一步的 context 段 + block 个候选」，再 `crop(start)`；
- 所以调用后 `cache_len == start`，块内候选的 KV 不残留；
- 下一步 `update(ctx)` 只登记待追加段，真正的追加减在下次 propose 里。

写错这层的症状很隐蔽：草稿 token 看着正常、验证也能过，但接受率会莫名偏低
（context 多一条/少一条 → 草稿看到的历史是错的）。所以用整型断言钉死长度。
"""

from __future__ import annotations

import pytest
import torch

from src.spec.dspark.drafter import DSparkDrafter
from src.spec.dspark.loader import check_transformers_symbols
from src.spec.dspark.modeling import Qwen3DSparkModel

HIDDEN = 64
VOCAB = 128
BLOCK = 3
SEG0 = 5  # 首步的 prompt 长度
SEG1 = 2  # 第二步验证通过的 accepted+1


def _tiny_config():
    from transformers import Qwen3Config

    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
    )
    cfg.architectures = ["Qwen3DSparkModel"]
    cfg.block_size = BLOCK
    cfg.num_target_layers = 4
    cfg.target_layer_ids = [0]
    cfg.mask_token_id = 1
    cfg.num_anchors = 2
    cfg.markov_rank = 8
    cfg.markov_head_type = "vanilla"
    cfg.enable_confidence_head = True
    cfg.confidence_head_with_markov = True
    cfg.layer_types = ["full_attention"] * 2
    cfg._attn_implementation = "eager"
    return cfg


@pytest.fixture(scope="module")
def drafter():
    if check_transformers_symbols():
        pytest.skip(f"missing transformers symbols: {check_transformers_symbols()}")
    torch.manual_seed(0)
    model = Qwen3DSparkModel(_tiny_config()).eval()
    return DSparkDrafter(model, device=torch.device("cpu"))


def _ctx(seg: int) -> torch.Tensor:
    """假 context 特征：[1, seg, len(target_layer_ids)*hidden]。"""
    return torch.randn(1, seg, HIDDEN)


class TestKvProtocol:
    def test_first_propose_keeps_only_context(self, drafter):
        start = SEG0
        tokens, probs = drafter.propose(
            1, anchor_token_id=7, start=start, first_ctx=_ctx(SEG0)
        )
        assert len(tokens) == BLOCK
        assert all(0 <= t < VOCAB for t in tokens)
        assert probs is None, "greedy 下 q 是 one-hot，不该返回分布"
        assert drafter.ctx_len(1) == start, "crop 后必须恰好等于已提交长度"

    def test_second_propose_appends_new_segment(self, drafter):
        drafter.reset(1)
        start = SEG0
        drafter.propose(1, anchor_token_id=7, start=start, first_ctx=_ctx(SEG0))
        # 上一步接受了 SEG1-1 个草稿：已提交长度 +SEG1
        new_start = start + SEG1
        drafter.update(1, _ctx(SEG1), start=new_start)
        tokens, _ = drafter.propose(1, anchor_token_id=9, start=new_start)
        assert len(tokens) == BLOCK
        assert drafter.ctx_len(1) == new_start

    def test_three_steps_stay_in_sync(self, drafter):
        drafter.reset(2)
        start = SEG0
        drafter.propose(2, anchor_token_id=3, start=start, first_ctx=_ctx(SEG0))
        assert drafter.ctx_len(2) == start
        for _ in range(3):
            start += SEG1
            drafter.update(2, _ctx(SEG1), start=start)
            drafter.propose(2, anchor_token_id=4, start=start)
            assert drafter.ctx_len(2) == start

    def test_ledger_drift_drops_state(self, drafter):
        drafter.reset(3)
        drafter.propose(3, anchor_token_id=1, start=SEG0, first_ctx=_ctx(SEG0))
        # 模拟 preempt 后长度被重置：与记录不一致 → 本步放弃起草
        tokens, probs = drafter.propose(3, anchor_token_id=1, start=SEG0 + 99)
        assert tokens == [] and probs is None
        assert drafter.ctx_len(3) == 0, "状态应被丢弃，避免用过期 context 起草"

    def test_no_context_returns_empty(self, drafter):
        drafter.reset(4)
        tokens, probs = drafter.propose(4, anchor_token_id=1, start=SEG0)
        assert tokens == [] and probs is None

    def test_greedy_is_deterministic(self, drafter):
        outs = []
        for _ in range(2):
            drafter.reset(5)
            torch.manual_seed(0)
            tokens, _ = drafter.propose(
                5, anchor_token_id=11, start=SEG0, first_ctx=_ctx(SEG0)
            )
            outs.append(tokens)
        assert outs[0] == outs[1]

    def test_truncates_to_max_proposal(self, drafter):
        """块宽由 checkpoint 固定（7），但引擎的 --spec-tokens 可能更小。

        多给会让 verify 写到**没分配块**的位置上，必须在 worker 侧截断。
        """
        dr = DSparkDrafter(
            drafter.model, device=torch.device("cpu"), max_proposal=2
        )
        tokens, probs = dr.propose(
            7, anchor_token_id=5, start=SEG0, first_ctx=_ctx(SEG0)
        )
        assert len(tokens) == 2, f"应截到 2，实际 {len(tokens)}"

    def test_accepts_2d_context(self, drafter):
        """引擎侧传的是二维切片 [n, F]，drafter 要能容忍（否则 cat 报 got 2 and 3）。"""
        drafter.reset(8)
        tokens, _ = drafter.propose(
            8, anchor_token_id=5, start=SEG0, first_ctx=_ctx(SEG0).squeeze(0)
        )
        assert len(tokens) == BLOCK

    def test_temperature_returns_probs(self, drafter):
        drafter.temperature = 0.8
        try:
            drafter.reset(6)
            tokens, probs = drafter.propose(
                6, anchor_token_id=11, start=SEG0, first_ctx=_ctx(SEG0)
            )
            assert len(tokens) == BLOCK
            assert probs is not None and probs.shape == (BLOCK, VOCAB)
            # 逐行归一化
            assert torch.allclose(probs.sum(-1), torch.ones(BLOCK), atol=1e-5)
        finally:
            drafter.temperature = 0.0
