"""批量起草 vs 逐序列起草：**token 必须逐位相同**。

这是草稿批量化的唯一验收标准。两者的区别只在"一次前向算 B 行"还是"B 次前向各算
一行"，模型的数学不变，所以输出必须完全相同；任何差异都说明 padding/掩码/行偏移
写错了——而这种错误不会崩，只会让接受率无声下滑。

用随机权重的极小模型在 CPU 上跑，不需要 checkpoint 也不需要 GPU。
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


def _tiny_config():
    from transformers import Qwen3Config

    cfg = Qwen3Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=HIDDEN * 2,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        attention_bias=False, attention_dropout=0.0, rms_norm_eps=1e-6,
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
def model():
    if check_transformers_symbols():
        pytest.skip(f"missing symbols: {check_transformers_symbols()}")
    torch.manual_seed(0)
    return Qwen3DSparkModel(_tiny_config()).eval()


def _ctx(seg: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seg)
    return torch.randn(1, seg, HIDDEN, generator=g)


class TestBatchEquivalence:
    def test_batch_matches_single(self, model):
        """两条序列（prompt 长度不同）分别起草，与批量起草结果逐位比对。"""
        seg_a, seg_b = 5, 4
        ctx_a, ctx_b = _ctx(seg_a), _ctx(seg_b)

        # 单跑：各自独立的 drafter（互不共享状态）
        single = []
        for seq_id, (seg, ctx, anchor, start) in enumerate(
            [(seg_a, ctx_a, 11, seg_a), (seg_b, ctx_b, 22, seg_b)], start=1
        ):
            d = DSparkDrafter(model, device=torch.device("cpu"))
            toks, _ = d.propose(
                seq_id, anchor_token_id=anchor, start=start, first_ctx=ctx
            )
            single.append(toks)

        # 批跑：同一个 drafter，一次前向
        batch_d = DSparkDrafter(model, device=torch.device("cpu"), max_rows=4)
        out = batch_d.propose_batch(
            [
                (1, 11, seg_a, ctx_a),
                (2, 22, seg_b, ctx_b),
            ]
        )
        assert len(out) == 2
        assert out[0][0] == single[0], f"seq0 batch={out[0][0]} single={single[0]}"
        assert out[1][0] == single[1], f"seq1 batch={out[1][0]} single={single[1]}"

    def test_batch_second_step_matches_single(self, model):
        """第二步（各自已有不同长度的 context）同样必须一致。"""
        seg_a, seg_b, next_a, next_b = 5, 4, 2, 3
        single_toks = []
        for seq_id, (seg, nxt, anchor) in enumerate(
            [(seg_a, next_a, 11), (seg_b, next_b, 22)], start=1
        ):
            d = DSparkDrafter(model, device=torch.device("cpu"))
            d.propose(seq_id, anchor_token_id=anchor, start=seg, first_ctx=_ctx(seg))
            d.update(seq_id, _ctx(nxt), start=seg + nxt)
            toks, _ = d.propose(seq_id, anchor_token_id=anchor + 1, start=seg + nxt)
            single_toks.append(toks)

        bd = DSparkDrafter(model, device=torch.device("cpu"), max_rows=4)
        bd.propose_batch([(1, 11, seg_a, _ctx(seg_a)), (2, 22, seg_b, _ctx(seg_b))])
        bd.update(1, _ctx(next_a), start=seg_a + next_a)
        bd.update(2, _ctx(next_b), start=seg_b + next_b)
        out = bd.propose_batch(
            [(1, 12, seg_a + next_a, None), (2, 23, seg_b + next_b, None)]
        )
        assert out[0][0] == single_toks[0]
        assert out[1][0] == single_toks[1]

    def test_falls_back_when_only_one_participant(self, model):
        """只有一个参与者时走单序列路径（结果仍要正确）。"""
        ctx = _ctx(5)
        d = DSparkDrafter(model, device=torch.device("cpu"))
        out = d.propose_batch([(1, 7, 5, ctx), (2, 8, 5, None)])
        assert len(out[0][0]) == BLOCK
        assert out[1] == ([], None), "没有 context 的序列不该被起草"
