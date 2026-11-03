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


class TestPagedEquivalence:
    """页式 KV 与稠密 KV 必须产出**逐位相同**的草稿 token。

    两者只是"K/V 存在哪"不同（页池 + 页表 vs 每行预留满长度），数学完全一样。
    任何差异都说明页表映射 / 写入偏移 / 退页边界写错了——而这类错误不会崩，
    只会让接受率无声下滑。

    注意：这是**同一实现家族内部**的比较（都走 SDPA），所以要求逐位；将来换成
    FA paged 之后，跨实现的比较只能放宽到 accept_len 差值。
    """

    def _pair(self, model, *, seed: int = 0):
        torch.manual_seed(seed)
        dense = DSparkDrafter(model, device=torch.device("cpu"))
        torch.manual_seed(seed)
        paged = DSparkDrafter(
            model,
            device=torch.device("cpu"),
            max_rows=4,
            use_paged=True,
            page_size=2,
            num_pages=64,
        )
        return dense, paged

    def test_single_sequence_two_steps(self, model):
        seg = 3
        ctx1 = _ctx(seg)
        dense, paged = self._pair(model)

        t_dense, _ = dense.propose(1, anchor_token_id=7, start=seg, first_ctx=ctx1)
        t_paged, _ = paged.propose(1, anchor_token_id=7, start=seg, first_ctx=ctx1)
        assert t_dense, "稠密路径不应返回空草稿"
        assert t_paged == t_dense, f"paged={t_paged} dense={t_dense}"

        # 第二步：各自追加 2 个 context（引擎侧由 update 推进 start）
        ctx2 = _ctx(2)
        start2 = seg + 2
        dense.update(1, ctx2, start=start2)
        paged.update(1, ctx2, start=start2)
        d2, _ = dense.propose(1, anchor_token_id=9, start=start2, first_ctx=None)
        p2, _ = paged.propose(1, anchor_token_id=9, start=start2, first_ctx=None)
        assert d2, "第二步不应返回空草稿"
        assert p2 == d2, f"第二步 paged={p2} dense={d2}"

    def test_two_sequences_isolated(self, model):
        """两条序列在页池里各占各的页，不能串位。"""
        dense, paged = self._pair(model)
        ctx_a, ctx_b = _ctx(3), _ctx(2)
        da, _ = dense.propose(1, anchor_token_id=5, start=3, first_ctx=ctx_a)
        db, _ = dense.propose(2, anchor_token_id=6, start=2, first_ctx=ctx_b)
        pa, _ = paged.propose(1, anchor_token_id=5, start=3, first_ctx=ctx_a)
        pb, _ = paged.propose(2, anchor_token_id=6, start=2, first_ctx=ctx_b)
        assert pa == da and pb == db, f"paged=({pa},{pb}) dense=({da},{db})"

    def test_pool_exhaustion_degrades_to_no_draft(self, model):
        """页池不足时返回空草稿（只损收益），不能抛错或写错位置。"""
        torch.manual_seed(0)
        d = DSparkDrafter(
            model,
            device=torch.device("cpu"),
            max_rows=2,
            use_paged=True,
            page_size=4,
            num_pages=1,          # 只够 4 个 token
        )
        toks, _ = d.propose(1, anchor_token_id=7, start=32, first_ctx=_ctx(32))
        assert toks == [], "页池不足应退化为不起草"


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

    def test_tensor_entry_matches_int_entry(self, model):
        """ZOS 用张量入参的入口，必须与 int 入参版本逐位相同。

        ZOS 下 anchor/start 都是设备张量（CPU 不知道值），位置编码只能用 broadcast
        现算；一旦两版的位置/掩码有差，草稿 token 就会分叉——而 accept_len 掉一点
        几乎看不出来，所以这里逐位比对。
        """
        seg_a, seg_b = 5, 4
        ctx_a, ctx_b = _ctx(seg_a), _ctx(seg_b)
        anchor_a, anchor_b = 11, 22

        d_int = DSparkDrafter(model, device=torch.device("cpu"))
        want_a, _ = d_int.propose(
            1, anchor_token_id=anchor_a, start=seg_a, first_ctx=ctx_a
        )
        want_b, _ = d_int.propose(
            2, anchor_token_id=anchor_b, start=seg_b, first_ctx=ctx_b
        )

        d_t = DSparkDrafter(model, device=torch.device("cpu"), max_rows=4)
        toks, _ = d_t.propose_batch_ids(
            [1, 2],
            torch.tensor([anchor_a, anchor_b], dtype=torch.long),
            torch.tensor([seg_a, seg_b], dtype=torch.long),
            [ctx_a, ctx_b],
        )
        assert toks.shape[0] == 2
        assert toks[0].tolist() == want_a, f"seq0 tensor={toks[0].tolist()} int={want_a}"
        assert toks[1].tolist() == want_b, f"seq1 tensor={toks[1].tolist()} int={want_b}"

    def test_falls_back_when_only_one_participant(self, model):
        """只有一个参与者时走单序列路径（结果仍要正确）。"""
        ctx = _ctx(5)
        d = DSparkDrafter(model, device=torch.device("cpu"))
        out = d.propose_batch([(1, 7, 5, ctx), (2, 8, 5, None)])
        assert len(out[0][0]) == BLOCK
        assert out[1] == ([], None), "没有 context 的序列不该被起草"
