"""vendored DSpark 草稿模型的 CPU 冒烟测试（无需 checkpoint、无需 GPU）。

为什么需要它：vendor 进来的 4 个文件是**别人训练时用的代码**，我们只改了 import。
如果哪天上游更新或 transformers 版本漂移，最可能的失败点是符号缺失、双流注意力
的形状、markov/confidence 头的接口。这些都能用一个"极小配置 + 随机权重"的前向
在 CPU 上一次性覆盖，不必等 GPU 和 4.5 GB 的 checkpoint。

这里**不测数值正确性**（没有官方权重可比），只测结构契约：
- vendored 模块能 import（说明 deepspec.* 依赖已彻底切断）；
- 8 个 transformers 符号齐全；
- `_forward_backbone` 在 [context | block] 双流下形状正确、块内非因果；
- `sample_draft_tokens` 走 markov 半自回归路径，输出 [B, block]；
- `predict_confidence_step` 输出逐位置 logit。
"""

from __future__ import annotations

import pytest
import torch

from src.spec.dspark import Qwen3DSparkModel, extract_context_feature
from src.spec.dspark.loader import (
    REQUIRED_CONFIG_FIELDS,
    check_config_leakage,
    check_transformers_symbols,
)

HIDDEN = 64
VOCAB = 128
BLOCK = 3
CTX = 4


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
    # DSpark 专有字段（官方 config.json 里同样以扁平字段出现）
    cfg.architectures = ["Qwen3DSparkModel"]
    cfg.block_size = BLOCK
    cfg.num_target_layers = 4
    cfg.target_layer_ids = [0]  # fc 输入维度 = 1 * hidden
    cfg.mask_token_id = 1
    cfg.num_anchors = 2
    cfg.markov_rank = 8
    cfg.markov_head_type = "vanilla"
    cfg.enable_confidence_head = True
    cfg.confidence_head_with_markov = True
    cfg.layer_types = ["full_attention"] * 2
    cfg._attn_implementation = "eager"  # CPU 上不依赖 flash-attn
    return cfg


@pytest.fixture(scope="module")
def draft():
    if check_transformers_symbols():
        pytest.skip(f"missing transformers symbols: {check_transformers_symbols()}")
    torch.manual_seed(0)
    return Qwen3DSparkModel(_tiny_config()).eval()


class TestSymbolsAndConfig:
    def test_no_missing_symbols(self):
        assert check_transformers_symbols() == []

    def test_config_fields_present(self):
        cfg = _tiny_config()
        assert check_config_leakage(cfg) == []

    def test_config_leakage_detected(self):
        cfg = _tiny_config()
        del cfg.markov_rank
        assert "markov_rank" in check_config_leakage(cfg)
        assert set(REQUIRED_CONFIG_FIELDS) - {"markov_rank"} <= set(dir(cfg))


class TestModelContract:
    def test_module_shapes(self, draft):
        assert draft.block_size == BLOCK
        assert draft.mask_token_id == 1
        assert draft.lm_head.weight.shape == (VOCAB, HIDDEN)
        # fc 把 len(target_layer_ids)*hidden 投回 hidden
        assert draft.fc.weight.shape == (HIDDEN, HIDDEN)
        assert draft.markov_head is not None
        assert draft.markov_head.markov_w1.weight.shape == (VOCAB, 8)
        assert draft.confidence_head is not None
        # confidence 输入 = hidden + markov_rank
        assert draft.confidence_head.proj.weight.shape == (1, HIDDEN + 8)

    def test_dual_stream_backbone(self, draft):
        """一次前向处理 [anchor | noise×block]：q 长度 = block，kv 长度 = ctx+block。"""
        noise = draft.embed_tokens(torch.randint(2, VOCAB, (1, BLOCK)))
        target_hidden = torch.randn(1, CTX, HIDDEN)  # 1 个 target 层 → hidden
        pos = torch.arange(CTX + BLOCK).unsqueeze(0)
        out = draft._forward_backbone(
            position_ids=pos,
            noise_embedding=noise,
            target_hidden_states=target_hidden,
            attention_mask=None,
            use_cache=False,
            is_causal=False,
        )
        assert out.shape == (1, BLOCK, HIDDEN)
        assert torch.isfinite(out).all()

    def test_markov_sampling_and_confidence(self, draft):
        base_logits = torch.randn(1, BLOCK, VOCAB)
        tokens, corrected = draft.sample_draft_tokens(
            base_logits,
            first_prev_token_ids=torch.tensor([7]),
            temperature=0.0,
            hidden_states=torch.randn(1, BLOCK, HIDDEN),
        )
        assert tokens.shape == (1, BLOCK)
        assert corrected.shape == (1, BLOCK, VOCAB)
        # markov 偏置确实改过 logits（否则是 vanilla softmax 采样）
        assert not torch.allclose(corrected, base_logits)

        prev = torch.cat([torch.tensor([[7]]), tokens[:, :-1]], dim=1)
        conf = draft.predict_confidence_step(
            torch.randn(1, BLOCK, HIDDEN), prev_token_ids=prev
        )
        assert conf is not None and conf.shape == (1, BLOCK)

    def test_extract_context_feature(self):
        """layer_ids 是 0 基；-1 表示 embedding 输出。"""
        hidden = [torch.randn(1, 2, HIDDEN) for _ in range(5)]  # embedding + 4 层
        got = extract_context_feature(hidden, [0, 2])
        assert got.shape == (1, 2, 2 * HIDDEN)
        assert torch.equal(got, torch.cat([hidden[1], hidden[3]], dim=-1))
        got_emb = extract_context_feature(hidden, [-1])
        assert torch.equal(got_emb, hidden[0])
