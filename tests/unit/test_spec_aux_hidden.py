"""DSpark aux hidden 的配置契约（无需 GPU / 无需权重）。

真正的采集逻辑（`Qwen3Model.set_aux_hidden_layers` + 循环里 `out=` 原地写入）
必须在 GPU 上跑一次前向才能验证；这里只钉住"哪个组合被允许、哪个必须报错"，
避免上线时把 PP>1 或空层表悄悄放进去。

`EngineConfig.__post_init__` 会校验模型目录并加载 `config.json`，所以用
`tmp_path` 造一个**只有 config.json** 的极小 Qwen3（不加载任何权重）。
"""

from __future__ import annotations

import json

import pytest

from src.config import EngineConfig

_TINY_QWEN3 = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 128,
    "max_position_embeddings": 128,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> str:
    d = tmp_path_factory.mktemp("tiny-qwen3")
    (d / "config.json").write_text(json.dumps(_TINY_QWEN3), encoding="utf-8")
    return str(d)


def _cfg(model_dir: str, **kw) -> EngineConfig:
    base = dict(
        model=model_dir,
        device_ids=[0],
        max_num_seqs=1,
        spec_tokens=4,
        spec_draft="dspark",
        spec_draft_model=model_dir,  # 只校验是目录，不校验内容
    )
    base.update(kw)
    return EngineConfig(**base)


class TestDsparkConfig:
    def test_defaults(self, model_dir):
        cfg = _cfg(model_dir)
        assert cfg.spec_target_layer_ids == (1, 9, 17, 25, 33)
        assert cfg.spec_exact_kv is False

    def test_requires_single_pp(self, model_dir):
        with pytest.raises(ValueError, match="pp_size == 1"):
            _cfg(model_dir, pp_size=2)

    def test_requires_single_tp(self, model_dir):
        with pytest.raises(ValueError, match="tp_size == 1"):
            _cfg(model_dir, tp_size=2)

    def test_requires_layer_ids(self, model_dir):
        with pytest.raises(ValueError, match="spec_target_layer_ids"):
            _cfg(model_dir, spec_target_layer_ids=())

    def test_requires_draft_model(self, model_dir):
        with pytest.raises(ValueError, match="spec_draft_model"):
            _cfg(model_dir, spec_draft_model="")

    def test_missing_draft_dir(self, model_dir):
        with pytest.raises(FileNotFoundError, match="draft model path not found"):
            _cfg(model_dir, spec_draft_model=str(model_dir) + "-nope")

    def test_ngram_does_not_need_layer_ids(self, model_dir):
        cfg = _cfg(
            model_dir,
            spec_draft="ngram",
            spec_target_layer_ids=(),
            spec_draft_model="",
        )
        assert cfg.spec_draft == "ngram"

    def test_spec_off_by_default(self, model_dir):
        cfg = EngineConfig(model=model_dir, device_ids=[0], max_num_seqs=1)
        assert cfg.spec_tokens == 0
        assert cfg.spec_draft == "ngram"
