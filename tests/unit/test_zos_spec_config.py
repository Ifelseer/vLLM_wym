"""投机 × ZOS 的配置护栏（方案 B）。

这三条不是"多此一举"的校验，每一条都对应一个**跑到一半才会崩**的组合：

| 组合 | 不加护栏的后果 |
|---|---|
| `zos_spec=True` + `zos_device_accept=False` | `_next_input` 拿不到 `_zos_next/_zos_adv` → 第一次 lookahead 抛 `RuntimeError: spec decode without device-side accept` |
| `zos_spec=True` + `spec_draft="ngram"` | ngram 的草稿数逐拍变化 → `m` 不固定 → "按 m 追加"的草稿 KV 立即错位 |
| `zos_spec=True` + `enable_zos=False` | 开关静默失效（用户以为开了） |

所以放在构造期 fail loud。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import EngineConfig

#: 配置构造期会校验 model 目录存在；这里用一个真实目录（仓库根）即可，
#: 本测试只关心 zos/spec 的组合护栏，不加载任何权重。
_ANY_DIR = str(Path(__file__).resolve().parents[2])


def _cfg(**kw) -> EngineConfig:
    base = dict(
        model=_ANY_DIR,
        enable_zos=True,
        zos_spec=True,
        zos_device_accept=True,
        spec_tokens=7,
        spec_draft="dspark",
        spec_draft_model=_ANY_DIR,
        max_num_seqs=1,
    )
    base.update(kw)
    return EngineConfig(**base)


# 注：不测"合法组合能构造成功"——那会走到 AutoConfig 加载真实模型（需要一个
# 真 checkpoint 目录）。护栏的价值在"非法组合必须报错"，那三条已经覆盖。

def test_guards_run_before_model_loading():
    """护栏必须早于"需要真实权重"的校验，否则用户先撞上无关的报错。"""
    # spec_draft_model 指向仓库根（不是真 checkpoint）：三条护栏仍然先触发
    with pytest.raises(ValueError, match="zos_device_accept"):
        _cfg(zos_device_accept=False, spec_draft_model=str(Path(__file__).resolve()))


def test_requires_device_accept():
    with pytest.raises(ValueError, match="zos_device_accept"):
        _cfg(zos_device_accept=False)


def test_requires_dspark():
    """ngram 的草稿数逐拍不定，无法按固定 m 追加草稿 KV。"""
    with pytest.raises(ValueError, match="dspark"):
        _cfg(spec_draft="ngram")


def test_requires_zos_enabled():
    with pytest.raises(ValueError, match="enable_zos"):
        _cfg(enable_zos=False)
