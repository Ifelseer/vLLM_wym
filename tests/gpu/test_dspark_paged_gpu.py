"""页式草稿 KV 的 GPU 验收。

分三件事，各自的口径不同：

1. **路径一 vs 稠密**：token 必须逐位相同、接受数相同（数学等价，只是存放位置不同）；
2. **页池耗尽**：允许退化（该序列本步不起草），但不许输出错；
3. **batch 模式下的显存对照**：那才是「671 MB → O(实际 token)」的对照对象
   （单序列的 dense 走 `DynamicCache`，只有约 1.6 MB，比页池还小，不能拿来做断言）。

读路径二（FA paged）**不在这里断言逐位**：换了 attention 实现，归约顺序不同，
只能比 `accept_len`（实测与路径一完全一致：4.2667 / 98 accepted / 128 token 逐位相同）。

需要：Qwen3-8B + dspark_qwen3_8b_block7。用 FP8 目标把显存压到 ~16G。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from tests.conftest import needs_cuda
from tests.helpers.engine import cfg, engine_session
from src import SamplingParams

MODEL = Path(os.path.expanduser("~/huggingface/Qwen3-8B"))
DRAFT = Path(os.path.expanduser("~/huggingface/dspark_qwen3_8b_block7"))

pytestmark = [
    pytest.mark.gpu,
    needs_cuda,
    pytest.mark.skipif(not MODEL.is_dir(), reason=f"missing target: {MODEL}"),
    pytest.mark.skipif(not DRAFT.is_dir(), reason=f"missing draft: {DRAFT}"),
]

_N = 48
_PROMPT = "Count down from 200 by threes, separated by commas: 200, 197, 194,"


def _run(
    *,
    paged: bool,
    pages: int = 512,
    n: int = _N,
    draft_batch: bool = False,
    max_num_seqs: int = 1,
):
    with engine_session(
        cfg(
            str(MODEL),
            max_num_seqs=max_num_seqs,
            enforce_eager=False,
            quantization="fp8",
            spec_tokens=7,
            spec_draft="dspark",
            spec_draft_model=str(DRAFT),
            spec_draft_batch=draft_batch,
            spec_draft_paged=paged,
            spec_draft_page_size=16,
            spec_draft_paged_pages=pages,
        )
    ) as eng:
        seqs = [
            eng.add_request(
                _PROMPT, SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
            )
            for _ in range(max_num_seqs)
        ]
        while not all(s.is_finished for s in seqs):
            eng.step()
        outs = [list(s.token_ids[s.num_prompt_tokens :]) for s in seqs]
        stats = dict(eng.scheduler.spec_stats)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    return outs, stats, peak


def test_paged_matches_dense_tokens():
    (base,), base_stats, base_peak = _run(paged=False)
    (got,), stats, paged_peak = _run(paged=True)

    assert got == base, (
        "页式与稠密的 greedy 输出不一致 → 查页表映射 / 写入偏移 / 退页边界\n"
        f"first diff at "
        f"{next((i for i, (a, b) in enumerate(zip(base, got)) if a != b), None)}\n"
        f"dense={base[:32]}\npaged={got[:32]}"
    )
    assert stats["accepted"] == base_stats["accepted"], (
        f"接受数不同：dense {base_stats['accepted']} vs paged {stats['accepted']}"
    )
    # 单序列的 dense 走 DynamicCache（~1.6 MB），页池按 num_pages 预留（512 页
    # = 8192 token ≈168 MB）——**这个用例里 paged 本来就更大**。所以只做宽松回归
    # 断言（防翻倍级问题），真正的显存对照见 test_batch_mode_paged_uses_less_memory。
    assert paged_peak <= base_peak + 0.5, (
        f"页式显存异常膨胀：dense {base_peak:.2f} GB vs paged {paged_peak:.2f} GB"
    )


def test_batch_mode_paged_uses_less_memory():
    """batch 模式：dense 走 PaddedKVCache（行数 × max_len），paged 按需分配。

    这是文档里「671 MB / 1.34 GB → O(实际 token)」的对照场景。
    """
    _, _, dense_peak = _run(paged=False, draft_batch=True)
    _, _, paged_peak = _run(paged=True, draft_batch=True, pages=256)
    assert paged_peak < dense_peak, (
        f"batch 模式下页式应更省：dense {dense_peak:.2f} GB vs paged {paged_peak:.2f} GB"
    )


def test_pool_exhaustion_degrades_but_stays_correct():
    """页池故意给很小：允许退化（该序列本步不起草），但输出不许错。"""
    (base,), _, _ = _run(paged=False, n=32)
    (got,), stats, _ = _run(paged=True, pages=4, n=32)   # 只够 64 token
    assert got == base[: len(got)], "页池耗尽时输出偏离 → 兜底路径写错了"
    # 计数不变量（stats 的值是 int，不能对 int 调 sum）
    assert stats["accepted"] <= stats["proposed"]
    assert stats["steps"] > 0
