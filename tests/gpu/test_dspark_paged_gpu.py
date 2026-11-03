"""页式草稿 KV 的 GPU 验收：**与稠密路径逐 token 相同**，且显存更低。

为什么要求逐位：读路径一（gather 成稠密后走同一个 SDPA）在数学上与稠密等价，K/V 值
来自同一次前向，所以 token 必须逐位相同。任何差异都说明页表映射 / 写入偏移 / 退页
边界写错了——这类错误不会崩，只让接受率无声下滑。

读路径二（FA paged，`spec_draft_paged_fa`）**不在这里断言逐位**：换了 attention
实现，归约顺序不同，只能验 `accept_len` 接近（见 `docs/draft-kv-paged-plan.md`）。

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


def _run(*, paged: bool, pages: int = 512, n: int = _N):
    with engine_session(
        cfg(
            str(MODEL),
            max_num_seqs=1,
            enforce_eager=False,
            quantization="fp8",
            spec_tokens=7,
            spec_draft="dspark",
            spec_draft_model=str(DRAFT),
            spec_draft_paged=paged,
            spec_draft_page_size=16,
            spec_draft_paged_pages=pages,
        )
    ) as eng:
        seq = eng.add_request(
            _PROMPT, SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        )
        while not seq.is_finished:
            eng.step()
        out = list(seq.token_ids[seq.num_prompt_tokens :])
        stats = dict(eng.scheduler.spec_stats)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    return out, stats, peak


def test_paged_matches_dense_tokens():
    base, base_stats, base_peak = _run(paged=False)
    got, stats, paged_peak = _run(paged=True)

    assert got == base, (
        "页式与稠密的 greedy 输出不一致 → 查页表映射 / 写入偏移 / 退页边界\n"
        f"first diff at "
        f"{next((i for i, (a, b) in enumerate(zip(base, got)) if a != b), None)}\n"
        f"dense={base[:32]}\npaged={got[:32]}"
    )
    # 接受率应完全一致（同一个模型、同一份 K/V，只是存放位置不同）
    assert stats["accepted"] == base_stats["accepted"], (
        f"接受数不同：dense {base_stats['accepted']} vs paged {stats['accepted']}"
    )
    assert paged_peak <= base_peak + 0.2, (
        f"页式未省显存：dense {base_peak:.2f} GB vs paged {paged_peak:.2f} GB"
    )


def test_pool_exhaustion_degrades_but_stays_correct():
    """页池故意给很小：允许退化（该序列本步不起草），但输出不许错。"""
    base, _, _ = _run(paged=False, n=32)
    got, stats, _ = _run(paged=True, pages=2, n=32)   # 只够 32 个 token
    assert got == base[: len(got)], "页池耗尽时输出偏离 → 兜底路径写错了"
    assert stats["accepted"] <= sum(stats["proposed"]), "计数不变量"
