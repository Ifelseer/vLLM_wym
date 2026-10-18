"""量化 × 并行：TP / TP+EP 下的贪心一致性与跨格式对比。

两个层次的验证：

1. **并行一致性**：同一量化配置在 tp1 / tp2 / tp2+ep 下的贪心 token 必须
   一致——量化参数的 TP/EP 切分（per-channel scale 随输出维、per-block scale
   按 block 网格、g_idx 不切、INT4 packed 列的单位换算）任何一处写错都会
   在这里暴露为 token 分歧。
2. **跨格式一致性**：同模型的 none / fp8 / int8 / auto 贪心结果应与 BF16
   基线高度一致（≥99% 的量化验收门槛；int4 允许更低，见各测试注释）。

每个 (tp, ep, quant) 组合跑在独立子进程（一种 layout 一个进程），与
``test_moe_parallel.py`` 的模式一致。
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import needs_2gpu, needs_cuda
from tests.helpers import MOE_MODEL, pick_gpus, require_model
from tests.helpers.devices import run_isolated

pytestmark = [
    pytest.mark.distributed,
    needs_cuda,
    needs_2gpu,
    require_model(MOE_MODEL),
]

_PROMPT = "The capital of France is"
_N = 8

# 量化验收门槛（vs BF16 基线的贪心一致率）
_CONSISTENCY_GATE = {"none": 1.0, "fp8": 0.99, "int8": 0.99, "auto": 0.95}

_WORKER = r"""
import json, sys
from src.moe import resolve_moe_backend
from tests.helpers.engine import cfg, greedy_tokens
from tests.helpers.parallel import layout_ok
from tests.helpers.paths import MOE_MODEL

prompt, n, tp_s, ep_s, quant, port = sys.argv[1:7]
n, tp, enable_ep, port = int(n), int(tp_s), ep_s == "1", int(port)
ok, reason = layout_ok(MOE_MODEL, tp=tp, enable_ep=enable_ep)
if not ok:
    raise SystemExit(f"bad layout: {reason}")
backend = resolve_moe_backend("triton")
tokens = greedy_tokens(
    cfg(
        MOE_MODEL,
        device_ids=list(range(tp)),
        tp_size=tp,
        enable_ep=enable_ep,
        quantization=quant,
        max_num_seqs=1,
        context_len=512,
        num_kvcache_blocks=8,
        gpu_memory_utilization=0.70,
        moe_backend=backend,
        nccl_port=port,
    ),
    prompt,
    n=n,
)
print(json.dumps({"tokens": tokens}))
"""


def _tokens(phys: list[int], *, tp: int, enable_ep: bool, quant: str, port: int):
    out = run_isolated(
        phys[:tp],
        _WORKER,
        _PROMPT,
        str(_N),
        str(tp),
        "1" if enable_ep else "0",
        quant,
        str(port),
        timeout=1800,
    )
    lines = [ln for ln in out.splitlines() if ln.strip().startswith("{")]
    assert lines, out
    return json.loads(lines[-1])["tokens"]


def _consistency(a: list[int], b: list[int]) -> float:
    if not a or not b:
        return float("nan")
    n = max(len(a), len(b))
    same = sum(1 for i in range(n) if i < len(a) and i < len(b) and a[i] == b[i])
    return same / n


@pytest.mark.parametrize("quant", ["none", "fp8", "int8", "auto"])
def test_quant_tp_layouts_match_tp1(quant: str):
    """tp1 / tp2 / tp2+ep 的贪心结果必须与 tp1 完全一致。"""
    phys = pick_gpus(2, min_free_mib=12000)
    base = _tokens(phys, tp=1, enable_ep=False, quant=quant, port=29631)
    assert base, "empty baseline tokens"
    for tp, ep, port in ((2, False, 29632), (2, True, 29633)):
        got = _tokens(phys, tp=tp, enable_ep=ep, quant=quant, port=port)
        assert got == base, (
            f"quant={quant} tp={tp} ep={ep}: greedy tokens diverged from tp1\n"
            f"tp1={base}\ntp{tp}ep={ep}={got}"
        )


def test_quant_formats_match_bf16_baseline():
    """各量化格式 vs BF16 基线的贪心一致率须过门槛。

    tp1 单卡口径（多卡一致性由上一个测试覆盖）；MoE 模型的 experts 在
    ``auto`` 档按 checkpoint 的 quantization_config 解析。
    """
    phys = pick_gpus(1, min_free_mib=12000)
    base = _tokens(phys, tp=1, enable_ep=False, quant="none", port=29641)
    assert base, "empty BF16 baseline"
    for quant in ("fp8", "int8", "auto"):
        got = _tokens(phys, tp=1, enable_ep=False, quant=quant, port=29642)
        rate = _consistency(base, got)
        gate = _CONSISTENCY_GATE[quant]
        assert rate >= gate, (
            f"quant={quant}: greedy consistency {rate:.1%} < gate {gate:.0%}\n"
            f"base={base}\ngot ={got}"
        )
