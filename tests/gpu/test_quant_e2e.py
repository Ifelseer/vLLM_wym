"""量化端到端正确性 gate：同 prompt 贪心逐 token 一致率（vs BF16 基线）。

这是计划里的验收口径（≥99%）在单卡上的落地——kernel 级数值对齐（
``test_quant_fp8/int4/int8``）只能证明算子正确，端到端还牵涉加载路由、
TP 切分、attention 通路与 CG 录图，任何一处错都会在这里表现为 token 分歧。

按 ``--quantizations`` 逐个在**独立子进程**里跑（每种量化一次模型加载，
避免显存碎片与全局状态污染），再由本进程比对。

本地 3060 可跑 int8（SM86 支持 int8 tensor core）；fp8 需 SM89+，
不满足时 worker 打印 SKIP 并由测试转为 skip。
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import needs_cuda
from tests.helpers import DENSE_MODEL, pick_gpus, require_model
from tests.helpers.devices import run_isolated

pytestmark = [pytest.mark.gpu, needs_cuda, require_model(DENSE_MODEL)]

_PROMPT = "The capital of France is"
_N = 16

# 各档位的一致性门槛（vs BF16）
_GATE = {"int8": 0.99, "fp8": 0.99, "auto": 0.95}

_WORKER = r"""
import json, sys
import torch
from tests.helpers.engine import cfg, greedy_tokens
from tests.helpers.paths import DENSE_MODEL

quant, n, port = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
major, minor = torch.cuda.get_device_capability(0)
if quant == "fp8" and (major, minor) < (8, 9):
    print(f"SKIP fp8 requires SM89+ (got SM{major}{minor})")
    raise SystemExit(0)
tokens = greedy_tokens(
    cfg(
        DENSE_MODEL,
        device_ids=[0],
        quantization=quant,
        max_num_seqs=1,
        context_len=512,
        num_kvcache_blocks=16,
        nccl_port=port,
    ),
    "The capital of France is",
    n=n,
)
print(json.dumps({"tokens": tokens}))
"""


def _tokens(quant: str, port: int) -> list[int] | None:
    """返回该量化档位的贪心 token；环境不支持时返回 None（跳过）。"""
    gpus = pick_gpus(1, min_free_mib=4000)
    out = run_isolated(gpus, _WORKER, quant, str(_N), str(port), timeout=1800)
    if "SKIP" in out:
        return None
    lines = [ln for ln in out.splitlines() if ln.strip().startswith("{")]
    assert lines, out
    return json.loads(lines[-1])["tokens"]


def _consistency(a: list[int], b: list[int]) -> float:
    if not a or not b:
        return float("nan")
    n = max(len(a), len(b))
    same = sum(1 for i in range(n) if i < len(a) and i < len(b) and a[i] == b[i])
    return same / n


def test_quant_greedy_consistency():
    """none / fp8 / int8 三档贪心结果与 BF16 基线的逐 token 一致率过门槛。"""
    base = _tokens("none", 29651)
    assert base and len(base) == _N, f"bad baseline tokens: {base}"

    ran = 0
    for quant, port in (("int8", 29652), ("fp8", 29653)):
        got = _tokens(quant, port)
        if got is None:
            continue
        rate = _consistency(base, got)
        assert rate >= _GATE[quant], (
            f"quant={quant}: greedy consistency {rate:.1%} < gate "
            f"{_GATE[quant]:.0%}\nbase={base}\ngot ={got}"
        )
        ran += 1
    assert ran > 0, "no quantization level could run on this GPU (need SM86+ for int8)"


@pytest.mark.parametrize("quant", ["int8", "fp8"])
def test_quant_greedy_cuda_graph(quant: str):
    """CG 路径下的一致性（decode 走录图，量化 kernel 必须能进图）。

    与上一个测试的区别：``enforce_eager=False``。量化 kernel 若在 capture
    期间做非法的同步/分配（如 autotune、D2H），这里会直接崩而不是数值错。
    """
    worker = _WORKER.replace(
        "max_num_seqs=1,", "max_num_seqs=1, enforce_eager=False,"
    )
    gpus = pick_gpus(1, min_free_mib=4000)
    base_out = run_isolated(gpus, worker, "none", str(_N), "29661", timeout=1800)
    if "SKIP" in base_out:
        pytest.skip("eager-only environment")
    base = json.loads(
        [ln for ln in base_out.splitlines() if ln.strip().startswith("{")][-1]
    )["tokens"]
    out = run_isolated(gpus, worker, quant, str(_N), "29662", timeout=1800)
    if "SKIP" in out:
        pytest.skip(f"{quant} unsupported on this GPU")
    got = json.loads(
        [ln for ln in out.splitlines() if ln.strip().startswith("{")][-1]
    )["tokens"]
    rate = _consistency(base, got)
    assert rate >= _GATE[quant], (
        f"quant={quant} (CG): consistency {rate:.1%} < {_GATE[quant]:.0%}\n"
        f"base={base}\ngot ={got}"
    )
