"""量化端到端正确性 gate（两条独立验收口径）。

1. **数值**（``test_quant_greedy_consistency``）：teacher-forced 对比。
   先用 BF16 贪心生成 N 个 token，再让 BF16 与量化模型在**完全相同的上下文**
   （prompt + 前 i 个 token）下各做一次贪心 argmax，比较：
   - 决定性位置（BF16 自身 top1-top2 间隔 ≥ 0.5）上的 top-1 一致率；
   - top-1 logit 的最大绝对偏差。
   近似并列位置（margin ≤ 0.5，约 1 个 bf16 ulp）不计——量化噪声本来就会
   翻转 argmax。旧口径「同一 prompt 自由生成 N token 逐位一致率 ≥99%」不可达：
   一次并列翻转会级联污染后续所有位置（16 token 里翻 3 个 = 81%），且 fp8 与
   int8 会给出逐位相同的序列——那是解码混沌，不是量化错误。

2. **CUDA Graph**（``test_quant_greedy_cuda_graph``）：同一量化档位的 eager 与
   CG 必须产生**完全相同**的 token 序列。量化 kernel 若在 capture 期间做非法
   同步/分配（autotune、D2H）会直接崩；若录图缓冲/scale 复用有误则序列不一致。
   该口径不依赖量化精度，专测 capture/replay 通路本身。

每个档位在**独立子进程**里跑（每种量化一次模型加载，避免显存碎片与全局状态
污染），再由本进程比对。fp8 需 SM89+，不满足时 worker 打印 SKIP 并转 skip。
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import needs_cuda
from tests.helpers import DENSE_MODEL, pick_gpus, require_model
from tests.helpers.devices import run_isolated

pytestmark = [pytest.mark.gpu, needs_cuda, require_model(DENSE_MODEL)]

_N = 16

# 决定性位置的 top-1 一致率门槛（实测 fp8 / int8 均为 14/14 = 100%）
_TOKEN_GATE = 0.95
# 近似并列判定：BF16 自身 top1-top2 logit 间隔（logits ~17-22，1 ulp≈0.125）
_DECISIVE_MARGIN = 0.5
# top-1 logit 最大绝对偏差上限（实测 fp8 1.125 / int8 0.75）
_LOGIT_GATE = 1.5

_WORKER = r"""
import json
import sys

import torch

from tests.helpers.engine import cfg, engine_session
from tests.helpers.paths import DENSE_MODEL
from src import SamplingParams

quant, n, port, mode, teacher_json, eager = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
    sys.argv[4], sys.argv[5], sys.argv[6],
)
teacher = json.loads(teacher_json) if teacher_json else []

major, minor = torch.cuda.get_device_capability(0)
if quant == "fp8" and (major, minor) < (8, 9):
    print(f"SKIP fp8 requires SM89+ (got SM{major}{minor})")
    raise SystemExit(0)

rec = []
if mode == "probe":
    # greedy 路径走 model.argmax（不经过 Sampler），patch 它记录 top-2 logits。
    # 仅 probe 模式需要，且只在 eager 下有效（CG replay 不执行 Python 回调）。
    from src.compute.models.qwen3 import Qwen3ForCausalLM

    _orig = Qwen3ForCausalLM.argmax

    def patched(self, hidden):
        logits = self.compute_logits(hidden).float()
        top = logits.topk(2, dim=-1)
        rec.append({"t1": top.values[:, 0].tolist(), "t2": top.values[:, 1].tolist()})
        return logits.argmax(dim=-1)

    Qwen3ForCausalLM.argmax = patched

config = cfg(
    DENSE_MODEL, device_ids=[0], quantization=quant, max_num_seqs=1,
    context_len=512, num_kvcache_blocks=16, nccl_port=port,
    enforce_eager=(eager == "1"),
)
with engine_session(config) as eng:
    prompt_ids = eng.tokenizer.encode("The capital of France is")
    base = teacher
    if not base:
        seq = eng.add_request(
            prompt_ids, SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        )
        while not seq.is_finished:
            eng.step()
        base = list(seq.token_ids[seq.num_prompt_tokens:])
    out = {"tokens": base, "probes": []}
    if mode == "probe":
        # teacher forcing：每个前缀单独一条 max_tokens=1 请求，上下文与基线一致
        for i in range(n):
            del rec[:]
            seq = eng.add_request(
                prompt_ids + base[:i],
                SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
            )
            while not seq.is_finished:
                eng.step()
            r = rec[-1]
            out["probes"].append(
                {
                    "tok": int(seq.token_ids[-1]),
                    "top1": float(r["t1"][0]),
                    "top2": float(r["t2"][0]),
                }
            )
print(json.dumps(out))
"""


def _run(
    quant: str,
    *,
    mode: str,
    port: int,
    teacher: list[int] | None = None,
    eager: bool = True,
) -> dict | None:
    """跑一个隔离 worker；环境不支持该量化档位时返回 None。"""
    gpus = pick_gpus(1, min_free_mib=4000)
    out = run_isolated(
        gpus,
        _WORKER,
        quant,
        str(_N),
        str(port),
        mode,
        json.dumps(teacher or []),
        "1" if eager else "0",
        timeout=1800,
    )
    if "SKIP" in out:
        return None
    lines = [ln for ln in out.splitlines() if ln.strip().startswith("{")]
    assert lines, out
    return json.loads(lines[-1])


def test_quant_greedy_consistency():
    """相同上下文下的 teacher-forced top-1 / logit 对齐（vs BF16 基线）。"""
    base = _run("none", mode="probe", port=29651)
    assert base and len(base["tokens"]) == _N, f"bad baseline: {base}"
    assert len(base["probes"]) == _N, f"bad baseline probes: {base}"

    ran = 0
    for quant, port in (("int8", 29652), ("fp8", 29653)):
        got = _run(quant, mode="probe", port=port, teacher=base["tokens"])
        if got is None:
            continue
        pb, pq = base["probes"], got["probes"]
        decisive = [
            i for i, p in enumerate(pb) if p["top1"] - p["top2"] >= _DECISIVE_MARGIN
        ]
        assert decisive, "baseline has no decisive position"
        hits = sum(pb[i]["tok"] == pq[i]["tok"] for i in decisive)
        agree = hits / len(decisive)
        dmax = max(abs(pb[i]["top1"] - pq[i]["top1"]) for i in range(_N))
        assert agree >= _TOKEN_GATE, (
            f"quant={quant}: decisive top-1 agreement {agree:.1%} < "
            f"{_TOKEN_GATE:.0%} ({hits}/{len(decisive)})\n"
            f"base={[p['tok'] for p in pb]}\n"
            f"got ={[p['tok'] for p in pq]}"
        )
        assert dmax <= _LOGIT_GATE, (
            f"quant={quant}: max |Δ top-1 logit| {dmax:.2f} > {_LOGIT_GATE}"
        )
        ran += 1
    assert ran > 0, "no quantization level could run on this GPU (need SM86+ for int8)"


@pytest.mark.parametrize("quant", ["int8", "fp8"])
def test_quant_greedy_cuda_graph(quant: str):
    """同一量化档位下 eager 与 CUDA Graph 的 token 序列必须完全一致。"""
    eager = _run(quant, mode="gen", port=29661, eager=True)
    if eager is None:
        pytest.skip(f"{quant} unsupported on this GPU")
    cg = _run(quant, mode="gen", port=29662, eager=False)
    if cg is None:
        pytest.skip(f"{quant} unsupported on this GPU")
    assert eager["tokens"] == cg["tokens"], (
        f"quant={quant}: eager vs CUDA Graph token mismatch\n"
        f"eager={eager['tokens']}\n"
        f"cg   ={cg['tokens']}"
    )
