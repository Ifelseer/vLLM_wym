"""投机解码 bench：同一 prompt 下对比 baseline(spec=0) 与 spec(k) 的
TTFT / decode 吞吐 / 接受率 / 接受长度 / 峰值显存 / greedy 一致率。

口径（与 ``bench/quant_bench.py`` 一致，便于横向比较）：

- **TTFT**：从 add_request 到首个 completion token 的墙钟时间。
- **decode_tok_s**：``(n-1) / (e2e - ttft)``。
- **accept_rate**：``accepted / proposed``（草稿被接受的比例）。
- **accept_len**：``(accepted + steps) / steps``，即"每个验证步平均产出多少
  token"——这是投机解码最直接的收益指标（上限是 k+1）。
- **一致率**：仅在 ``--temperature 0`` 有意义；投机解码必须**无损**，任何
  非 100% 都说明 verify/回滚有 bug（不是"精度差异"）。

用法::

    # baseline vs k=4（greedy，必须 100% 一致）
    python bench/spec_bench.py --model ~/huggingface/Qwen3-0.6B \
        --prompt-file prompt.txt --spec-tokens 4 --cg

    # 温度采样（一致率列无意义，看吞吐）
    python bench/spec_bench.py --model ~/huggingface/Qwen3-0.6B \
        --spec-tokens 4 --temperature 0.8 --cg

    # 与量化组合（Phase 5 矩阵的一格）
    python bench/spec_bench.py --model ~/huggingface/Qwen3-0.6B-FP8 \
        --quantization auto --spec-tokens 4 --cg

注：v1 的 verify 是单序列实现，所以本脚本固定 ``max_num_seqs=1``；
多序列（每序列一棵树）留到 v2。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import Engine, EngineConfig, SamplingParams  # noqa: E402

_QUANT_CHOICES = ("none", "fp8", "int8", "auto")


def _run_one(
    model: str,
    prompt: str,
    *,
    spec_tokens: int,
    spec_draft: str,
    temperature: float,
    n: int,
    warmup: int,
    reps: int,
    devices: list[int],
    tp: int,
    quantization: str,
    cg: bool,
    tie_eps: float = 0.0,
    exact_kv: bool = False,
    draft_model: str = "",
    num_blocks: int = 64,
    verify_graph: bool = False,
) -> dict:
    import torch

    eng = Engine(
        EngineConfig(
            model=model,
            device_ids=devices[:tp],
            tp_size=tp,
            quantization=quantization,  # type: ignore[arg-type]
            enforce_eager=not cg,
            max_num_seqs=1,
            context_len=2048,
            num_kvcache_blocks=num_blocks,
            gpu_memory_utilization=0.85,
            prefix_backend="none",
            spec_tokens=spec_tokens,
            spec_draft=spec_draft,  # type: ignore[arg-type]
            spec_tie_eps=tie_eps,
            spec_exact_kv=exact_kv,
            spec_draft_model=draft_model,
            spec_verify_graph=verify_graph,
        )
    )
    try:
        sp_w = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
        for _ in range(warmup):
            eng.add_request(prompt, sp_w)
            while eng.has_unfinished():
                eng.step()

        e2es, ttfts, outs = [], [], []
        sp = SamplingParams(temperature=temperature, max_tokens=n, ignore_eos=True)
        for _ in range(reps):
            torch.cuda.reset_peak_memory_stats()
            seq = eng.add_request(prompt, sp)
            t0 = time.perf_counter()
            first = None
            while eng.has_unfinished():
                eng.step()
                if first is None and seq.num_completion_tokens >= 1:
                    first = time.perf_counter()
            t1 = time.perf_counter()
            e2es.append(t1 - t0)
            ttfts.append((first - t0) * 1e3 if first else float("nan"))
            outs.append(list(seq.token_ids[seq.num_prompt_tokens :]))

        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        stats = dict(eng.scheduler.spec_stats)
        pos_rates = eng.scheduler.spec_position_rates()
        e2e = statistics.mean(e2es)
        ttft = statistics.mean(ttfts)
        decode_s = max(e2e - ttft / 1e3, 1e-6)
        steps = max(stats["steps"], 1)
        return {
            "spec_tokens": spec_tokens,
            "temperature": temperature,
            "ttft_ms": ttft,
            "decode_tok_s": (n - 1) / decode_s if n > 1 else float("nan"),
            "e2e_tok_s": n / e2e,
            "peak_mem_gb": peak_gb,
            "spec_tie_eps": tie_eps,
            "spec_exact_kv": exact_kv,
            "steps": stats["steps"],
            "proposed": stats["proposed"],
            "accepted": stats["accepted"],
            "accept_rate": (
                stats["accepted"] / stats["proposed"] if stats["proposed"] else float("nan")
            ),
            "accept_len": (stats["accepted"] + stats["steps"]) / steps,
            # 逐位置接受率：诊断"草稿在块内第几位开始失效"用，比 accept_len
            # 单个数字有信息量（上游 eval 也统计这个）
            "pos_rates": pos_rates,
            "tokens": outs[-1],
        }
    finally:
        eng.destroy()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _consistency(a: list[int], b: list[int]) -> float:
    if not a or not b:
        return float("nan")
    n = max(len(a), len(b))
    same = sum(1 for i in range(n) if i < len(a) and i < len(b) and a[i] == b[i])
    return same / n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="模型目录")
    p.add_argument("--prompt", default="List the first ten prime numbers, separated by commas:")
    p.add_argument("--prompt-file", default="", help="从文件读 prompt（覆盖 --prompt）")
    p.add_argument("--spec-tokens", type=int, default=4, help="草稿长度 k")
    p.add_argument("--spec-draft", default="ngram", choices=("ngram", "dspark"))
    p.add_argument(
        "--spec-draft-model", default="", help="DSpark 草稿目录（--spec-draft dspark 必填）"
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--n", type=int, default=64)
    p.add_argument(
        "--num-kvcache-blocks",
        type=int,
        default=64,
        help="KV 块数。FP8 目标 + BF16 草稿约 19G，24G 卡保持 64；放不下就调小",
    )
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--devices", default="0")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--quantization", default="none", choices=_QUANT_CHOICES)
    p.add_argument("--cg", action="store_true", help="启用 decode CUDA Graph")
    p.add_argument(
        "--spec-verify-graph",
        action="store_true",
        help="把 verify 也录进 CUDA Graph（按 m=2..k+1 分档；录图失败自动回退）",
    )
    p.add_argument(
        "--spec-tie-eps",
        type=float,
        default=0.0,
        help="emit 位置近并列时用 decode 内核重算（部分修复，0=关闭）",
    )
    p.add_argument(
        "--spec-exact-kv",
        action="store_true",
        help="bit-exact 模式：逐位置用 decode 内核重写 KV（每步多 accepted+1 次"
        "前向；greedy 下与 baseline 逐位一致，但会吃掉大部分加速）",
    )
    p.add_argument("--json", default="", help="把结果写成 JSON 文件")
    args = p.parse_args()

    if not Path(args.model).is_dir():
        raise SystemExit(f"missing model: {args.model}")
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    devices = [int(x) for x in args.devices.split(",") if x.strip()]

    common = dict(
        temperature=args.temperature,
        n=args.n,
        warmup=args.warmup,
        reps=args.reps,
        devices=devices,
        tp=args.tp,
        quantization=args.quantization,
        cg=args.cg,
        spec_draft=args.spec_draft,
        tie_eps=args.spec_tie_eps,
        exact_kv=args.spec_exact_kv,
        draft_model=args.spec_draft_model,
        num_blocks=args.num_kvcache_blocks,
        verify_graph=args.spec_verify_graph,
    )
    print(f"[run] baseline spec=0 model={args.model} quant={args.quantization}")
    # baseline 不需要守卫：spec_tokens=0 时它们无意义，显式置 0 以免 config 校验误判
    base = _run_one(
        args.model,
        prompt,
        spec_tokens=0,
        **{**common, "tie_eps": 0.0, "exact_kv": False, "verify_graph": False},
    )
    print(
        f"[run] spec k={args.spec_tokens} draft={args.spec_draft} "
        f"tie_eps={args.spec_tie_eps} exact_kv={args.spec_exact_kv}"
    )
    spec = _run_one(args.model, prompt, spec_tokens=args.spec_tokens, **common)

    # 口径：spec 相对 baseline 的倍数（>1 才是加速）。写成 base/spec 会把 3.4×
    # 印成 0.29×，直接反过来误判"投机没有收益"。
    speedup = spec["decode_tok_s"] / base["decode_tok_s"]
    match = _consistency(base["tokens"], spec["tokens"])

    rows = [
        ("spec=0", base),
        (f"spec={args.spec_tokens}", spec),
    ]
    print()
    print(f"{'config':>10} {'ttft_ms':>9} {'decode_tok/s':>13} {'e2e_tok/s':>10} "
          f"{'peak_GB':>8} {'accept_rate':>12} {'accept_len':>11}")
    for name, r in rows:
        ar = "—" if r["proposed"] == 0 else f"{r['accept_rate']:.1%}"
        print(
            f"{name:>10} {r['ttft_ms']:>9.2f} {r['decode_tok_s']:>13.1f} "
            f"{r['e2e_tok_s']:>10.1f} {r['peak_mem_gb']:>8.2f} "
            f"{ar:>12} {r['accept_len']:>11.2f}"
        )
    print()
    print(f"decode 加速比: {speedup:.3f}x（k={args.spec_tokens}，上限 {args.spec_tokens + 1}x）")
    rates = spec.get("pos_rates") or []
    if rates:
        print("逐位置接受率：", "  ".join(
            f"@{i} {'—' if r is None else f'{r:.3f}'}" for i, r in enumerate(rates)
        ))

    if args.temperature <= 1e-5:
        print(f"greedy 逐 token 一致率: {match:.1%}" + ("" if match == 1.0 else "  ← 必须 100%，否则是 bug"))
    else:
        print("温度采样：一致率无意义（只影响分布，不影响吞吐口径）")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {"baseline": base, "spec": spec, "speedup": speedup, "consistency": match},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"JSON 已写入 {args.json}")


if __name__ == "__main__":
    main()
