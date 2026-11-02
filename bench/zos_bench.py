"""ZOS 开/关对照：decode 步时的分位数 + 吞吐 + 逐 token 一致性 + 流水线计数。

## 为什么单独一个 bench

`spec_bench` 比的是"投机 vs 不投机"，`zos_bench` 比的是**同一配置下"调度时序"**，
只切换 `enable_zos`。而且 ZOS 的收益主要体现在**单步耗时**上（把 host 工作藏进 GPU
计算），吞吐只是它的结果——所以这里必须报 `decode_step_ms` 的分位数，光看 tok/s 会
被测量窗口稀释掉。

## 口径

- **decode_step_ms**：进入 decode 阶段后，每次 `eng.step()` 的墙钟耗时；报 p50/p99。
  ZOS 的收益上限就是"p50 下降多少"，p99 反而可能变差（drain 那些拍更慢）。
- **decode_tok_s**：`(n-1) × bs / (e2e - ttft)`，与 `spec_bench` 同口径。
- **一致性**：`--temperature 0` 下开/关两组必须**逐 token 相同**；这是硬 gate。
- **zos_counters**：`lookahead_submitted` 为 0 说明每拍都被 drain，等于没测到。

用法::

    # 一次跑完（同进程；显存紧张时用 --run 拆开）
    python bench/zos_bench.py --model ~/huggingface/Qwen3-0.6B --cg --n 128

    # 拆分进程（推荐，避免两份模型权重叠加）
    python bench/zos_bench.py --model ... --cg --run sync --json zos_sync.json
    python bench/zos_bench.py --model ... --cg --run zos  --json zos_on.json
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import Engine, EngineConfig, SamplingParams  # noqa: E402

_QUANT_CHOICES = ("none", "fp8", "int8", "auto")


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def _run_one(
    model: str,
    prompt: str,
    *,
    zos: bool,
    n: int,
    warmup: int,
    reps: int,
    max_num_seqs: int,
    quantization: str,
    cg: bool,
    devices: list[int],
    num_blocks: int,
    temperature: float,
) -> dict:
    import torch

    eng = Engine(
        EngineConfig(
            model=model,
            device_ids=devices,
            enforce_eager=not cg,
            max_num_seqs=max_num_seqs,
            context_len=2048,
            num_kvcache_blocks=num_blocks,
            quantization=quantization,  # type: ignore[arg-type]
            prefix_backend="none",
            enable_zos=zos,
        )
    )
    try:
        sp_w = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
        for _ in range(warmup):
            eng.add_request(prompt, sp_w)
            while eng.has_unfinished():
                eng.step()

        e2es, ttfts, step_ms, outs = [], [], [], []
        sp = SamplingParams(temperature=temperature, max_tokens=n, ignore_eos=True)
        for _ in range(reps):
            torch.cuda.reset_peak_memory_stats()
            seqs = [eng.add_request(prompt, sp) for _ in range(max_num_seqs)]
            t0 = time.perf_counter()
            first = None
            while eng.has_unfinished():
                ts = time.perf_counter()
                eng.step()
                dt = (time.perf_counter() - ts) * 1e3
                if first is None and any(s.num_completion_tokens >= 1 for s in seqs):
                    first = time.perf_counter()
                elif first is not None:
                    # 只统计 decode 阶段（首个 token 之后）
                    step_ms.append(dt)
            t1 = time.perf_counter()
            e2es.append(t1 - t0)
            ttfts.append((first - t0) * 1e3 if first else float("nan"))
            outs.append([list(s.token_ids[s.num_prompt_tokens :]) for s in seqs])

        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        e2e = statistics.mean(e2es)
        ttft = statistics.mean(ttfts)
        decode_s = max(e2e - ttft / 1e3, 1e-6)
        return {
            "zos": zos,
            "temperature": temperature,
            "max_num_seqs": max_num_seqs,
            "ttft_ms": ttft,
            "decode_step_ms_p50": _pct(step_ms, 0.50),
            "decode_step_ms_p99": _pct(step_ms, 0.99),
            "decode_steps": len(step_ms),
            "decode_tok_s": (n - 1) * max_num_seqs / decode_s if n > 1 else float("nan"),
            "e2e_tok_s": n * max_num_seqs / e2e,
            "peak_mem_gb": peak_gb,
            "tokens": outs[-1][0],
            "tokens_all": outs[-1],
            "zos_stats": eng.zos_stats(),
        }
    finally:
        eng.destroy()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="Count down from 300 by sevens: 300, 293, 286,")
    p.add_argument("--prompt-file", default="")
    p.add_argument("--n", type=int, default=128, help="每条请求生成多少 token")
    p.add_argument("--max-num-seqs", type=int, default=1, help="并发请求数")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--devices", default="0")
    p.add_argument("--quantization", default="none", choices=_QUANT_CHOICES)
    p.add_argument("--cg", action="store_true", help="启用 decode CUDA Graph")
    p.add_argument(
        "--num-kvcache-blocks", type=int, default=64, help="显存紧张时调小"
    )
    p.add_argument(
        "--run",
        choices=("both", "sync", "zos"),
        default="both",
        help="both=同进程跑两组；sync/zos=只跑一组（推荐拆进程，避免两份权重叠加）",
    )
    p.add_argument("--json", default="")
    args = p.parse_args()

    prompt = (
        Path(args.prompt_file).read_text(encoding="utf-8").strip()
        if args.prompt_file
        else args.prompt
    )
    common = dict(
        n=args.n,
        warmup=args.warmup,
        reps=args.reps,
        max_num_seqs=args.max_num_seqs,
        quantization=args.quantization,
        cg=args.cg,
        devices=[int(d) for d in args.devices.split(",") if d != ""],
        num_blocks=args.num_kvcache_blocks,
        temperature=args.temperature,
    )

    base = (
        _run_one(args.model, prompt, zos=False, **common)
        if args.run in ("both", "sync")
        else None
    )
    on = (
        _run_one(args.model, prompt, zos=True, **common)
        if args.run in ("both", "zos")
        else None
    )

    def show(r: dict, label: str) -> None:
        st = r["zos_stats"]
        print(
            f"{label:6s} zos={'on ' if r['zos'] else 'off'}  "
            f"step p50={r['decode_step_ms_p50']:7.2f} ms  "
            f"p99={r['decode_step_ms_p99']:7.2f} ms  "
            f"decode={r['decode_tok_s']:7.1f} tok/s  "
            f"e2e={r['e2e_tok_s']:7.1f}  "
            f"peak={r['peak_mem_gb']:.2f} GB"
        )
        if st:
            print(
                f"       counters: lookahead={st.get('lookahead_submitted', 0)} "
                f"drains={st.get('drains', 0)} "
                f"(waiting={st.get('waiting_drains', 0)} "
                f"batch={st.get('batch_drains', 0)} "
                f"terminal={st.get('terminal_drains', 0)} "
                f"memory={st.get('memory_drains', 0)}) "
                f"fences={st.get('eos_fences', 0) + st.get('prefix_fences', 0)}"
            )

    print()
    if base:
        show(base, "sync")
    if on:
        show(on, "zos")
    print()

    if base and on:
        if args.temperature <= 1e-5:
            same = base["tokens"] == on["tokens"]
            print(f"greedy 逐 token 一致: {'100%' if same else '不一致 ← 必须排查'}")
            if not same:
                diff = next(
                    (
                        i
                        for i, (a, b) in enumerate(zip(base["tokens"], on["tokens"]))
                        if a != b
                    ),
                    None,
                )
                print(f"  first diff at {diff}")
                print(f"  sync={base['tokens'][:32]}")
                print(f"  zos ={on['tokens'][:32]}")
        p50_gain = (
            (base["decode_step_ms_p50"] - on["decode_step_ms_p50"])
            / base["decode_step_ms_p50"]
            * 100
            if base["decode_step_ms_p50"]
            else float("nan")
        )
        print(
            f"decode 步时 p50 降低 {p50_gain:.1f}%  "
            f"({base['decode_step_ms_p50']:.2f} → {on['decode_step_ms_p50']:.2f} ms)"
        )
        print(
            f"decode 吞吐 {base['decode_tok_s']:.1f} → {on['decode_tok_s']:.1f} tok/s "
            f"（{on['decode_tok_s'] / base['decode_tok_s']:.3f}x）"
        )
        if on["zos_stats"].get("lookahead_submitted", 0) == 0:
            print(
                "⚠️ lookahead_submitted=0：流水线从未提前提交（每拍都被 drain），"
                "本次对照无法证明 ZOS 生效"
            )

    if args.json:
        Path(args.json).write_text(
            json.dumps({"sync": base, "zos": on}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nJSON 已写入 {args.json}")


if __name__ == "__main__":
    main()
