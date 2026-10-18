"""量化 bench：同一 prompt 下对比 none / fp8 / int8 / int4(auto) 的
TTFT / decode 吞吐 / 端到端吞吐 / 峰值显存 / greedy token 一致率。

口径（与 ``bench/vs_vllm.py`` 的 ``_ours`` 一致，便于横向比较）：

- **TTFT**：从 add_request 到首个 completion token 的墙钟时间。
- **decode_tok_s**：``(n-1) / (e2e - ttft)``。
- **峰值显存**：``torch.cuda.max_memory_allocated``（引擎生命周期内的峰值，
  含权重 + 激活 + KV；每轮 reset 后单独统计）。
- **贪心一致率**：与第一轮（基线，通常 ``none``）逐 token 比对的一致比例。
  temperature=0 下这是量化正确性的最强信号——格式错乱会直接掉到接近 0。

两种用法::

    # 在线量化：同一 BF16 模型，多档量化（none 作基线）
    python bench/quant_bench.py --model ~/huggingface/Qwen3-0.6B \
        --quantizations none,fp8,int8

    # 预量化 checkpoint：与 BF16 兄弟模型对比（一致率的参照）
    python bench/quant_bench.py --model ~/huggingface/Qwen3-0.6B \
        --baseline none,fp8 --quantized-model ~/huggingface/Qwen3-0.6B-FP8

注意：每轮量化配置在引擎构造时确定，因此多档量化是「顺序构造/销毁引擎」，
轮间 ``torch.cuda.empty_cache()`` + ``reset_peak_memory_stats`` 隔离显存统计。
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

# 量化档位 → EngineConfig.quantization 取值
_QUANT_CHOICES = ("none", "fp8", "int8", "auto")


def _run_one(
    model: str,
    prompt: str,
    *,
    quantization: str,
    n: int,
    warmup: int,
    reps: int,
    devices: list[int],
    tp: int,
    enable_ep: bool,
    moe_backend: str,
    quant_kv_cache: bool,
    cg: bool,
) -> dict:
    import torch

    eng = Engine(
        EngineConfig(
            model=model,
            device_ids=devices[:tp],
            tp_size=tp,
            enable_ep=enable_ep,
            quantization=quantization,  # type: ignore[arg-type]
            quant_kv_cache=quant_kv_cache,
            enforce_eager=not cg,
            max_num_seqs=1,
            context_len=2048,
            num_kvcache_blocks=64,
            gpu_memory_utilization=0.85,
            prefix_backend="none",
            moe_backend=moe_backend,  # type: ignore[arg-type]
        )
    )
    try:
        sp_w = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
        for _ in range(warmup):
            seq = eng.add_request(prompt, sp_w)
            while not seq.is_finished:
                eng.step()

        e2es, ttfts, last = [], [], []
        sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        for _ in range(reps):
            seq = eng.add_request(prompt, sp)
            t0 = time.perf_counter()
            first = None
            while not seq.is_finished:
                eng.step()
                if first is None and seq.num_completion_tokens >= 1:
                    first = time.perf_counter()
            t1 = time.perf_counter()
            e2es.append(t1 - t0)
            ttfts.append((first - t0) * 1e3 if first else float("nan"))
            last = list(seq.token_ids[seq.num_prompt_tokens :])

        torch.cuda.reset_peak_memory_stats()
        seq = eng.add_request(prompt, sp)
        while not seq.is_finished:
            eng.step()
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

        e2e = statistics.mean(e2es)
        ttft = statistics.mean(ttfts)
        decode_s = max(e2e - ttft / 1e3, 1e-6)
        return {
            "quant": quantization,
            "ttft_ms": ttft,
            "decode_tok_s": (n - 1) / decode_s if n > 1 else float("nan"),
            "e2e_tok_s": n / e2e,
            "peak_mem_gb": peak_gb,
            "tokens": last,
        }
    finally:
        eng.destroy()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _consistency(a: list[int], b: list[int]) -> float:
    """逐 token 一致率（长度不一致时按较长者计缺失为不一致）。"""
    if not a or not b:
        return float("nan")
    n = max(len(a), len(b))
    same = sum(1 for i in range(n) if i < len(a) and i < len(b) and a[i] == b[i])
    return same / n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="模型目录（或注册表名）")
    p.add_argument(
        "--quantizations",
        default="none,fp8,int8",
        help="逗号分隔的量化档位（none/fp8/int8/auto）",
    )
    p.add_argument(
        "--quantized-model",
        default="",
        help="预量化 checkpoint 目录；给了它就只跑 auto 档并与基线对比",
    )
    p.add_argument("--baseline-model", default="", help="BF16 基线目录（默认同 --model）")
    p.add_argument("--devices", default="1")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--ep", action="store_true")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--moe-backend", default="triton")
    p.add_argument("--quant-kv-cache", action="store_true")
    p.add_argument("--cg", action="store_true", help="启用 decode CUDA Graph")
    p.add_argument("--json", default="", help="把结果写成 JSON 文件")
    args = p.parse_args()

    devices = [int(x) for x in args.devices.split(",") if x.strip() != ""]
    if not Path(args.model).is_dir():
        raise SystemExit(f"missing model: {args.model}")

    results: list[dict] = []
    if args.quantized_model:
        # 预量化模式：基线（BF16 兄弟）+ auto 档
        baseline_model = args.baseline_model or args.model
        for model, q in ((baseline_model, "none"), (args.quantized_model, "auto")):
            if not Path(model).is_dir():
                raise SystemExit(f"missing model: {model}")
            print(f"[run] quant={q:<5} model={model}")
            results.append(
                _run_one(
                    model,
                    args.prompt,
                    quantization=q,
                    n=args.n,
                    warmup=args.warmup,
                    reps=args.reps,
                    devices=devices,
                    tp=args.tp,
                    enable_ep=bool(args.ep),
                    moe_backend=args.moe_backend,
                    quant_kv_cache=args.quant_kv_cache,
                    cg=args.cg,
                )
            )
    else:
        for q in [x.strip() for x in args.quantizations.split(",") if x.strip()]:
            if q not in _QUANT_CHOICES:
                raise SystemExit(f"invalid quantization {q!r}; expected {_QUANT_CHOICES}")
            print(f"[run] quant={q:<5} model={args.model}")
            results.append(
                _run_one(
                    args.model,
                    args.prompt,
                    quantization=q,
                    n=args.n,
                    warmup=args.warmup,
                    reps=args.reps,
                    devices=devices,
                    tp=args.tp,
                    enable_ep=bool(args.ep),
                    moe_backend=args.moe_backend,
                    quant_kv_cache=args.quant_kv_cache,
                    cg=args.cg,
                )
            )

    baseline = results[0]
    print()
    header = (
        f"{'quant':<7}{'TTFT ms':>10}{'decode tok/s':>14}{'e2e tok/s':>12}"
        f"{'peak GB':>10}{'mem vs base':>12}{'token match':>13}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        match = _consistency(baseline["tokens"], r["tokens"])
        mem_ratio = r["peak_mem_gb"] / baseline["peak_mem_gb"]
        print(
            f"{r['quant']:<7}{r['ttft_ms']:>10.1f}{r['decode_tok_s']:>14.1f}"
            f"{r['e2e_tok_s']:>12.1f}{r['peak_mem_gb']:>10.2f}"
            f"{mem_ratio:>11.2f}x{match:>12.1%}"
        )
    print("\n口径：temperature=0 贪心；token match = 与首行（基线）逐 token 一致率")
    print("     TTFT 从 add_request 到首个 completion token；peak 为引擎生命周期峰值")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "model": args.model,
                    "prompt": args.prompt,
                    "n": args.n,
                    "reps": args.reps,
                    "cg": args.cg,
                    "results": [
                        {k: v for k, v in r.items() if k != "tokens"} for r in results
                    ],
                    "consistency": [
                        _consistency(baseline["tokens"], r["tokens"]) for r in results
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n[saved] {args.json}")


if __name__ == "__main__":
    main()
