"""MoE / dense end-to-end latency on this engine only."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import Engine, EngineConfig, SamplingParams  # noqa: E402
from tests.helpers.paths import DENSE_MODEL, MOE_MODEL  # noqa: E402


def _measure(eng: Engine, prompt: str, *, n: int, warmup: int, reps: int) -> dict:
    sp_w = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
    for _ in range(warmup):
        seq = eng.add_request(prompt, sp_w)
        while not seq.is_finished:
            eng.step()
    e2es, ttfts = [], []
    last: list[int] = []
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
    e2e = statistics.mean(e2es)
    ttft = statistics.mean(ttfts)
    decode_s = max(e2e - ttft / 1e3, 1e-6)
    return {
        "ttft_ms": ttft,
        "e2e_s": e2e,
        "e2e_tok_s": n / e2e,
        "decode_tok_s": (n - 1) / decode_s,
        "tokens": last,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("dense", "moe"), default="moe")
    p.add_argument("--devices", default="1", help="comma CUDA indices (visible order)")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--ep", action="store_true")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--moe-backend", default="triton")
    p.add_argument("--nccl-port", type=int, default=29500)
    cg = p.add_mutually_exclusive_group()
    cg.add_argument(
        "--cg",
        action="store_true",
        help="enable Decode CUDA Graph (enforce_eager=False)",
    )
    cg.add_argument(
        "--eager",
        action="store_true",
        help="force eager (default if neither --cg nor --eager)",
    )
    args = p.parse_args()
    # Default remains eager for historical bench scripts; pass --cg to measure Graph.
    use_cg = bool(args.cg) and not bool(args.eager)

    devices = [int(x) for x in args.devices.split(",") if x.strip() != ""]
    model = str(DENSE_MODEL if args.model == "dense" else MOE_MODEL)
    if not Path(model).is_dir():
        raise SystemExit(f"missing model: {model}")

    eng = Engine(
        EngineConfig(
            model=model,
            device_ids=devices[: max(args.tp, 1)],
            tp_size=args.tp,
            enable_ep=bool(args.ep),
            enforce_eager=not use_cg,
            max_num_seqs=1,
            context_len=2048,
            num_kvcache_blocks=64,
            gpu_memory_utilization=0.85,
            prefix_backend="none",
            moe_backend=args.moe_backend,  # type: ignore[arg-type]
            nccl_port=int(args.nccl_port),
        )
    )
    try:
        r = _measure(eng, args.prompt, n=args.n, warmup=args.warmup, reps=args.reps)
    finally:
        eng.destroy()

    mode = "cg" if use_cg else "eager"
    print(
        f"model={args.model} tp={args.tp} ep={args.ep} mode={mode} "
        f"devices={devices} n={args.n}\n"
        f"  ttft_ms={r['ttft_ms']:.2f}  decode_tok/s={r['decode_tok_s']:.2f}  "
        f"e2e_tok/s={r['e2e_tok_s']:.2f}\n"
        f"  tokens={r['tokens'][:16]}{'...' if len(r['tokens']) > 16 else ''}"
    )


if __name__ == "__main__":
    main()
