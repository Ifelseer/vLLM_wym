"""Compare this engine vs official vLLM on the same greedy workload.

Official vLLM is loaded via a **separate** interpreter (``--vllm-python``) so it
does not fight this repo's Torch / FlashInfer stack.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import Engine, EngineConfig, SamplingParams  # noqa: E402
from tests.helpers.paths import DENSE_MODEL, MOE_MODEL  # noqa: E402

_VLLM_WORKER = r"""
import json, sys, time, statistics
model, prompt, n, warmup, reps, tp, max_model_len = sys.argv[1:8]
n, warmup, reps, tp, max_model_len = map(int, (n, warmup, reps, tp, max_model_len))
from vllm import LLM, SamplingParams
llm = LLM(
    model=model,
    tensor_parallel_size=tp,
    max_model_len=max_model_len,
    enforce_eager=True,
    disable_log_stats=True,
    gpu_memory_utilization=0.85,
)
sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
for _ in range(warmup):
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
e2es, ttfts, last = [], [], []
for _ in range(reps):
    t0 = time.perf_counter()
    outs = llm.generate([prompt], sp)
    t1 = time.perf_counter()
    # vLLM LLM.generate is opaque for true TTFT; approximate with e2e for n=1 warmup check.
    e2es.append(t1 - t0)
    last = list(outs[0].outputs[0].token_ids)
# Approximate TTFT: run max_tokens=1
sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
tt = []
for _ in range(reps):
    t0 = time.perf_counter()
    llm.generate([prompt], sp1)
    tt.append((time.perf_counter() - t0) * 1e3)
e2e = statistics.mean(e2es)
ttft = statistics.mean(tt)
decode_s = max(e2e - ttft / 1e3, 1e-6)
print(json.dumps({
    "engine": "vllm",
    "ttft_ms": ttft,
    "e2e_s": e2e,
    "e2e_tok_s": n / e2e,
    "decode_tok_s": (n - 1) / decode_s if n > 1 else float("nan"),
    "tokens": last,
}))
"""


def _ours(
    model: str,
    prompt: str,
    *,
    n: int,
    warmup: int,
    reps: int,
    devices: list[int],
    tp: int,
    enable_ep: bool,
    moe_backend: str,
) -> dict:
    eng = Engine(
        EngineConfig(
            model=model,
            device_ids=devices[:tp],
            tp_size=tp,
            enable_ep=enable_ep,
            enforce_eager=True,
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
        e2e = statistics.mean(e2es)
        ttft = statistics.mean(ttfts)
        decode_s = max(e2e - ttft / 1e3, 1e-6)
        return {
            "engine": "vLLM_wym",
            "ttft_ms": ttft,
            "e2e_s": e2e,
            "e2e_tok_s": n / e2e,
            "decode_tok_s": (n - 1) / decode_s,
            "tokens": last,
        }
    finally:
        eng.destroy()


def _vllm(
    python: str,
    model: str,
    prompt: str,
    *,
    n: int,
    warmup: int,
    reps: int,
    tp: int,
    max_model_len: int,
    env: dict,
) -> dict:
    cmd = [
        python,
        "-c",
        _VLLM_WORKER,
        model,
        prompt,
        str(n),
        str(warmup),
        str(reps),
        str(tp),
        str(max_model_len),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            "official vLLM worker failed.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}\n"
            "Install vLLM in a separate env and pass --vllm-python."
        )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"no JSON from vLLM worker\n{proc.stdout}\n{proc.stderr}")
    return json.loads(lines[-1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("dense", "moe"), default="dense")
    p.add_argument("--devices", default="1")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--ep", action="store_true", help="vLLM_wym only (official uses TP)")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--moe-backend", default="triton")
    p.add_argument(
        "--vllm-python",
        default="",
        help="Python with official `vllm` installed (separate venv recommended)",
    )
    p.add_argument("--skip-vllm", action="store_true")
    args = p.parse_args()

    devices = [int(x) for x in args.devices.split(",") if x.strip() != ""]
    model = str(DENSE_MODEL if args.model == "dense" else MOE_MODEL)
    if not Path(model).is_dir():
        raise SystemExit(f"missing model: {model}")

    print(f"== vLLM_wym ({args.model}) ==")
    ours = _ours(
        model,
        args.prompt,
        n=args.n,
        warmup=args.warmup,
        reps=args.reps,
        devices=devices,
        tp=args.tp,
        enable_ep=bool(args.ep),
        moe_backend=args.moe_backend,
    )
    print(json.dumps(ours, ensure_ascii=False))

    if args.skip_vllm:
        return
    vpy = args.vllm_python.strip()
    if not vpy:
        print(
            "\n[skip official vLLM] pass --vllm-python /path/to/venv/bin/python "
            "(see bench/README.md). Use --skip-vllm to silence this."
        )
        return

    import os

    env = os.environ.copy()
    # Map selected devices into the child as cuda:0..
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices[: args.tp])
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    print(f"\n== official vLLM ({args.model}) via {vpy} ==")
    theirs = _vllm(
        vpy,
        model,
        args.prompt,
        n=args.n,
        warmup=args.warmup,
        reps=args.reps,
        tp=args.tp,
        max_model_len=2048,
        env=env,
    )
    print(json.dumps(theirs, ensure_ascii=False))
    print("\n== ratio (vLLM_wym / vllm; >1 means vLLM_wym slower on that metric's inverse) ==")
    for k in ("ttft_ms", "decode_tok_s", "e2e_tok_s"):
        a, b = ours[k], theirs[k]
        if k.endswith("_s"):
            # throughput: higher better → ratio ours/theirs
            print(f"  {k}: vLLM_wym={a:.2f} vllm={b:.2f}  vLLM_wym/vllm={a / b:.3f}")
        else:
            # latency: lower better → ratio ours/theirs
            print(f"  {k}: vLLM_wym={a:.2f} vllm={b:.2f}  vLLM_wym/vllm={a / b:.3f}")
    print(f"  token_match={ours['tokens'] == theirs['tokens']}")


if __name__ == "__main__":
    main()
