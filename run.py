"""vLLM_wym 统一入口 — FlashAttention + CUDA Graph + Continuous Batching + MoE EP。

用法:
    python run.py                                    # 单条推理（使用 config.yaml）
    python run.py --eager                            # 强制 eager（关闭 Decode CUDA Graph）
    python run.py --cg                               # 启用 Decode CUDA Graph
    python run.py --compile                          # 启用 prefill torch.compile
    python run.py --no-compile                       # 关闭 prefill torch.compile
    python run.py --batch                            # 批量推理演示
    python run.py --model ~/huggingface/Qwen1.5-MoE-A2.7B --tp 2 --ep
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.run_utils import (  # noqa: E402
    cuda_sync,
    drain_engine,
    handle_hub_cli,
    load_config,
    merge_cli,
    print_bench_report,
    warmup_engine,
)

from src import Engine, EngineConfig, SamplingParams  # noqa: E402
from model_hub import resolve_model_path  # noqa: E402

DEFAULTS = {
    "model": "Qwen3-0.6B",
    "prompt": "Hello, my name is",
    "max_new_tokens": 200,
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 20,
    "enforce_eager": True,
    "torch_compile": False,
    "compile_mode": "reduce-overhead",
    "compile_dynamic": True,
    "dtype": "auto",
    "context_len": 4096,
    "max_num_seqs": 256,
    "max_num_batched_tokens": 8192,
    "mix_prefill_decode": True,
    "kvcache_block_size": 256,
    "prefix_backend": "hash",
    "gpu_memory_utilization": 0.9,
    "num_kvcache_blocks": 0,
    "tp_size": 1,
    "pp_size": 1,
    "dp_size": 1,
    "pp_layer_counts": None,
    "device_ids": None,
    "nccl_port": 29500,
    "master_addr": "127.0.0.1",
    "master_port": 29500,
    "nccl_ifname": "",
    "enable_ep": False,
    "moe_backend": "auto",
    "quantization": "none",
    "quant_kv_cache": False,
    "spec_tokens": 0,
    "spec_draft": "ngram",
    "spec_tie_eps": 0.0,
}


def _make_engine(cfg: dict) -> Engine:
    model_path = resolve_model_path(cfg["model"])
    print(f"Loading {model_path} ...")
    return Engine(
        EngineConfig(
            model=str(model_path),
            enforce_eager=cfg["enforce_eager"],
            torch_compile=cfg["torch_compile"],
            compile_mode=cfg["compile_mode"],
            compile_dynamic=cfg["compile_dynamic"],
            dtype=cfg["dtype"],
            context_len=cfg["context_len"],
            max_num_seqs=cfg["max_num_seqs"],
            max_num_batched_tokens=cfg["max_num_batched_tokens"],
            mix_prefill_decode=cfg["mix_prefill_decode"],
            kvcache_block_size=cfg["kvcache_block_size"],
            prefix_backend=cfg["prefix_backend"],
            gpu_memory_utilization=cfg["gpu_memory_utilization"],
            num_kvcache_blocks=cfg["num_kvcache_blocks"],
            tp_size=cfg["tp_size"],
            pp_size=cfg["pp_size"],
            dp_size=cfg["dp_size"],
            pp_layer_counts=cfg["pp_layer_counts"],
            device_ids=cfg["device_ids"],
            nccl_port=cfg["nccl_port"],
            master_addr=cfg["master_addr"],
            master_port=cfg["master_port"],
            nccl_ifname=cfg["nccl_ifname"],
            enable_ep=bool(cfg.get("enable_ep", False)),
            moe_backend=cfg.get("moe_backend", "auto"),
            quantization=cfg.get("quantization", "none"),
            quant_kv_cache=bool(cfg.get("quant_kv_cache", False)),
            spec_tokens=int(cfg.get("spec_tokens", 0)),
            spec_draft=cfg.get("spec_draft", "ngram"),
            spec_tie_eps=float(cfg.get("spec_tie_eps", 0.0)),
        )
    )


def _warmup(engine: Engine, prompt_lens: list[int]) -> float:
    warm_sp = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
    return warmup_engine(
        prompt_lens,
        add_request=engine.add_request,
        drain=lambda: drain_engine(engine.has_unfinished, engine.step),
        sync=cuda_sync,
        sampling_params=warm_sp,
    )


def run_single(cfg: dict) -> None:
    """单条推理：等长 dummy 先编译再录图，第三条起才计时。"""
    engine = _make_engine(cfg)
    prompt = cfg["prompt"]
    token_ids = (
        list(prompt) if isinstance(prompt, list) else engine.tokenizer.encode(prompt)
    )
    warmup_s = _warmup(engine, [len(token_ids)])

    sp = SamplingParams(
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
        max_tokens=cfg["max_new_tokens"],
    )
    seq = engine.add_request(token_ids, sp)

    cuda_sync()
    t0 = time.perf_counter()
    t_first = None
    while not seq.is_finished:
        engine.step()
        if t_first is None and seq.num_completion_tokens > 0:
            cuda_sync()
            t_first = time.perf_counter()
    cuda_sync()
    t_end = time.perf_counter()

    completion = seq.token_ids[seq.num_prompt_tokens :]
    text = engine.tokenizer.decode(completion)
    print(f"Prompt: {prompt}")
    print(f"{prompt}{text}")
    print_bench_report(
        (t_first - t0) if t_first is not None else None,
        len(completion),
        t_end - t0,
        decode_s=(t_end - t_first) if t_first is not None else None,
        warmup_s=warmup_s,
    )


def run_batch(cfg: dict) -> None:
    """批量推理：按各 prompt 等长 dummy 预热后计时。"""
    engine = _make_engine(cfg)
    prompts = [
        "Hello, my name is",
        "Once upon a time",
        "The capital of France is",
        "In machine learning,",
    ]
    encoded = [engine.tokenizer.encode(p) for p in prompts]
    warmup_s = _warmup(engine, [len(ids) for ids in encoded])

    sp = SamplingParams(temperature=0.0, max_tokens=cfg["max_new_tokens"])
    seqs = [engine.add_request(ids, sp) for ids in encoded]

    cuda_sync()
    t0 = time.perf_counter()
    t_first = None
    while engine.has_unfinished():
        engine.step()
        if t_first is None and any(s.num_completion_tokens > 0 for s in seqs):
            cuda_sync()
            t_first = time.perf_counter()
    cuda_sync()
    t_end = time.perf_counter()

    n = 0
    for p, seq in zip(prompts, seqs):
        ids = seq.token_ids[seq.num_prompt_tokens :]
        n += len(ids)
        print(f"\n>>> {p}\n{engine.tokenizer.decode(ids)}")
    print_bench_report(
        (t_first - t0) if t_first is not None else None,
        n,
        t_end - t0,
        decode_s=(t_end - t_first) if t_first is not None else None,
        warmup_s=warmup_s,
    )
    print(f"Batch:  {len(prompts)} reqs, {n} completion tokens")


def main() -> None:
    p = argparse.ArgumentParser(
        description="vLLM_wym FlashAttention + CUDA Graph + Continuous Batching + MoE EP"
    )
    p.add_argument("--config", default=None, help="YAML 配置文件路径")
    p.add_argument("--model", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    cg_group = p.add_mutually_exclusive_group()
    cg_group.add_argument("--cg", action="store_true", help="启用 Decode CUDA Graph")
    cg_group.add_argument(
        "--eager", action="store_true", help="强制 eager 模式（关闭 Decode CUDA Graph）"
    )
    compile_group = p.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--compile", action="store_true", help="启用 prefill torch.compile"
    )
    compile_group.add_argument(
        "--no-compile", action="store_true", help="关闭 prefill torch.compile"
    )
    p.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"),
                   default=None)
    p.add_argument("--compile-dynamic", action="store_true",
                   help="符号形状（一次编译覆盖任意长度，避免按长度重编译）")
    p.add_argument("--no-compile-dynamic", action="store_true",
                   help="固定形状重放（仅离线固定长度；新长度会重编译）")
    p.add_argument("--prefix-backend", choices=("none", "hash", "radix"), default=None)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--tp", type=int, default=None, help="张量并行度（机内多卡）")
    p.add_argument("--pp", type=int, default=None, help="流水线并行度（跨机/多卡）")
    p.add_argument("--dp", type=int, default=None, help="数据并行副本数（多卡吞吐）")
    p.add_argument(
        "--ep",
        action="store_true",
        help="MoE expert parallel：在现有 TP ranks 上切 expert，不扩大 world",
    )
    p.add_argument(
        "--moe-backend",
        choices=("auto", "cutlass", "triton"),
        default=None,
        help="fused MoE 后端（auto=SM89+ cutlass，否则 Triton）",
    )
    p.add_argument(
        "--quantization",
        choices=("none", "auto", "fp8", "int8", "awq", "gptq"),
        default=None,
        help=(
            "量化方式：none=不量化；auto=按模型自带 quantization_config；"
            "fp8/int8=W8A8（BF16 模型时加载时在线量化）；"
            "awq/gptq=INT4 weight-only（需预量化 checkpoint）"
        ),
    )
    p.add_argument(
        "--quant-kv-cache",
        action="store_true",
        help="FP8 KV cache（仅 fp8 路径 + SM89+；自动切 FlashInfer attention）",
    )
    p.add_argument(
        "--spec-tokens",
        type=int,
        default=None,
        help="投机解码草稿长度 k（0=关闭；开启后每步 verify 写入 1+k 个位置；"
        "v1 要求 max_num_seqs=1）",
    )
    p.add_argument(
        "--spec-draft",
        choices=("ngram", "dspark"),
        default=None,
        help="草稿来源：ngram=后缀匹配（无模型）/ dspark=DSpark 草稿头（Phase 3）",
    )
    p.add_argument(
        "--spec-tie-eps",
        type=float,
        default=None,
        help="greedy tie 守卫阈值（0=关闭）。verify 与 decode 用不同 attention "
        "内核，精确并列处 argmax 可能翻转；>0 时在 top1-top2 < eps 的位置用 "
        "decode 内核重算，保证与 baseline 逐位一致",
    )
    p.add_argument(
        "--pp-layer-counts",
        type=str,
        default=None,
        help="不均分 PP 每 stage 层数（逗号分隔，如 12,28）",
    )
    p.add_argument(
        "--device-ids",
        type=str,
        default=None,
        help="逗号分隔 GPU 映射（如 1,0）；长度应为 dp×tp×pp",
    )
    p.add_argument("--batch", action="store_true", help="运行批量推理演示")
    p.add_argument("--download", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if handle_hub_cli(args):
        return

    config_path = Path(args.config) if args.config else HERE / "config.yaml"
    cfg = load_config(config_path, DEFAULTS)
    merge_cli(
        cfg,
        args,
        ("model", "prompt", "max_new_tokens", "temperature", "top_p", "top_k", "prefix_backend", "max_num_seqs"),
    )
    if args.tp is not None:
        cfg["tp_size"] = args.tp
    if args.pp is not None:
        cfg["pp_size"] = args.pp
    if args.dp is not None:
        cfg["dp_size"] = args.dp
    if args.pp_layer_counts is not None:
        cfg["pp_layer_counts"] = [int(x) for x in args.pp_layer_counts.split(",")]
    if args.device_ids is not None:
        cfg["device_ids"] = [int(x) for x in args.device_ids.split(",")]
    if args.cg:
        cfg["enforce_eager"] = False
    if args.eager:
        cfg["enforce_eager"] = True
    if args.compile:
        cfg["torch_compile"] = True
    if args.no_compile:
        cfg["torch_compile"] = False
    if args.compile_mode:
        cfg["compile_mode"] = args.compile_mode
    if args.compile_dynamic:
        cfg["compile_dynamic"] = True
    if args.no_compile_dynamic:
        cfg["compile_dynamic"] = False
    if args.ep:
        cfg["enable_ep"] = True
    if args.moe_backend is not None:
        cfg["moe_backend"] = args.moe_backend
    if args.quantization is not None:
        cfg["quantization"] = args.quantization
    if args.quant_kv_cache:
        cfg["quant_kv_cache"] = True
    if args.spec_tokens is not None:
        cfg["spec_tokens"] = args.spec_tokens
    if args.spec_draft is not None:
        cfg["spec_draft"] = args.spec_draft
    if args.spec_tie_eps is not None:
        cfg["spec_tie_eps"] = args.spec_tie_eps

    if args.batch:
        run_batch(cfg)
    else:
        run_single(cfg)


if __name__ == "__main__":
    main()
