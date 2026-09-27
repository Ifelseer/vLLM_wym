"""vLLM_wym 多机启动入口（torchrun 拉起，每卡一进程）。

用法（单机 2 卡 TP，机内 NVLink）:
    torchrun --nproc_per_node=2 run_torchrun.py --tp 2 --pp 1 --prompt "Hello"

用法（2 节点 × 2 卡 = TP 2 × PP 2，PP 跨机）:
    # 两个节点分别执行（IP 换成本机可达地址）:
    torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 \
        --master_addr=10.0.0.1 --master_port=29500 run_torchrun.py --tp 2 --pp 2
    torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 \
        --master_addr=10.0.0.1 --master_port=29500 run_torchrun.py --tp 2 --pp 2

rank 布局：global rank = pp_rank * tp_size + tp_rank；rank 0 兼调度器（Scheduler）。
本入口只服务 **TP×PP**（WORLD = tp×pp）。**DP 请用 `run.py --dp`**（MPClient 多 Engine），不要与 torchrun 混用。
控制面（调度广播 + 采样回收）走 gloo；计算面（TP all-reduce / PP 中间张量）走 NCCL。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import Engine, EngineConfig, SamplingParams  # noqa: E402
from src.compute.worker import Worker  # noqa: E402
from src.distributed import set_nccl_env  # noqa: E402
from src.control.executor import MultiprocExecutor, worker_loop  # noqa: E402

DEFAULTS = {
    "model": "Qwen3-0.6B",
    "context_len": 4096,
    "max_num_seqs": 256,
    "max_num_batched_tokens": 8192,
    "kvcache_block_size": 256,
    "prefix_backend": "hash",
    "gpu_memory_utilization": 0.85,
    "num_kvcache_blocks": 0,
    "dtype": "auto",
    "enforce_eager": True,
    "torch_compile": False,
    "compile_mode": "reduce-overhead",
    "compile_dynamic": True,
    "pp_layer_counts": None,
    "device_ids": None,
    "nccl_port": 29500,
    "master_addr": "127.0.0.1",
    "master_port": 29500,
    "nccl_ifname": "",
    "enable_ep": False,
    "moe_backend": "auto",
}


def _load_config(args: argparse.Namespace) -> EngineConfig:
    cfg = dict(DEFAULTS)
    cfg_path = Path(args.config) if args.config else ROOT / "config.yaml"
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            for k, v in (yaml.safe_load(f) or {}).items():
                if k in cfg:
                    cfg[k] = v
    # CLI 开关覆盖 config：--cg 开图、--compile 开编译。
    if args.cg:
        cfg["enforce_eager"] = False
    if args.compile:
        cfg["torch_compile"] = True

    from model_hub import resolve_model_path  # noqa: E402

    model = resolve_model_path(args.model or cfg["model"])

    pp_layer_counts = cfg["pp_layer_counts"]
    if args.pp_layer_counts is not None:
        pp_layer_counts = [int(x) for x in args.pp_layer_counts.split(",")]
    device_ids = cfg["device_ids"]
    if args.device_ids is not None:
        device_ids = [int(x) for x in args.device_ids.split(",")]

    return EngineConfig(
        model=str(model),
        tp_size=args.tp,
        pp_size=args.pp,
        enforce_eager=cfg["enforce_eager"],
        torch_compile=cfg["torch_compile"],
        compile_mode=cfg["compile_mode"],
        compile_dynamic=cfg["compile_dynamic"],
        pp_layer_counts=pp_layer_counts,
        device_ids=device_ids,
        dtype=cfg["dtype"],
        context_len=cfg["context_len"],
        max_num_seqs=cfg["max_num_seqs"],
        max_num_batched_tokens=cfg["max_num_batched_tokens"],
        kvcache_block_size=cfg["kvcache_block_size"],
        prefix_backend=cfg["prefix_backend"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        num_kvcache_blocks=cfg["num_kvcache_blocks"],
        nccl_port=cfg["nccl_port"],
        master_addr=cfg["master_addr"],
        master_port=cfg["master_port"],
        nccl_ifname=cfg["nccl_ifname"],
        enable_ep=bool(args.ep or cfg.get("enable_ep", False)),
        moe_backend=args.moe_backend or cfg.get("moe_backend", "auto"),
    )


def _run_driver(config: EngineConfig, worker: Worker, args: argparse.Namespace) -> None:
    """rank0：调度器 + 广播/gather 控制面 + 输出。"""
    executor = MultiprocExecutor(config, torch.device("cuda:0"), worker.dtype, worker)
    engine = Engine(config, executor=executor)
    try:
        prompt = args.prompt or "Hello, my name is"
        token_ids = engine.tokenizer.encode(prompt)
        sp = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        seq = engine.add_request(token_ids, sp)
        while not seq.is_finished:
            engine.step()
        text = engine.tokenizer.decode(seq.token_ids[seq.num_prompt_tokens:])
        print(f"Prompt: {prompt}")
        print(f"Output: {prompt}{text}")
    finally:
        engine.destroy()
        executor.shutdown()  # 通知各 worker 退出


def main() -> None:
    p = argparse.ArgumentParser(description="vLLM_wym torchrun 多机多卡入口")
    p.add_argument("--config", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument(
        "--cg",
        action="store_true",
        help="启用 decode piecewise CUDA Graph（覆盖 config）",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="启用 prefill torch.compile（覆盖 config）",
    )
    p.add_argument(
        "--pp-layer-counts",
        type=str,
        default=None,
        help="不均分 PP 每 stage 层数（如 12,28）",
    )
    p.add_argument(
        "--device-ids",
        type=str,
        default=None,
        help="逗号分隔设备映射（覆盖 config）",
    )
    p.add_argument(
        "--ep",
        action="store_true",
        help="MoE expert parallel：在现有 TP ranks 上切 expert",
    )
    p.add_argument(
        "--moe-backend",
        choices=("auto", "cutlass", "triton"),
        default=None,
    )
    args = p.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    config = _load_config(args)
    tp, pp = config.tp_size, config.pp_size
    if tp * pp != world_size:
        raise SystemExit(
            f"tp({tp}) * pp({pp}) = {tp * pp} != WORLD_SIZE({world_size}). "
            f"请用 torchrun --nproc_per_node 等参数对齐。"
        )
    tp_rank = rank % tp
    pp_rank = rank // tp

    set_nccl_env(config.nccl_ifname)

    worker = Worker(config, rank, tp_rank, pp_rank, local_rank=local_rank)
    worker.init_environment(init_method="env://")
    worker.initialize()

    try:
        if rank == 0:
            _run_driver(config, worker, args)
        else:
            worker_loop(worker)
    finally:
        worker.destroy_environment()


if __name__ == "__main__":
    main()
