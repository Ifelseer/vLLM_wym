from __future__ import annotations

import copy
import gc
import io
from contextlib import redirect_stdout

import torch

from ..config import EngineConfig
from ..data import SchedulerOutput
from ..distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_pp_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_expert_parallel,
    set_nccl_env,
    set_pp_layer_ranges,
)
from .kv_pool import KVPool
from .model_runner import ModelRunner
from .models.loader import default_dtype_context, load_model, skip_param_init
from .models.qwen3 import Qwen3ForCausalLM
from .sampler import Sampler


class Worker:
    def __init__(
        self,
        config: EngineConfig,
        rank: int,
        tp_rank: int,
        pp_rank: int,
        local_rank: int | None = None,
    ):
        self.config = config
        self.rank = rank
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.tp_size = config.tp_size
        self.pp_size = config.pp_size
        self.world_size = config.world_size
        # local_rank 是机内 GPU 下标：单机 spawn 时等于 rank；多机 torchrun 时为 LOCAL_RANK。
        self.local_rank = rank if local_rank is None else local_rank
        self.device = torch.device(f"cuda:{self._device_id()}")
        self.dtype = self._resolve_dtype()
        self.sampler: Sampler | None = None
        self.kv: KVPool | None = None
        self.model: torch.nn.Module | None = None
        self.is_last_pp = False
        self.model_runner: ModelRunner | None = None
        self._mem_baseline = 0

    def init_environment(self, init_method: str | None = None) -> None:
        from ..utils.cuda_env import apply_flashinfer_arch_list

        apply_flashinfer_arch_list([self._device_id()], overwrite=True)
        # 单机 spawn 默认走 lo；跨机在 config.nccl_ifname 填 ib0/eth0。
        ifname = self.config.nccl_ifname or "lo"
        set_nccl_env(ifname, disable_ib=not bool(self.config.nccl_ifname))
        # 单机（loopback）用文件 rendezvous，不依赖 rank0 先起 TCPStore 的时序。
        if init_method is None and self.config.master_addr in ("", "127.0.0.1", "localhost"):
            init_method = (
                f"file:///tmp/vllm_wym_rdv_{self.config.nccl_port}"
                f"_{self.config.tp_size}_{self.config.pp_size}"
            )
        else:
            init_method = init_method or (
                f"tcp://{self.config.master_addr}:{self.config.nccl_port}"
            )
        init_distributed_environment(
            world_size=self.world_size,
            rank=self.rank,
            local_rank=self._device_id(),
            backend="nccl",
            init_method=init_method,
        )
        initialize_model_parallel(
            self.tp_size,
            self.pp_size,
            self.tp_rank,
            self.pp_rank,
        )
        set_expert_parallel(self.config.enable_ep and self.config.is_moe)
        self.is_last_pp = get_pp_group().is_last_rank
        # 不均分 PP：按 config.pp_layer_counts 覆盖等分层区间（如 12/28 适配 12G+24G 异卡）。
        if self.config.pp_layer_counts is not None:
            counts = self.config.pp_layer_counts
            assert len(counts) == self.pp_size, "pp_layer_counts 长度须等于 pp_size"
            assert sum(counts) == self.config.hf_config.num_hidden_layers, (
                "pp_layer_counts 之和须等于模型层数"
            )
            set_pp_layer_ranges(counts)

    def _device_id(self) -> int:
        # device_ids 显式映射（如单卡调试 [0,0]）优先，否则用机内 local_rank。
        if self.config.device_ids is not None:
            return self.config.device_ids[self.rank]
        return self.local_rank

    def initialize(self) -> None:
        model = self._load_model()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.model = model
        self.sampler = Sampler()
        self._mem_baseline = torch.cuda.memory_reserved()
        self.kv = self._make_kv()
        self.model_runner = ModelRunner(
            model, self.kv, self.sampler, config=self.config, device=self.device
        )

    def _resolve_dtype(self) -> torch.dtype:
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)
        dt = getattr(self.config.hf_config, "torch_dtype", None)
        return dt if isinstance(dt, torch.dtype) else torch.float16

    def _load_model(self) -> torch.nn.Module:
        from .models.qwen3_moe import Qwen3MoeForCausalLM
        from ..moe.parallel import MoEParallelConfig, resolve_moe_backend

        with default_dtype_context(self.dtype):
            with skip_param_init():
                if self.config.is_moe:
                    backend = resolve_moe_backend(self.config.moe_backend)
                    parallel = MoEParallelConfig.from_engine(
                        tp_size=self.tp_size,
                        tp_rank=self.tp_rank,
                        enable_ep=self.config.enable_ep,
                        fused_backend=backend,
                    )
                    model = Qwen3MoeForCausalLM(
                        self.config.hf_config,
                        parallel=parallel,
                        quant=self.config.quant_config,
                    )
                else:
                    model = Qwen3ForCausalLM(
                        self.config.hf_config, quant=self.config.quant_config
                    )
        load_model(model, self.config.model)
        return model.to(dtype=self.dtype, device=self.device).eval()

    def _probe_cg_footprint(self) -> int:
        probe_config = copy.copy(self.config)
        probe_config.torch_compile = False
        kv = KVPool(
            self.config.hf_config,
            num_blocks=8,
            block_size=self.config.block_size,
            device=self.device,
            dtype=self.dtype,
        )
        with redirect_stdout(io.StringIO()):
            runner = ModelRunner(
                self.model, kv, self.sampler, config=probe_config, device=self.device
            )
        torch.cuda.synchronize()
        footprint = torch.cuda.memory_reserved() - self._mem_baseline
        runner.destroy()
        del runner, kv
        gc.collect()
        torch.cuda.empty_cache()
        return int(footprint)

    def _make_kv(self) -> KVPool:
        use_cg = not self.config.enforce_eager
        if self.config.num_kvcache_blocks > 0:
            num_blocks = self.config.num_kvcache_blocks
        else:
            reserve = self._probe_cg_footprint() if use_cg else 0
            num_blocks = KVPool.auto_num_blocks(
                self.config.hf_config,
                block_size=self.config.block_size,
                dtype=self.dtype,
                device=self.device,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                reserve_bytes=reserve,
            )
            self.config.num_kvcache_blocks = num_blocks
            if use_cg:
                print(
                    f"CG footprint measured: {reserve / 2**30:.3f} GiB; "
                    f"KV pool sized to remainder ({num_blocks} blocks)"
                )
        return KVPool(
            self.config.hf_config,
            num_blocks=num_blocks,
            block_size=self.config.block_size,
            device=self.device,
            dtype=self.dtype,
        )

    def get_kv_cache_size(self) -> int:
        assert self.kv is not None, "initialize() must be called first"
        return self.kv.num_blocks_total

    def reset_kv(self, prefix_backend: str | None = None) -> None:
        """重建 GPU KV 池与 ModelRunner。多卡 driver 广播后每 rank 都要走这里。"""
        if self.model_runner is not None:
            self.model_runner.destroy()
            self.model_runner = None
        if prefix_backend is not None:
            self.config.prefix_backend = prefix_backend
        assert self.model is not None and self.sampler is not None
        self.kv = self._make_kv()
        self.model_runner = ModelRunner(
            self.model, self.kv, self.sampler, config=self.config, device=self.device
        )

    def execute_model(self, sched: SchedulerOutput) -> tuple[int, int] | None:
        """只发射前向 + 异步 D2H，不等待采样结果。"""
        assert self.model_runner is not None
        result = self.model_runner.run_forward(sched)
        if not self.is_last_pp:
            return None
        return result

    def read_samples(self, n_p: int, n_d: int) -> tuple[list[int], list[int]]:
        assert self.model_runner is not None
        return self.model_runner.read_samples(n_p, n_d)

    def destroy_environment(self) -> None:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass
        if self.model_runner is not None:
            self.model_runner.destroy()
        self.model_runner = None
        self.kv = None
        self.model = None
        set_pp_layer_ranges(None)
        destroy_model_parallel()
        destroy_distributed_environment()
