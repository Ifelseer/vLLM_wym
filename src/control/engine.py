from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import replace

import torch
from transformers import AutoTokenizer

from ..config import EngineConfig
from ..data import Sequence, SequenceStatus
from ..sampling_params import SamplingParams
from .client import InprocClient, make_client


# ==============================================================================
# Engine：推理引擎的对外门面。
#
# 职责：
#   - 加载 tokenizer，处理请求的编码、校验与截断。
#   - 通过 EngineCoreClient（InprocClient 或 MPClient）驱动底层推理。
#   - 维护门面侧的请求镜像 _seqs，用于流式输出和最终文本解码。
#
# Engine 本身不持有 EngineCore，也不感知 TP / PP / DP 的具体展开方式。
# 所有并行策略的分支都下沉到 make_client 和 EngineCore 内部。
# ==============================================================================


class Engine:
    """
        LLM 推理引擎门面。

        对外提供四个核心操作：
            add_request：编码并校验请求，加入调度队列。
            step：执行一步调度 → 前向 → 采样。
            generate_batch：批量同步生成，阻塞直到所有请求完成。
            generate：单条流式生成，逐 token yield。

        底层执行路径由 make_client 决定：
            dp=1：InprocClient，同进程直接调用。
            dp>1：MPClient，经 ZMQ 路由到多个副本子进程。
    """

    def __init__(self, config: EngineConfig, executor=None) -> None:
        self.config = config
        self.device = torch.device("cuda:0")
        self.dtype = self._resolve_dtype()

        # 混合 GPU 主机：按本 Engine 实际用到的卡锁定 FlashInfer arch，
        # 避免默认扫到 Blackwell 后在 A100/3090 上 NoKernelImageForDevice。
        device_ids = config.device_ids or list(range(config.world_size))
        from ..utils.cuda_env import apply_flashinfer_arch_list

        apply_flashinfer_arch_list(device_ids, overwrite=True)

        # 禁用 BF16 matmul 的低精度 reduction，保证累加精度。
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        self.tokenizer = self._load_tokenizer()
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        self.context_len = config.context_len
        self.block_size = config.block_size

        # 构造底层客户端。
        # executor 参数仅在 torchrun 多机路径下使用：
        # 外部已初始化好 rank0 的 MultiprocExecutor，直接注入，避免重复创建。
        self._client = make_client(
            config,
            eos_token_id=self.eos_token_id,
            device=self.device,
            dtype=self.dtype,
            executor=executor,
        )

        # 门面侧的请求镜像。
        # dp=1 时，这里的 Sequence 与调度器内是同一对象（共享引用），
        #   调度器原地更新，门面侧无需额外同步。
        # dp>1 时，调度器在子进程里，门面侧需要根据 step 返回的增量输出
        #   手动推进这些镜像对象。
        self._seqs: dict[int, Sequence] = {}

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #

    def _resolve_dtype(self) -> torch.dtype:
        """根据配置解析计算精度，默认回退到模型自带的 torch_dtype。"""
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)
        dt = getattr(self.config.hf_config, "torch_dtype", None)
        return dt if isinstance(dt, torch.dtype) else torch.float16

    def _load_tokenizer(self):
        """加载 tokenizer，若缺少 pad_token 则用 eos_token 补位。"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @property
    def scheduler(self):
        """
            暴露底层调度器，仅用于调试和测试。
            只在 dp=1 时可用，dp>1 时调度器在子进程内，无法直接访问。
        """
        if not isinstance(self._client, InprocClient):
            raise NotImplementedError(
                "scheduler is only reachable for the in-process (dp=1) engine"
            )
        return self._client.engine_core.scheduler

    @property
    def last_num_cached_tokens(self) -> int:
        """上一步命中前缀缓存的 token 数。dp>1 时暂不支持，返回 0。"""
        if not isinstance(self._client, InprocClient):
            return 0
        return self._client.engine_core.last_num_cached_tokens

    def reset_kv(self, prefix_backend: str | None = None) -> None:
        """
            清空 KV cache 并重建调度器。
            可选切换前缀缓存后端（"none" / "hash" / "radix"）。
            仅支持 dp=1。
        """
        if not isinstance(self._client, InprocClient):
            raise NotImplementedError("reset_kv is only supported for dp=1")
        if prefix_backend is not None and prefix_backend not in ("none", "hash", "radix"):
            raise ValueError(f"invalid prefix_backend: {prefix_backend!r}")
        core = self._client.engine_core
        core.kv = core.executor.reset_kv(prefix_backend)
        core.scheduler = core._build_scheduler()
        core.last_num_cached_tokens = 0
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 请求入队
    # ------------------------------------------------------------------ #

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
    ) -> Sequence:
        """
            编码 prompt，校验长度，创建 Sequence 并提交到调度队列。

            若 prompt 超过 context_len，截断保留末尾（保留最近的上下文）。
            若 max_tokens + prompt 长度超过 context_len，压缩 max_tokens。

            Returns:
                已入队的 Sequence 对象。
                dp=1 时该对象与调度器内部共享同一引用，可直接读取生成进度。
        """
        sp = sampling_params or SamplingParams()
        token_ids = (
            list(prompt) if isinstance(prompt, list) else self.tokenizer.encode(prompt)
        )
        if not token_ids:
            raise ValueError("empty prompt")

        # prompt 超长：截断，保留末尾部分。
        if len(token_ids) > self.context_len:
            warnings.warn(
                f"prompt length {len(token_ids)} > context_len={self.context_len}; "
                f"truncating to last {self.context_len} tokens",
                stacklevel=2,
            )
            token_ids = token_ids[-self.context_len :]

        # 生成预算校验：确保 prompt + 生成不超过上下文窗口。
        max_ok = max(0, self.context_len - len(token_ids))
        if sp.max_tokens > max_ok:
            warnings.warn(
                f"max_tokens={sp.max_tokens} + prompt={len(token_ids)} exceeds "
                f"context_len={self.context_len}; clamping max_tokens to {max_ok}",
                stacklevel=2,
            )
            sp = replace(sp, max_tokens=max_ok)

        seq = Sequence.from_prompt(token_ids, block_size=self.block_size, sampling=sp)
        self._seqs[seq.seq_id] = seq
        self._client.add_request(seq)
        return seq

    # ------------------------------------------------------------------ #
    # 执行
    # ------------------------------------------------------------------ #

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        """
            驱动一步推理：调度 → 前向 → 采样 → 后处理。

            Returns:
                (finished, metric)
                - finished：本步完成的请求列表，每项为 (seq_id, completion_token_ids)。
                - metric：本步处理的 token 数（用于吞吐统计）。
        """
        outputs, metric = self._client.step()

        # dp>1 时，门面侧的 Sequence 是独立镜像，需要根据返回的增量手动推进。
        # dp=1 时，Sequence 与调度器共享同一对象，已由底层原地更新，这里跳过。
        if not isinstance(self._client, InprocClient):
            for out in outputs:
                seq = self._seqs.get(out.seq_id)
                if seq is None:
                    continue
                for tid in out.new_token_ids:
                    seq.append_token(tid)
                if out.is_finished:
                    seq.status = SequenceStatus.FINISHED

        # 收集本步完成的请求，提取生成部分的 token ids。
        finished: list[tuple[int, list[int]]] = []
        for out in outputs:
            if out.is_finished:
                seq = self._seqs.get(out.seq_id)
                if seq is not None:
                    finished.append(
                        (out.seq_id, seq.token_ids[seq.num_prompt_tokens :])
                    )
        return finished, metric

    def generate_batch(
        self,
        prompts: list[str | list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
    ) -> list[str]:
        """
        批量同步生成。

        将所有 prompt 一次性入队，循环 step 直到全部完成，
        返回与输入顺序对应的生成文本列表。
        """
        sps = self._normalize_sampling_params(prompts, sampling_params)
        seqs = [self.add_request(p, sp) for p, sp in zip(prompts, sps)]
        finished_text: dict[int, str] = {}
        while self.has_unfinished():
            outputs, _ = self.step()
            for seq_id, completion_ids in outputs:
                finished_text[seq_id] = self.tokenizer.decode(completion_ids)
        return [finished_text[seq.seq_id] for seq in seqs]

    def generate(
        self,
        prompt: str | list[int],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
    ) -> Iterator[str]:
        """
            单条流式生成。

            每步检查是否有新 token 产出，有则立即 yield 解码后的文本片段。
            调用方可用 for 循环逐段消费。
        """
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_new_tokens,
        )
        seq = self.add_request(prompt, sp)
        emitted = 0
        while not seq.is_finished:
            self.step()
            # 将本步新产出的 token 逐个解码并 yield。
            while emitted < seq.num_completion_tokens:
                tid = seq.token_ids[seq.num_prompt_tokens + emitted]
                emitted += 1
                yield self.tokenizer.decode([tid])

    def has_unfinished(self) -> bool:
        """是否还有未完成的请求在调度队列中。"""
        return self._client.has_unfinished()

    @staticmethod
    def _normalize_sampling_params(
        prompts: list[str | list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None,
    ) -> list[SamplingParams]:
        """
            将采样参数统一为与 prompts 等长的列表。
            支持三种输入：None（全部默认）、单个（广播）、列表（逐一对应）。
        """
        if sampling_params is None:
            return [SamplingParams() for _ in prompts]
        if isinstance(sampling_params, SamplingParams):
            return [sampling_params for _ in prompts]
        if len(sampling_params) != len(prompts):
            raise ValueError("sampling_params length must match prompts")
        return list(sampling_params)

    def destroy(self) -> None:
        """释放引擎资源：关闭底层 client，清空请求镜像。"""
        self._client.destroy()
        self._seqs.clear()