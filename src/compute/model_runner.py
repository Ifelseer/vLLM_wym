from __future__ import annotations

import torch

from ..config import EngineConfig
from ..data import SchedulerOutput, SeqSchedule
from ..distributed import (
    IntermediateTensors,
    get_pp_group,
    get_tp_group,
    get_world_group,
    recv_tensors,
    send_tensors,
)
from .batch import Batch
from .cuda_graph import CUDAGraphRunner
from .kv_pool import KVPool
from .layers.attention import AttnInputs, AttnMetadata, attention_context
from .sampler import Sampler


class ModelRunner:
    def __init__(
        self,
        model,
        kv: KVPool,
        sampler: Sampler,
        *,
        config: EngineConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.kv = kv
        self.sampler = sampler.to(device=device)
        self.config = config
        self.device = device
        self.block_size = kv.block_size
        self.kv_cache = kv.kv_cache
        self.dtype = kv.dtype
        self.hidden_size = config.hf_config.hidden_size
        self._meta = AttnMetadata(kv_cache=self.kv_cache)

        pp = get_pp_group()
        tp = get_tp_group()
        self._pp_group = pp
        self.is_first_pp = pp.is_first_rank
        self.is_last_pp = pp.is_last_rank
        self.is_driver = self.is_last_pp and tp.group_rank == 0

        max_bs = min(config.max_num_seqs, 256)
        max_blocks = (config.context_len + self.block_size - 1) // self.block_size
        self.max_bs = max_bs
        self.batch = Batch(
            max_bs=max_bs,
            max_tokens=max(config.max_num_batched_tokens, max_bs),
            max_blocks=max_blocks,
            vocab_size=config.hf_config.vocab_size,
            device=device,
            block_size=self.block_size,
        )

        # piecewise CUDA Graph（对齐 vLLM #35162）：
        #   图外 recv → 图内 forward（读写静态 buffer）→ 图外 send
        # 与「整段 recv+层+send 录进一张图」相比：P2P 更稳，调度可与图解耦。
        self._pp_recv = self._make_pp_buffer(max_bs) if not self.is_first_pp else None
        self._pp_send = self._make_pp_buffer(max_bs) if not self.is_last_pp else None
        self._decode_out = (
            torch.zeros(max_bs, self.hidden_size, dtype=self.dtype, device=device)
            if self.is_last_pp
            else None
        )

        self._copy_stream = torch.cuda.Stream(device=device)
        self._copy_event = torch.cuda.Event()
        self._sample_cpu = torch.empty(max_bs * 2, dtype=torch.int64, pin_memory=True)

        self.cuda_graph_runner: CUDAGraphRunner | None = None
        if not config.enforce_eager:
            # capture 路径含 TP all-reduce / PP 边界，各 rank 先对齐再录，
            # 避免先到的 rank 卡在 warmup 的 NCCL 集合通信。
            world = get_world_group()
            if world.size > 1:
                world.barrier()
            self.cuda_graph_runner = CUDAGraphRunner(
                CUDAGraphRunner.make_bs_list(max_bs), device
            )
            self.cuda_graph_runner.capture(self._decode_body)
            if world.size > 1:
                world.barrier()

        self.prefill_fn = None
        if config.torch_compile:
            self.prefill_fn = self._build_compiled_prefill()

    def _make_pp_buffer(self, n: int) -> IntermediateTensors:
        return IntermediateTensors(
            hidden_states=torch.empty(
                n, self.hidden_size, dtype=self.dtype, device=self.device
            ),
            residual=torch.empty(
                n, self.hidden_size, dtype=self.dtype, device=self.device
            ),
        )

    def _pp_dst(self) -> int:
        return self._pp_group.next_group_rank

    def _pp_src(self) -> int:
        return self._pp_group.prev_group_rank

    def _send_intermediate(self, out: IntermediateTensors) -> None:
        send_tensors(
            (out["hidden_states"], out["residual"]),
            self._pp_dst(),
            self._pp_group,
        )

    def _recv_intermediate(self, n: int, *, static: bool = False) -> IntermediateTensors:
        buf = self._pp_recv if static else self._make_pp_buffer(n)
        h = buf["hidden_states"][:n]
        r = buf["residual"][:n]
        recv_tensors((h, r), self._pp_src(), self._pp_group)
        return IntermediateTensors(hidden_states=h, residual=r)

    @torch.inference_mode()
    def run_forward(self, sched: SchedulerOutput):
        """一步 sync 前向：decode 再 prefill；PP 为 lockstep 切层，非 1F1B。

        非末 PP stage 只做本段计算 + P2P，返回 None；末 stage 异步 D2H 采样 id。
        """
        d = self._run_decode(sched.decodes) if sched.decodes else None
        p = self._run_prefill(sched.prefills) if sched.prefills else None
        if not self.is_last_pp:
            return None
        return self._stage_samples(p, d)

    def _decode_attn(self, bs: int) -> AttnInputs:
        return AttnInputs(
            slot_mapping=self.batch.slot_mapping[:bs],
            is_prefill=False,
            cache_seqlens=self.batch.cache_seqlens[:bs],
            block_tables=self.batch.block_tables[:bs],
        )

    def _decode_body(self, bs: int):
        """仅本地 compute；PP recv/send 在 replay 前后执行。"""
        it = None
        if not self.is_first_pp:
            assert self._pp_recv is not None
            it = IntermediateTensors(
                hidden_states=self._pp_recv["hidden_states"][:bs],
                residual=self._pp_recv["residual"][:bs],
            )
        with attention_context(self._meta):
            out = self.model(
                self.batch.input_ids[:bs],
                self.batch.positions[:bs],
                self._decode_attn(bs),
                it,
            )
        if self.is_last_pp:
            self._decode_out[:bs] = out
            return self._decode_out[:bs]
        assert self._pp_send is not None
        self._pp_send["hidden_states"][:bs].copy_(out["hidden_states"])
        self._pp_send["residual"][:bs].copy_(out["residual"])
        return out

    def _run_decode(self, seqs: list[SeqSchedule]):
        if not seqs:
            return None
        bs = len(seqs)
        pad_to = (
            self.cuda_graph_runner._pad_bs(bs)
            if self.cuda_graph_runner is not None and bs <= self.max_bs
            else None
        )
        self.batch.build_decode(seqs, pad_to=pad_to)

        if self.cuda_graph_runner is not None and bs <= self.max_bs:
            if not self.is_first_pp:
                self._recv_intermediate(bs, static=True)
            hidden, _ = self.cuda_graph_runner.replay(bs)
            if not self.is_last_pp:
                self._send_intermediate(
                    IntermediateTensors(
                        hidden_states=self._pp_send["hidden_states"][:bs],
                        residual=self._pp_send["residual"][:bs],
                    )
                )
                return None
            hidden = hidden[:bs]
        else:
            it = self._recv_intermediate(bs) if not self.is_first_pp else None
            with attention_context(self._meta):
                hidden = self.model(
                    self.batch.input_ids[:bs],
                    self.batch.positions[:bs],
                    self._decode_attn(bs),
                    it,
                )
            if not self.is_last_pp:
                self._send_intermediate(hidden)
                return None

        return self._sample_logits(hidden, bs)

    def _prefill_attn(
        self,
        bs: int,
        n_tokens: int,
        max_q: int,
        max_k: int,
        has_cache: bool,
    ) -> AttnInputs:
        return AttnInputs(
            slot_mapping=self.batch.slot_mapping[:n_tokens],
            is_prefill=True,
            cu_seqlens_q=self.batch.cu_seqlens_q[: bs + 1],
            cu_seqlens_k=self.batch.cu_seqlens_k[: bs + 1],
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            block_tables=self.batch.block_tables[:bs] if has_cache else None,
        )

    def _run_prefill(self, seqs: list[SeqSchedule]):
        if not seqs:
            return None
        n_tokens, max_q, max_k, has_cache = self.batch.build_prefill(seqs)
        bs = len(seqs)
        it = self._recv_intermediate(n_tokens) if not self.is_first_pp else None

        if self.prefill_fn is not None and all(
            s.num_cached_tokens == 0 for s in seqs
        ):
            with attention_context(self._meta):
                hidden = self.prefill_fn(
                    self.batch.input_ids[:n_tokens],
                    self.batch.positions[:n_tokens],
                    self.batch.slot_mapping[:n_tokens],
                    self.batch.cu_seqlens_q[: bs + 1],
                    max_q,
                    it,
                )
        else:
            with attention_context(self._meta):
                hidden = self.model(
                    self.batch.input_ids[:n_tokens],
                    self.batch.positions[:n_tokens],
                    self._prefill_attn(bs, n_tokens, max_q, max_k, has_cache),
                    it,
                )

        if not self.is_last_pp:
            self._send_intermediate(hidden)
            return None
        if not any(s.need_sample for s in seqs):
            if not self.is_driver:
                return None
            return torch.zeros(bs, dtype=torch.int64, device=self.device)
        last = self.batch.cu_seqlens_q[1 : bs + 1] - 1
        return self._sample_logits(hidden[last], bs)

    def _sample_logits(self, hidden: torch.Tensor, bs: int) -> torch.Tensor | None:
        if self.batch.all_greedy:
            ids = self.model.argmax(hidden)
            if not self.is_driver:
                return None
            return ids
        logits = self.model.compute_logits(hidden)
        if not self.is_driver:
            return None
        return self._sample(bs, logits)

    def _sample(self, bs: int, logits: torch.Tensor) -> torch.Tensor:
        return self.sampler(
            logits,
            self.batch.temperatures[:bs],
            top_ps=self.batch.top_ps[:bs],
            top_ks=self.batch.top_ks[:bs],
            min_ps=self.batch.min_ps[:bs],
            all_greedy=self.batch.all_greedy,
            any_greedy=self.batch.any_greedy,
            use_top_k=self.batch.use_top_k,
            use_top_p=self.batch.use_top_p,
            use_min_p=self.batch.use_min_p,
            max_top_k=self.batch.max_top_k,
        )

    def _stage_samples(
        self, p: torch.Tensor | None, d: torch.Tensor | None
    ) -> tuple[int, int]:
        n_p = 0 if p is None else p.numel()
        n_d = 0 if d is None else d.numel()
        need = n_p + n_d
        if need > self._sample_cpu.numel():
            self._sample_cpu = torch.empty(need, dtype=torch.int64, pin_memory=True)
        cur = torch.cuda.current_stream()
        self._copy_stream.wait_stream(cur)
        with torch.cuda.stream(self._copy_stream):
            if d is not None:
                self._sample_cpu[:n_d].copy_(d, non_blocking=True)
            if p is not None:
                self._sample_cpu[n_d : n_d + n_p].copy_(p, non_blocking=True)
        self._copy_event.record(self._copy_stream)
        return n_p, n_d

    def read_samples(self, n_p: int, n_d: int) -> tuple[list[int], list[int]]:
        self._copy_event.synchronize()
        p_ids = self._sample_cpu[n_d : n_d + n_p].tolist() if n_p else []
        d_ids = self._sample_cpu[:n_d].tolist() if n_d else []
        return p_ids, d_ids

    def _resolve_compile_mode(self) -> str:
        """单卡 / 仅 PP 用用户 mode；含 TP 时用 max-autotune-no-cudagraphs。"""
        if get_tp_group().size == 1:
            return self.config.compile_mode
        return "max-autotune-no-cudagraphs"

    def _prefill_forward(
        self,
        input_ids,
        positions,
        slot_mapping,
        cu_seqlens,
        max_seqlen,
        intermediate_tensors,
    ):
        attn = AttnInputs(
            slot_mapping=slot_mapping,
            is_prefill=True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
        )
        return self.model(input_ids, positions, attn, intermediate_tensors)

    def _build_compiled_prefill(self):
        fn = torch.compile(
            self._prefill_forward,
            mode=self._resolve_compile_mode(),
            dynamic=self.config.compile_dynamic,
        )
        n = min(256, self.config.context_len)
        ids = torch.arange(1, n + 1, dtype=torch.int64, device=self.device)
        pos = torch.arange(n, dtype=torch.int64, device=self.device)
        cu = torch.tensor([0, n], dtype=torch.int32, device=self.device)
        sm = torch.arange(n, dtype=torch.int32, device=self.device)
        it = self._make_pp_buffer(n) if not self.is_first_pp else None
        with attention_context(self._meta), torch.inference_mode():
            fn(ids, pos, sm, cu, n, it)
        torch.cuda.synchronize()
        self.kv_cache.zero_()
        return fn

    def destroy(self) -> None:
        if self.cuda_graph_runner is not None:
            self.cuda_graph_runner.destroy()
            self.cuda_graph_runner = None
        self.prefill_fn = None
