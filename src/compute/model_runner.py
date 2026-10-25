from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..config import EngineConfig
from ..data import SchedulerOutput, SeqSchedule
from ..spec.verify import accept_greedy, accept_rejection
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


@dataclass
class _SpecSamples:
    """投机 decode 的结果：扁平 token 列表 + 每序列 token 数。

    ``lens[i] = 接受该序列的草稿数 + 1``（+1 是 anchor 位置产生的新 token）。
    """

    tokens: list[int] = field(default_factory=list)
    lens: list[int] = field(default_factory=list)
    #: 每序列**提出**的草稿数（worker 侧起草时调度器算不出来，只能回传）
    proposed: list[int] = field(default_factory=list)


class ModelRunner:
    def __init__(
        self,
        model,
        kv: KVPool,
        sampler: Sampler,
        *,
        config: EngineConfig,
        device: torch.device,
        num_qo_heads: int = 0,
        num_kv_heads: int = 0,
        head_dim: int = 0,
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
        # DSpark 的 aux hidden 缓冲（懒分配；关闭投机时恒为 None）
        self._aux_buf: torch.Tensor | None = None
        # DSpark 草稿器（懒加载草稿模型）与每个序列的待追加 context 段
        self._dspark = None
        self._dspark_pending: dict[int, torch.Tensor] = {}
        self._draft_probs: torch.Tensor | None = None
        # verify 的 CUDA Graph 分档（m -> graph）与静态输出缓冲
        self._verify_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._verify_out: torch.Tensor | None = None
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

        # FP8 KV cache（kv.kv_dtype == fp8 时启用；scale 校准与 FlashInfer
        # wrapper 见 compute/fp8_kv.py）
        self._fp8_kv = None
        if kv.kv_dtype == torch.float8_e4m3fn:
            from .fp8_kv import Fp8KVManager

            self._fp8_kv = Fp8KVManager(
                num_layers=self.kv_cache.shape[1],
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=self.block_size,
                max_batch_size=max_bs,
                max_blocks_per_seq=max_blocks,
                device=device,
                dtype=kv.dtype,
                use_cuda_graph=not config.enforce_eager,
            )
            self._fp8_kv.set_kv_cache(self.kv_cache)
            self._meta.fp8_kv = self._fp8_kv
            self._calibrate_fp8_kv()

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
        # 投机 decode 的同步结果（普通 decode 复用 _sample_cpu 的异步 D2H）
        self._spec: _SpecSamples | None = None

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
            # FP8 KV：录图期间 attention 走 CG wrapper（plan 已在 on_warmup 完成）
            if self._fp8_kv is not None:
                self._meta.fp8_kv_cg = True
            self.cuda_graph_runner.capture(
                self._decode_body,
                on_warmup=(self._fp8_kv_plan_warmup if self._fp8_kv is not None else None),
            )
            if self._fp8_kv is not None:
                self._meta.fp8_kv_cg = False
            if world.size > 1:
                world.barrier()

        self.prefill_fn = None
        if config.torch_compile:
            self.prefill_fn = self._build_compiled_prefill()

        # 投机：把 verify 的 M=1+k 形状预热掉（FP8 autotune / Triton 首次编译），
        # 否则第一个请求会吃到秒级尖刺。放在所有 warmup 之后执行。
        self._warmup_verify_shapes()
        # 可选：把 verify 本身也录图（默认关；失败自动回退 eager）
        self._capture_verify_graphs()

    def _make_pp_buffer(self, n: int) -> IntermediateTensors:
        return IntermediateTensors(
            hidden_states=torch.empty(
                n, self.hidden_size, dtype=self.dtype, device=self.device
            ),
            residual=torch.empty(
                n, self.hidden_size, dtype=self.dtype, device=self.device
            ),
        )

    # ------------------------------------------------------------------ #
    # FP8 KV cache：校准 / 预热 / plan
    # ------------------------------------------------------------------ #

    def _fake_decode_batch(self, bs: int) -> None:
        """构造 bs 行假 decode 批（cache_seqlens=1、页表指向 page 0）。

        slot_mapping 全部有效（=0）：warmup 前向会真实写入 page 0，用于触发
        各层 scale 校准；page 0 之后会被真实序列覆写，无副作用。
        """
        seqs = [
            SeqSchedule(seq_id=-1, block_table=[0], num_cached_tokens=0, token_ids=[1])
            for _ in range(bs)
        ]
        self.batch.build_decode(seqs)

    def _calibrate_fp8_kv(self) -> None:
        """录图前跑一次最小前向：校准各层 scale + 预热 FlashInfer 路径。

        必须在 CUDA Graph capture 之前完成——captured 区域不能再写 scale
        （replay 时会用新数据重算，导致存量 KV 的写/读 scale 不一致）。
        """
        self._fake_decode_batch(1)
        self._fp8_kv.plan_decode(
            1, self.batch.block_tables[:1], self.batch.cache_seqlens[:1], cg=False
        )
        with attention_context(self._meta):
            self.model(
                self.batch.input_ids[:1],
                self.batch.positions[:1],
                self._decode_attn(1),
                None,
            )
        torch.cuda.synchronize(self.device)

    def _fp8_kv_plan_warmup(self, bs: int) -> None:
        """capture 每个 bs 前的 warmup 钩子：plan 该 bs 的 FlashInfer wrapper。

        plan 内部有 D2H 拷贝，不能进图；在每个 bs 录图前执行一次，
        captured 区域只含 wrapper.run（读 plan 写入的设备缓冲）。
        """
        self._fake_decode_batch(bs)
        self._fp8_kv.plan_decode(
            bs, self.batch.block_tables[:bs], self.batch.cache_seqlens[:bs], cg=True
        )

    def _plan_decode_fp8(self, bs: int, pad_to: int | None) -> None:
        """decode 前 plan FlashInfer（录图外）；CG 补齐行做 KV 长度修正。"""
        plan_bs = pad_to if pad_to is not None else bs
        if pad_to is not None and pad_to > bs:
            # CG 补齐行：KV 长度置 1、页表指向 page 0——FlashInfer 要求
            # last_page_len ∈ [1, page_size]；补齐行输出本就被 hidden[:bs] 丢弃。
            self.batch.cache_seqlens[bs:pad_to].fill_(1)
            self.batch.block_tables[bs:pad_to].zero_()
        # 走 CG replay 时 plan CG wrapper；eager（bs 超上限）plan 非 CG wrapper
        use_cg = self.cuda_graph_runner is not None and bs <= self.max_bs
        self._fp8_kv.plan_decode(
            plan_bs,
            self.batch.block_tables[:plan_bs],
            self.batch.cache_seqlens[:plan_bs],
            cg=use_cg,
        )

    @torch.inference_mode()
    def _warmup_verify_shapes(self) -> None:
        """启动期把 verify 会用到的 ``M = 1+k`` 形状预热一遍。

        为什么必须：FP8 per-block GEMM 带 ``@triton.autotune(key=["M","N","K"])``。
        decode 预热只见过 ``M=1``、prefill 见过大 M，而 verify 的 ``M = 1+k`` 是
        **新形状** —— 首次调用会现场 ``do_bench``（每轮清 256MB L2 + 同步），
        落在第一个请求上就是秒级尖刺。提前跑一遍把 autotune 结果缓存好。

        取证方式与 `_fake_decode_batch` 一致：假序列 + 页表指向 page 0，写入的
        脏 KV 会在真实请求首次写该槽时被覆盖。
        """
        if self.config.spec_tokens <= 0 or self._pp_group.size > 1:
            return
        cached = 8  # 让 cu_seqlens_k > cu_seqlens_q，走"带缓存"的注意力分支
        for m in range(2, self.config.spec_tokens + 2):
            vseq = SeqSchedule(
                seq_id=-1,
                block_table=[0],
                num_cached_tokens=cached,
                token_ids=[1] * m,
            )
            self.batch.build_prefill([vseq])
            it = self._recv_intermediate(m) if not self.is_first_pp else None
            with attention_context(self._meta):
                self.model(
                    self.batch.input_ids[:m],
                    self.batch.positions[:m],
                    self._prefill_attn(1, m, m, cached + m, True),
                    it,
                    self._ensure_aux(m),
                )
        torch.cuda.synchronize(self.device)

    def _fake_verify_batch(self, m: int, cached: int = 8) -> None:
        """构造一行假 verify 批（m 个 token、页表指向 page 0）。"""
        vseq = SeqSchedule(
            seq_id=-1,
            block_table=[0],
            num_cached_tokens=cached,
            token_ids=[1] * m,
        )
        self.batch.build_prefill([vseq])

    def _verify_body(self, m: int, max_k: int):
        """verify 的可录图部分：只含模型前向，输出拷进静态缓冲。"""
        with attention_context(self._meta):
            out = self.model(
                self.batch.input_ids[:m],
                self.batch.positions[:m],
                self._prefill_attn(1, m, m, max_k, True),
                None,
                self._ensure_aux(m),
            )
        self._verify_out[:m].copy_(out)
        return out

    def _capture_verify_graphs(self) -> None:
        """按 m = 2..k+1 各录一张 verify 图（失败则整体回退 eager）。

        `max_seqlen_k` 是 attention kernel 的标量参数（进图后不可变），所以用
        `spec_verify_graph_max_k` 当上界桶：实际 `p+m <= 桶` 才用图，否则回退。
        桶内浪费的只是 tile 粒度（空块早退），不是正确性问题。
        """
        if not self.config.spec_verify_graph or self.config.spec_tokens <= 0:
            return
        if self._pp_group.size > 1:
            print("[WARN] spec_verify_graph 需要 pp_size == 1，已回退 eager verify")
            return
        max_k = int(self.config.spec_verify_graph_max_k)
        graphs: dict[int, torch.cuda.CUDAGraph] = {}
        try:
            self._verify_out = torch.empty(
                self.config.max_num_batched_tokens,
                self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
            for m in range(2, self.config.spec_tokens + 2):
                self._fake_verify_batch(m)
                self._verify_body(m, max_k)  # 图外 warmup：算子初始化 / autotune
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._verify_body(m, max_k)
                graphs[m] = g
            torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - 录图失败要能整体降级
            print(
                f"[WARN] spec_verify_graph 录图失败（{type(exc).__name__}: {exc}），"
                "已回退 eager verify"
            )
            self._verify_out = None
            graphs.clear()
        self._verify_graphs = graphs

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
        if self.config.spec_tokens > 0 and self.config.spec_draft == "dspark":
            # DSpark 每步都走 verify（eager 的 prefill 通路）：草稿需要它产出的
            # aux hidden，同时这条路径完全不进 decode CUDA Graph。
            return self._run_verify(seqs, drafts=self._propose_dspark(seqs))
        if self.config.spec_tokens > 0 and any(s.draft_ids for s in seqs):
            return self._run_verify(seqs)
        bs = len(seqs)
        pad_to = (
            self.cuda_graph_runner._pad_bs(bs)
            if self.cuda_graph_runner is not None and bs <= self.max_bs
            else None
        )
        self.batch.build_decode(seqs, pad_to=pad_to)
        if self._fp8_kv is not None:
            self._plan_decode_fp8(bs, pad_to)

        if self.cuda_graph_runner is not None and bs <= self.max_bs:
            if not self.is_first_pp:
                self._recv_intermediate(bs, static=True)
            self._meta.fp8_kv_cg = self._fp8_kv is not None
            hidden, _ = self.cuda_graph_runner.replay(bs)
            self._meta.fp8_kv_cg = False
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

    def _ensure_aux(self, n_tokens: int) -> torch.Tensor | None:
        """按需分配 aux hidden 缓冲并注册要采集的层（仅 DSpark 用）。

        缓冲一次分配、按 ``n_tokens`` 切片传入；前向里用 ``out=`` **原地写入**
        （不新建中间张量），所以地址固定，将来若要进 CUDA Graph 也不用改。
        """
        wants_aux = self.config.spec_draft == "dspark" or self.config.spec_aux_hidden
        if self.config.spec_tokens <= 0 or not wants_aux:
            return None
        if self._aux_buf is None:
            ids = list(self.config.spec_target_layer_ids)
            body = self.model.model
            setter = getattr(body, "set_aux_hidden_layers", None)
            if setter is None:
                raise NotImplementedError(
                    "spec_draft='dspark' 需要目标模型输出 aux hidden states；"
                    f"{type(self.model).__name__} 未提供该能力（v1 仅 dense Qwen3）"
                )
            setter(ids)
            self._aux_buf = torch.empty(
                self.config.max_num_batched_tokens,
                len(ids) * body.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
        if n_tokens > self._aux_buf.shape[0]:
            raise ValueError(
                f"aux buffer too small: need {n_tokens} > {self._aux_buf.shape[0]}"
            )
        # 顺手把草稿模型也加载好：prefill 结束就要把整段 prompt 的 context 交给它
        self._ensure_dspark()
        return self._aux_buf[:n_tokens]

    def _ensure_dspark(self):
        """懒加载草稿模型并校验它与 config 的层表一致。"""
        if self.config.spec_tokens <= 0 or self.config.spec_draft != "dspark":
            return None
        if self._dspark is None:
            from ..spec.dspark.drafter import DSparkDrafter
            from ..spec.dspark.loader import load_dspark_draft

            draft = load_dspark_draft(
                self.config.spec_draft_model, device=self.device, dtype=self.dtype
            )
            want = list(self.config.spec_target_layer_ids)
            got = list(draft.target_layer_ids)
            if got != want:
                raise ValueError(
                    f"draft target_layer_ids {got} != config {want}; "
                    "两边必须一致，否则喂给草稿的 context 特征顺序会错位"
                )
            self._dspark = DSparkDrafter(draft, device=self.device)
        return self._dspark

    def _propose_dspark(self, seqs: list[SeqSchedule]) -> list[int]:
        """worker 侧起草：用上一步验证出的 context 段 + anchor 出 k 个候选。"""
        dr = self._ensure_dspark()
        if dr is None or len(seqs) != 1:
            return []
        s = seqs[0]
        first = self._dspark_pending.pop(s.seq_id, None)
        self._draft_probs = None
        tokens, probs = dr.propose(
            s.seq_id,
            anchor_token_id=s.token_ids[-1],
            start=s.num_cached_tokens,
            first_ctx=first,
        )
        self._draft_probs = probs
        return tokens

    @torch.inference_mode()
    def _run_verify(
        self, seqs: list[SeqSchedule], drafts: list[int] | None = None
    ) -> "_SpecSamples | None":
        """投机验证：一次因果前向算出 ``[anchor] + drafts`` 全部位置的 logits。

        greedy 接受规则：位置 ``p+i`` 的 argmax 预测的是 ``p+i+1`` 的 token，
        所以草稿 ``drafts[j]`` 被接受 ⟺ ``argmax(logits[j]) == drafts[j]``。
        取最长可接受前缀 j 后，第 j+1 个 token 由 ``logits[j]`` 给出（全接受时
        是最后一个位置的 bonus）。

        为什么复用 prefill 通路：verify 本质是「一段带因果掩码的多 token
        前向 + 读已有 KV」，与带前缀缓存的 prefill 完全同构（``cu_seqlens`` +
        ``block_tables``），不需要为它单独写 attention 内核。

        注意：这里 ``argmax().tolist()`` 是一次同步读回。接受裁决是控制面
        决策（决定 KV 账本），无法像普通 decode 那样只把 id 异步 D2H；v1 每次
        只读 ≤8 个整数，开销可忽略，Phase 5 再考虑把裁决整体搬到 GPU。
        """
        assert len(seqs) == 1, "v1 speculative decode supports a single sequence"
        s = seqs[0]
        drafts = list(s.draft_ids if drafts is None else drafts)
        m = 1 + len(drafts)

        vseq = SeqSchedule(
            seq_id=s.seq_id,
            block_table=s.block_table,
            num_cached_tokens=s.num_cached_tokens,
            token_ids=[s.token_ids[-1]] + drafts,
            need_sample=False,
            temperature=s.temperature,
            top_p=s.top_p,
            top_k=s.top_k,
            min_p=s.min_p,
        )
        n_tokens, max_q, max_k, has_cache = self.batch.build_prefill([vseq])
        assert n_tokens == m, f"verify built {n_tokens} tokens, expected {m}"
        if self._fp8_kv is not None and has_cache:
            self._fp8_kv.plan_prefill(
                self.batch.cu_seqlens_q[:2],
                self.batch.cu_seqlens_k[:2],
                self.batch.block_tables[:1],
            )
        graph = self._verify_graphs.get(m)
        if graph is not None and (
            s.num_cached_tokens + m <= self.config.spec_verify_graph_max_k
        ):
            # 录图口径：build_prefill 已把 input_ids/positions/slot_mapping/
            # cu_seqlens 写进静态 GPU 缓冲，replay 只跑模型前向
            graph.replay()
            hidden = self._verify_out[:m]
        else:
            it = self._recv_intermediate(n_tokens) if not self.is_first_pp else None
            with attention_context(self._meta):
                hidden = self.model(
                    self.batch.input_ids[:n_tokens],
                    self.batch.positions[:n_tokens],
                    self._prefill_attn(1, n_tokens, max_q, max_k, has_cache),
                    it,
                    self._ensure_aux(n_tokens),
                )
        if not self.is_last_pp:
            self._send_intermediate(hidden)
            return None

        # compute_logits 内含 vocab-parallel 集合通信，必须所有 TP rank 都执行
        logits = self.model.compute_logits(hidden)  # [m, vocab]
        toks: list[int] = []
        accepted = 0
        if self.batch.all_greedy:
            toks, accepted = accept_greedy(drafts, logits.argmax(dim=-1).tolist())
            if self.config.spec_exact_kv:
                # bit-exact：逐个位置用 decode 内核重写 KV（含集合通信，必须在
                # 所有 rank 上执行；只有 driver 的返回值会被上层使用）
                toks = self._kv_exact_rerun(s, drafts, accepted, toks)
            elif self.config.spec_tie_eps > 0.0:
                toks = self._tie_rerun(s, drafts, logits, accepted, toks)
        elif self.is_driver:
            # 温度采样：用**采样器实际使用的分布**（含 top-k/p/min-p 截断）做
            # 拒绝采样，才能保证输出分布与普通采样逐位一致。
            draft_probs = self._draft_probs
            if draft_probs is not None and draft_probs.shape[0] != len(drafts):
                draft_probs = None  # 草稿被截断 → 退回 one-hot 语义
            toks, accepted = accept_rejection(
                drafts, self._verify_probs(logits), draft_probs
            )

        if self._dspark is not None and self._aux_buf is not None and self.is_driver:
            # 草稿侧账本：本次验证通过的位置 p..p+accepted 的 hidden 就是下一步
            # 要追加的 context 段（上游 `_update` 同款，取 [:accepted+1]）
            self._dspark.update(
                s.seq_id,
                self._aux_buf[: accepted + 1].clone().unsqueeze(0),
                start=s.num_cached_tokens + 1 + accepted,
            )
        self._draft_probs = None

        if not self.is_driver:
            return _SpecSamples()
        return _SpecSamples(
            tokens=toks,
            lens=[len(toks)],
            # worker 侧起草（dspark）时，调度器不知道提了几个，必须回传
            proposed=[len(drafts)] if self._dspark is not None else [],
        )

    def _kv_exact_rerun(
        self,
        s: SeqSchedule,
        drafts: list[int],
        accepted: int,
        toks: list[int],
    ) -> list[int]:
        """bit-exact 模式：把 verify 写入的 anchor + accepted 草稿 KV 逐个用
        **decode 内核**重写，并用最后一次重算的 argmax 作为输出 token。

        为什么必须"逐个"重写：位置 q 的 K/V = ``k_proj(norm(h_q))``，而 ``h_q``
        是上一层 attention 的输出。verify 走 paged-varlen、baseline decode 走
        ``flash_attn_with_kvcache``，归约顺序不同 → ``h_q`` 差 ~1ulp → 写进
        cache 的 K/V 也不同。只重算 emit 位置（``_tie_rerun``）管不到这件事：
        8B 实测分叉发生在 verify 之后的**普通 decode** 步（第 45 个 token），
        原因是它读到了 verify 写下的、略不同的 KV。

        重写顺序必须从前往后：重算位置 p+i 时会读 p..p+i-1 的 KV，而那些位置
        在前一轮已被纠正成 decode 内核的版本。

        代价：每个 verify 步多 ``accepted+1`` 次 1-token 前向，accept_len 低时
        收益会被吃光——所以定位是校验/参考模式，不是性能模式。
        """
        block_tokens = [s.token_ids[-1]] + list(drafts)
        base_pos = s.num_cached_tokens
        token: int | None = None
        for i in range(accepted + 1):
            vseq = SeqSchedule(
                seq_id=s.seq_id,
                block_table=s.block_table,
                num_cached_tokens=base_pos + i,
                token_ids=[block_tokens[i]],
                need_sample=False,
                temperature=s.temperature,
                top_p=s.top_p,
                top_k=s.top_k,
                min_p=s.min_p,
            )
            out = self._run_decode([vseq])
            if out is not None:
                token = int(out[0])
        if token is None:  # 非 driver rank：只为参与集合通信
            return toks
        return block_tokens[1 : accepted + 1] + [token]

    def _tie_rerun(
        self,
        s: SeqSchedule,
        drafts: list[int],
        logits: torch.Tensor,
        accepted: int,
        toks: list[int],
    ) -> list[int]:
        """（**部分**修复）greedy tie 守卫：emit 位置近并列时改用 decode 内核重算。

        背景：verify 走 paged-varlen prefill（``flash_attn_varlen_func``），
        baseline decode 走 ``flash_attn_with_kvcache``，attention 归约顺序不同，
        logits 差 ~1ulp；目标分布精确并列时 argmax 会翻转。

        ⚠️ **它只能覆盖"emit 位置恰好并列"这一种情况**，因为 verify 写进 cache 的
        anchor/accepted KV 本身也带着 ~1ulp 差异（K/V 的输入 hidden 由 attention
        算出），后续普通 decode 读到这份 KV 仍会在并列处翻转——8B 实测分叉正是
        发生在 verify 之后的 decode 步（第 45 个 token），本守卫完全没触发。
        需要逐位保证时用 ``spec_exact_kv``（逐个位置用 decode 内核重写 KV）。

        修法：把该位置重新按 decode 通路跑一次（`_run_decode`，与 baseline 完全
        同路径，含 CUDA Graph / FP8 KV plan），用它的 argmax 作为 emit token。
        只多一次 1-token 前向，且只在 tie 步发生。
        """
        top2 = logits[accepted].topk(2).values
        if float(top2[0] - top2[1]) >= self.config.spec_tie_eps:
            return toks

        pos = s.num_cached_tokens + accepted
        block_tokens = [s.token_ids[-1]] + list(drafts)
        vseq = SeqSchedule(
            seq_id=s.seq_id,
            block_table=s.block_table,
            num_cached_tokens=pos,
            token_ids=[block_tokens[accepted]],
            need_sample=False,
            temperature=s.temperature,
            top_p=s.top_p,
            top_k=s.top_k,
            min_p=s.min_p,
        )
        # 注意：这里会重写位置 pos 的 KV——K/V 由线性层算出，与 attention 内核
        # 无关，所以写入的值与 verify 那次完全相同，只是用 decode 内核重新
        # 计算了 logits。
        out = self._run_decode([vseq])
        if out is None:  # 非 driver rank：只为参与集合通信，结果不使用
            return toks
        return toks[:accepted] + [int(out[0])]

    def _verify_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """把单序列的采样参数广播到 verify 的 m 个位置，交给 ``Sampler`` 归一化。"""
        m = logits.shape[0]
        b = self.batch
        return self.sampler.final_probs(
            logits,
            b.temperatures[:1].expand(m),
            top_ps=b.top_ps[:1].expand(m),
            top_ks=b.top_ks[:1].expand(m),
            min_ps=b.min_ps[:1].expand(m),
            all_greedy=False,
            any_greedy=b.any_greedy,
            use_top_k=b.use_top_k,
            use_top_p=b.use_top_p,
            use_min_p=b.use_min_p,
            max_top_k=b.max_top_k,
        )

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
        if self._fp8_kv is not None and has_cache:
            # 有前缀缓存：FlashInfer paged prefill 读 fp8 cache，需先 plan
            self._fp8_kv.plan_prefill(
                self.batch.cu_seqlens_q[: bs + 1],
                self.batch.cu_seqlens_k[: bs + 1],
                self.batch.block_tables[:bs],
            )
        it = self._recv_intermediate(n_tokens) if not self.is_first_pp else None

        aux = self._ensure_aux(n_tokens)
        if aux is None and self.prefill_fn is not None and all(
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
                    aux,
                )

        if (
            self._dspark is not None
            and self._aux_buf is not None
            and len(seqs) == 1
            and seqs[0].num_cached_tokens == 0
            and seqs[0].need_sample
        ):
            # 首步的 context = 整段 prompt 的 aux hidden。只在"单 chunk、从头开始"
            # 时记录；chunked prefill 下不做草稿（v1 取舍，正确性不受影响）。
            # 草稿模型的 target_hidden_states 约定是 [1, n, F]（带 batch 维）
            self._dspark_pending[seqs[0].seq_id] = (
                self._aux_buf[:n_tokens].clone().unsqueeze(0)
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
        self, p: torch.Tensor | None, d: torch.Tensor | "_SpecSamples | None"
    ) -> tuple[int, int]:
        spec = d if isinstance(d, _SpecSamples) else None
        self._spec = spec
        n_p = 0 if p is None else p.numel()
        if spec is not None:
            # 投机路径的 token 已在 _run_verify 里同步取回，不再走 pinned 缓冲
            n_d = len(spec.tokens)
        else:
            n_d = 0 if d is None else d.numel()
        need = n_p + n_d
        if need > self._sample_cpu.numel():
            self._sample_cpu = torch.empty(need, dtype=torch.int64, pin_memory=True)
        cur = torch.cuda.current_stream()
        self._copy_stream.wait_stream(cur)
        with torch.cuda.stream(self._copy_stream):
            if spec is None and d is not None:
                self._sample_cpu[:n_d].copy_(d, non_blocking=True)
            if p is not None:
                self._sample_cpu[n_d : n_d + n_p].copy_(p, non_blocking=True)
        self._copy_event.record(self._copy_stream)
        return n_p, n_d

    def read_samples(
        self, n_p: int, n_d: int
    ) -> tuple[list[int], list[int], list[int] | None, list[int] | None]:
        """返回 ``(prefill_ids, decode_ids, decode_lens, decode_proposed)``。

        ``decode_lens[i]`` = 第 i 条序列本轮产出的 token 数（= 接受草稿数+1）。
        ``decode_proposed[i]`` = 第 i 条序列**提出**的草稿数——只有 worker 侧起草
        （dspark）时才非空；ngram 在调度器侧就统计过了，避免重复计数。
        """
        self._copy_event.synchronize()
        p_ids = self._sample_cpu[n_d : n_d + n_p].tolist() if n_p else []
        if self._spec is not None:
            return (
                p_ids,
                list(self._spec.tokens),
                list(self._spec.lens),
                list(self._spec.proposed) or None,
            )
        d_ids = self._sample_cpu[:n_d].tolist() if n_d else []
        return p_ids, d_ids, None, None

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
        # 量化路径下 prefill 编译暂不启用：Dynamo × Triton autotune kernel
        # （fp8 per-block GEMM）的交互尚未验证，正确性优先，回退 eager prefill。
        # 待 4090 验证后开放（quant_config 非 None 即门控）。
        if getattr(self.config, "quant_config", None) is not None:
            print(
                "[WARN] prefill torch.compile disabled under quantization "
                "(unverified Dynamo x Triton-fp8 interaction; eager prefill)"
            )
            return None
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
        if self._fp8_kv is not None:
            self._fp8_kv.destroy()
            self._fp8_kv = None
        self.prefill_fn = None
