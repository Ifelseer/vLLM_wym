from __future__ import annotations

import copy
import multiprocessing as mp
import tempfile
import time

import msgspec
import torch
import zmq

from ..config import EngineConfig
from ..data import EngineCoreOutput, Sequence
from .client import EngineCoreClient
from .core import EngineCore

# ==============================================================================
# MPClient：dp>1 时 Engine 到各副本 EngineCore 的调用层。
#
# make_client 在 dp_size>1 时构造本类。之后 Engine 的 add_request / step /
# has_unfinished / destroy 都走这里，不再持有本进程 EngineCore。
#
# 每个副本是独立进程里的完整引擎，
# 经 _replica_config 把 dp 归 1、切开 device_ids 与 nccl_port。
# 门面只做 round-robin 投递和输出汇总；副本内部的 TP/PP 仍由 Executor 展开。
# ==============================================================================


# ------------------------------------------------------------------ #
# 门面 ↔ 副本 的消息
# ------------------------------------------------------------------ #


class RequestWire(msgspec.Struct):
    """一次 add_request 所需的字段。跨进程不能传 Sequence 指针，因此拷贝 token 与采样参数。"""

    seq_id: int
    token_ids: list[int]
    block_size: int
    temperature: float
    top_p: float | None
    top_k: int | None
    min_p: float | None
    max_tokens: int
    ignore_eos: bool

    @classmethod
    def from_sequence(cls, seq: Sequence) -> "RequestWire":
        return cls(
            seq_id=seq.seq_id,
            token_ids=list(seq.token_ids),
            block_size=seq.block_size,
            temperature=seq.temperature,
            top_p=seq.top_p,
            top_k=seq.top_k,
            min_p=seq.min_p,
            max_tokens=seq.max_tokens,
            ignore_eos=seq.ignore_eos,
        )

    def to_sequence(self) -> Sequence:
        return Sequence(
            token_ids=list(self.token_ids),
            block_size=self.block_size,
            seq_id=self.seq_id,
            num_prompt_tokens=len(self.token_ids),
            last_token=self.token_ids[-1],
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            max_tokens=self.max_tokens,
            ignore_eos=self.ignore_eos,
        )


class CoreRequest(msgspec.Struct):
    """门面 → 副本。"""

    op: str  # "add" | "shutdown"
    req: RequestWire | None = None


class CoreOutput(msgspec.Struct):
    """副本 → 门面：本步 EngineCore.step 的产出。"""

    outputs: list[EngineCoreOutput]
    metric: int = 0


# ------------------------------------------------------------------ #
# 副本进程
# ------------------------------------------------------------------ #


def run_engine_core_proc(
    config: EngineConfig,
    *,
    eos_token_id: int,
    device: str,
    dtype: torch.dtype,
    in_addr: str,
    out_addr: str,
    ready,  # multiprocessing.Event
) -> None:
    """
        单个 DP 副本：本进程 EngineCore + 一对 ZMQ socket。

        先排空入口上的 add/shutdown，有未完成请求则 step 一次并把结果推给门面；
        队列空时短睡，避免空转占满 CPU。
    """
    assert config.dp_size == 1, "replica config must flatten dp_size to 1"
    core = EngineCore(
        config, eos_token_id=eos_token_id, device=torch.device(device), dtype=dtype
    )
    ctx = zmq.Context()
    in_sock = ctx.socket(zmq.PULL)
    in_sock.bind(in_addr)
    out_sock = ctx.socket(zmq.PUSH)
    out_sock.bind(out_addr)

    poller = zmq.Poller()
    poller.register(in_sock, zmq.POLLIN)
    enc = msgspec.msgpack.Encoder()
    dec = msgspec.msgpack.Decoder(CoreRequest)

    ready.set()
    try:
        while True:
            shutdown = False
            while True:
                socks = dict(poller.poll(0))
                if in_sock not in socks:
                    break
                msg = dec.decode(in_sock.recv())
                if msg.op == "shutdown":
                    shutdown = True
                    break
                if msg.op == "add" and msg.req is not None:
                    core.add_request(msg.req.to_sequence())
            if shutdown:
                break

            if core.has_unfinished():
                outputs, metric = core.step()
                out_sock.send(enc.encode(CoreOutput(outputs=outputs, metric=metric)))
            else:
                time.sleep(0.0002)
    finally:
        core.destroy()
        in_sock.close(0)
        out_sock.close(0)
        ctx.term()


# ------------------------------------------------------------------ #
# 门面客户端
# ------------------------------------------------------------------ #


class MPClient(EngineCoreClient):
    """
        spawn dp_size 个副本进程，ZMQ IPC 投递请求、汇总各副本的 step 输出。

        add_request 按 round-robin 选副本；step 阻塞等到至少一路输出，再非阻塞收齐当前已就绪的结果。
    """

    def __init__(
        self,
        config: EngineConfig,
        *,
        eos_token_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.config = config
        self.dp_size = config.dp_size
        assert self.dp_size > 1

        self._ctx = zmq.Context()
        self._enc = msgspec.msgpack.Encoder()
        self._dec = msgspec.msgpack.Decoder(CoreOutput)
        self._in: list[zmq.Socket] = []
        self._out: list[zmq.Socket] = []
        self._poller = zmq.Poller()
        self._procs: list[mp.Process] = []
        self._pending: set[int] = set()
        self._next = 0

        ctx = mp.get_context("spawn")
        tmp = tempfile.mkdtemp(prefix="vllm_wym_dp_")
        for r in range(self.dp_size):
            replica_cfg = self._replica_config(r)
            in_addr = f"ipc://{tmp}/in_{r}.ipc"
            out_addr = f"ipc://{tmp}/out_{r}.ipc"
            ready = ctx.Event()
            p = ctx.Process(
                target=run_engine_core_proc,
                args=(),
                kwargs=dict(
                    config=replica_cfg,
                    eos_token_id=eos_token_id,
                    device=f"cuda:{replica_cfg.device_ids[0]}",
                    dtype=dtype,
                    in_addr=in_addr,
                    out_addr=out_addr,
                    ready=ready,
                ),
                # 副本内 TP/PP 还会再 spawn Worker；daemon 进程不能有子进程。
                daemon=False,
            )
            p.start()
            self._procs.append(p)
            ready.wait(timeout=60)
            if not ready.is_set():
                raise RuntimeError(f"DP replica {r} failed to initialize")

            in_sock = self._ctx.socket(zmq.PUSH)
            in_sock.connect(in_addr)
            out_sock = self._ctx.socket(zmq.PULL)
            out_sock.connect(out_addr)
            self._in.append(in_sock)
            self._out.append(out_sock)
            self._poller.register(out_sock, zmq.POLLIN)

    def _replica_config(self, r: int) -> EngineConfig:
        """第 r 个副本：dp=1，GPU 切成 tp×pp 一段，nccl_port 错开以免 FileStore 撞车。"""
        per = self.config.tp_size * self.config.pp_size
        if self.config.device_ids is not None:
            devices = self.config.device_ids[r * per : (r + 1) * per]
            if len(devices) != per:
                raise ValueError(
                    f"device_ids length {len(self.config.device_ids)} "
                    f"incompatible with dp={self.dp_size} × (tp×pp={per})"
                )
        else:
            start = r * per
            devices = list(range(start, start + per))
        replica = copy.copy(self.config)
        replica.dp_size = 1
        replica.device_ids = devices
        replica.nccl_port = self.config.nccl_port + r * 16
        return replica

    def add_request(self, seq: Sequence) -> None:
        self._pending.add(seq.seq_id)
        r = self._next
        self._next = (self._next + 1) % self.dp_size
        req = CoreRequest(op="add", req=RequestWire.from_sequence(seq))
        self._in[r].send(self._enc.encode(req))

    def step(self) -> tuple[list[EngineCoreOutput], int]:
        return self._drain(block=True)

    def has_unfinished(self) -> bool:
        return bool(self._pending)

    def _drain(self, *, block: bool) -> tuple[list[EngineCoreOutput], int]:
        """block=True 时先等到任意一路有输出，再把当前已就绪的副本结果收完。"""
        outputs: list[EngineCoreOutput] = []
        metric = 0
        first = True
        while True:
            timeout = None if (block and first) else 0
            socks = dict(self._poller.poll(timeout))
            if not socks:
                break
            for sock in socks:
                msg = self._dec.decode(sock.recv())
                outputs.extend(msg.outputs)
                metric += msg.metric
                for o in msg.outputs:
                    if o.is_finished:
                        self._pending.discard(o.seq_id)
            first = False
        return outputs, metric

    def destroy(self) -> None:
        for sock in self._in:
            try:
                sock.send(self._enc.encode(CoreRequest(op="shutdown")))
            except zmq.ZMQError:
                pass
        for p in self._procs:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()
        for s in self._in:
            s.close(0)
        for s in self._out:
            s.close(0)
        self._ctx.term()
        self._procs.clear()
        self._pending.clear()
