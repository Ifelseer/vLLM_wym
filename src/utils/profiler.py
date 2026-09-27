from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

MODULE_TREE: dict[str, str] = {
    "engine": "Driver loop (schedule → execute → postprocess)",
    "rpc": "ZMQ/pipe RPC + pickle of Sequence batch",
    "sched": "Scheduler + BlockManager + Radix prefix cache",
    "runner": "ModelRunner: inputs / attn meta / forward / sample / CG",
    "worker": "Worker brackets around forward + PP wait",
    "nccl": "TP all-reduce",
    "pp": "PP P2P tensor + metadata",
    "layer": "Optional fine-grained layer hooks",
}

_ENABLED = os.environ.get("VLLM_WYM_PROF", "").strip() in ("1", "true", "True")


def enabled() -> bool:
    return _ENABLED


def set_enabled(flag: bool) -> None:
    """Test helper / dynamic toggle (normally set via env before import)."""
    global _ENABLED
    _ENABLED = bool(flag)


@dataclass
class _Bucket:
    total_ms: float = 0.0
    count: int = 0

    def add(self, ms: float) -> None:
        self.total_ms += ms
        self.count += 1


@dataclass
class SysProfiler:
    buckets: dict[str, _Bucket] = field(default_factory=lambda: defaultdict(_Bucket))

    def add(self, name: str, ms: float) -> None:
        self.buckets[name].add(ms)

    def reset(self) -> None:
        self.buckets.clear()

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, b in sorted(self.buckets.items()):
            out[k] = {
                "count": b.count,
                "total_ms": round(b.total_ms, 3),
                "mean_ms": round(b.total_ms / max(b.count, 1), 4),
            }
        return out

    def rollup(self) -> list[dict[str, Any]]:
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for name, b in self.buckets.items():
            top = name.split(".", 1)[0]
            totals[top] += b.total_ms
            counts[top] += b.count
        grand = sum(totals.values()) or 1e-9
        rows = []
        for top, ms in sorted(totals.items(), key=lambda x: -x[1]):
            rows.append(
                {
                    "module": top,
                    "desc": MODULE_TREE.get(top, ""),
                    "total_ms": round(ms, 3),
                    "count": counts[top],
                    "pct": round(100.0 * ms / grand, 2),
                }
            )
        return rows


_PROF = SysProfiler()


def get_profiler() -> SysProfiler:
    return _PROF


def reset() -> None:
    _PROF.reset()


def summary() -> dict[str, Any]:
    return _PROF.summary()


def rollup() -> list[dict[str, Any]]:
    return _PROF.rollup()


class Timer:
    """Wall-clock timer; no-op when profiling is disabled."""

    __slots__ = ("name", "_t0")

    def __init__(self, name: str):
        self.name = name
        self._t0 = 0.0

    def __enter__(self):
        if _ENABLED:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if _ENABLED:
            _PROF.add(self.name, (time.perf_counter() - self._t0) * 1000.0)
        return False


class CudaTimer:
    """CUDA-event timer (device sync on exit). Prefer ``Timer`` for CPU paths."""

    __slots__ = ("name", "_start", "_end")

    def __init__(self, name: str):
        self.name = name
        self._start = None
        self._end = None

    def __enter__(self):
        if not _ENABLED:
            return self
        import torch

        if not torch.cuda.is_available():
            return self
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record()
        return self

    def __exit__(self, *exc):
        if not _ENABLED or self._start is None or self._end is None:
            return False
        import torch

        self._end.record()
        torch.cuda.synchronize()
        _PROF.add(self.name, self._start.elapsed_time(self._end))
        return False
