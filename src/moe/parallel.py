"""TP / EP layout and fused MoE backend selection."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Literal

MoeBackend = Literal["cutlass", "triton"]
MoeBackendPrefer = Literal["auto", "cutlass", "triton"]

_CUTLASS_SMS = frozenset({89, 90, 100, 103, 110, 120, 121})


def _sm_tag(device: int | None = None) -> int | None:
    import torch

    if not torch.cuda.is_available():
        return None
    if device is None:
        major, minor = torch.cuda.get_device_capability()
    else:
        major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


def _nvcc_version() -> tuple[int, int] | None:
    candidates: list[str] = []
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if home:
        candidates.append(os.path.join(home, "bin", "nvcc"))
    candidates.append("nvcc")
    for nvcc in candidates:
        try:
            out = subprocess.check_output(
                [nvcc, "--version"], stderr=subprocess.STDOUT, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            continue
        m = re.search(r"release\s+(\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def cutlass_moe_available(*, device: int | None = None) -> bool:
    sm = _sm_tag(device)
    if sm is None or sm not in _CUTLASS_SMS:
        return False
    try:
        from flashinfer import cutlass_fused_moe  # noqa: F401
    except ImportError:
        return False
    if sm >= 120:
        ver = _nvcc_version()
        if ver is None or ver < (12, 8):
            return False
    return True


def triton_moe_available() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_moe_backend(
    prefer: MoeBackendPrefer = "auto",
    *,
    device: int | None = None,
) -> MoeBackend:
    if prefer == "cutlass":
        if not cutlass_moe_available(device=device):
            raise RuntimeError(
                "moe backend=cutlass requested but unavailable "
                f"(sm={_sm_tag(device)} nvcc={_nvcc_version()})"
            )
        return "cutlass"
    if prefer == "triton":
        if not triton_moe_available():
            raise RuntimeError("moe backend=triton requested but triton is not installed")
        return "triton"
    if cutlass_moe_available(device=device):
        return "cutlass"
    if triton_moe_available():
        return "triton"
    raise RuntimeError(
        "No MoE backend available: need FlashInfer cutlass (SM89+) or Triton. "
        f"sm={_sm_tag(device)} nvcc={_nvcc_version()}"
    )


@dataclass(frozen=True)
class MoEParallelConfig:
    """``enable_ep``: collapse MoE TP; shard experts across the old TP ranks."""

    tp_size: int
    tp_rank: int
    ep_size: int
    ep_rank: int
    use_ep: bool
    fused_backend: MoeBackend

    @classmethod
    def from_engine(
        cls,
        *,
        tp_size: int,
        tp_rank: int,
        enable_ep: bool,
        fused_backend: MoeBackend,
    ) -> "MoEParallelConfig":
        if tp_size < 1:
            raise ValueError("tp_size must be >= 1")
        if not 0 <= tp_rank < tp_size:
            raise ValueError(f"tp_rank={tp_rank} out of range for tp_size={tp_size}")
        if fused_backend not in ("cutlass", "triton"):
            raise ValueError(f"fused_backend must be cutlass|triton, got {fused_backend!r}")
        if enable_ep and tp_size > 1:
            return cls(
                tp_size=1,
                tp_rank=0,
                ep_size=tp_size,
                ep_rank=tp_rank,
                use_ep=True,
                fused_backend=fused_backend,
            )
        return cls(
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=1,
            ep_rank=0,
            use_ep=False,
            fused_backend=fused_backend,
        )

    @classmethod
    def single(cls, fused_backend: MoeBackend = "triton") -> "MoEParallelConfig":
        return cls.from_engine(
            tp_size=1, tp_rank=0, enable_ep=False, fused_backend=fused_backend
        )
