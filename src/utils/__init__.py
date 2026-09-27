"""Cross-cutting helpers with no Engine dependency.

  paths.py      model / HF cache resolution
  profiler.py   optional full-stack timers (``VLLM_WYM_PROF``)
  cuda_env.py   FlashInfer arch pinning for mixed-GPU hosts
"""

from __future__ import annotations

from .cuda_env import apply_flashinfer_arch_list
from .paths import resolve_hf_home, resolve_model_path
from .profiler import CudaTimer, Timer, enabled, reset, rollup, summary

__all__ = [
    "CudaTimer",
    "Timer",
    "apply_flashinfer_arch_list",
    "enabled",
    "reset",
    "resolve_hf_home",
    "resolve_model_path",
    "rollup",
    "summary",
]
