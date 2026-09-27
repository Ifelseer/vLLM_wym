"""Pick GPUs and map physical PCI ids → isolated CUDA_VISIBLE_DEVICES runs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch

from src.utils.cuda_env import format_flashinfer_arch


def gpu_name(i: int) -> str:
    return torch.cuda.get_device_properties(i).name


def gpu_free_mib(i: int) -> int:
    free, _ = torch.cuda.mem_get_info(i)
    return int(free // (1024 * 1024))


def pick_gpus(n: int, *, prefer: str = "3090", min_free_mib: int = 8000) -> list[int]:
    """Same-SKU device indices in the **current** visible CUDA order."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    groups: dict[str, list[int]] = {}
    for i in range(torch.cuda.device_count()):
        if gpu_free_mib(i) < min_free_mib:
            continue
        groups.setdefault(gpu_name(i), []).append(i)
    preferred = [g for name, g in groups.items() if prefer.lower() in name.lower()]
    others = [g for name, g in groups.items() if prefer.lower() not in name.lower()]
    for g in preferred + others:
        if len(g) >= n:
            return g[:n]
    raise RuntimeError(
        f"need {n} homogeneous GPUs with >={min_free_mib} MiB free; {groups}"
    )


def arch_for_device(i: int) -> str:
    major, minor = torch.cuda.get_device_capability(i)
    return format_flashinfer_arch(int(major), int(minor))


def run_isolated(
    physical_ids: list[int],
    worker_src: str,
    *args: str,
    timeout: int = 600,
) -> str:
    """Run ``worker_src`` as ``python -c`` with only ``physical_ids`` visible.

    Sets ``CUDA_VISIBLE_DEVICES`` and ``FLASHINFER_CUDA_ARCH_LIST`` **before**
    importing FlashInfer (required on mixed-GPU hosts).
    """
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in physical_ids)
    # Resolve arch from the parent process view (full bus).
    archs = []
    seen: set[str] = set()
    for i in physical_ids:
        a = arch_for_device(i)
        if a not in seen:
            seen.add(a)
            archs.append(a)
    env["FLASHINFER_CUDA_ARCH_LIST"] = " ".join(archs)
    env.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
    root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = root + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    cmd = [sys.executable, "-c", worker_src, *args]
    proc = subprocess.run(
        cmd,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"isolated worker failed (devices={physical_ids} arch={env['FLASHINFER_CUDA_ARCH_LIST']})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout
