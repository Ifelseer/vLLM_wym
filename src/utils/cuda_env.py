from __future__ import annotations

import os
from collections.abc import Sequence


def format_flashinfer_arch(major: int, minor: int) -> str:
    if major == 9:
        return f"{major}.{minor}a"
    if major == 12:
        # SM120: ``12.0f`` needs *nvcc* ≥ 12.9. Do NOT trust torch.version.cuda
        # alone — wheels may be cu129 while system toolkit is still 12.8.
        import re
        import subprocess

        nvcc_ver: tuple[int, int] | None = None
        home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or ""
        candidates = []
        if home:
            candidates.append(os.path.join(home, "bin", "nvcc"))
        candidates.append("nvcc")
        for nvcc in candidates:
            try:
                out = subprocess.check_output(
                    [nvcc, "--version"],
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=5,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            m = re.search(r"release\s+(\d+)\.(\d+)", out)
            if m:
                nvcc_ver = (int(m.group(1)), int(m.group(2)))
                break
        if nvcc_ver is not None and nvcc_ver >= (12, 9):
            return "12.0f" if minor == 0 else f"12.{minor}a"
        # nvcc 12.8 (or unknown): a-family works with CUDA toolkit 12.8.
        return "12.0a" if minor == 0 else f"12.{minor}a"
    if major >= 10:
        return f"{major}.{minor}a"
    return f"{major}.{minor}"


def flashinfer_arch_list_for_devices(device_ids: Sequence[int]) -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to resolve FlashInfer arch list")
    n = torch.cuda.device_count()
    archs: list[str] = []
    seen: set[str] = set()
    for d in device_ids:
        di = int(d)
        if di < 0 or di >= n:
            raise ValueError(
                f"device_ids contains invalid GPU index {di} (visible count={n})"
            )
        major, minor = torch.cuda.get_device_capability(di)
        arch = format_flashinfer_arch(int(major), int(minor))
        if arch not in seen:
            seen.add(arch)
            archs.append(arch)
    if not archs:
        raise ValueError("device_ids is empty; cannot set FlashInfer arch list")
    return " ".join(archs)


def refresh_flashinfer_compilation_context() -> None:
    """Rebuild FlashInfer's process-global arch set from the current env."""
    try:
        import flashinfer.jit.core as fi_core
        from flashinfer.compilation_context import CompilationContext
    except ImportError:
        return
    fi_core.current_compilation_context = CompilationContext()


def apply_flashinfer_arch_list(
    device_ids: Sequence[int],
    *,
    overwrite: bool = False,
) -> str:
    if overwrite or "FLASHINFER_CUDA_ARCH_LIST" not in os.environ:
        arch_list = flashinfer_arch_list_for_devices(device_ids)
        os.environ["FLASHINFER_CUDA_ARCH_LIST"] = arch_list
    else:
        arch_list = os.environ["FLASHINFER_CUDA_ARCH_LIST"]
    refresh_flashinfer_compilation_context()
    return arch_list
