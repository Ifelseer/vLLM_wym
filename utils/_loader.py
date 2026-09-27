"""Load utils.* modules into version-local shims."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def repo_root(shim_file: Path, *, depth: int) -> Path:
    return shim_file.resolve().parents[depth]


def export_module(
    globals_dict: dict,
    shim_file: Path,
    name: str,
    *,
    depth: int,
) -> None:
    root = repo_root(shim_file, depth=depth)
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    mod = importlib.import_module(f"utils.{name}")
    names = list(mod.__all__)
    globals_dict.update({n: getattr(mod, n) for n in names})
    globals_dict["__all__"] = names
