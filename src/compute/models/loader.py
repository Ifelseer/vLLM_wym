from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils._loader import export_module  # noqa: E402

export_module(globals(), Path(__file__), "model_loader", depth=4)
