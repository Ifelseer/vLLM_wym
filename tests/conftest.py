from __future__ import annotations

import pytest
import torch


needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)
needs_2gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs >=2 GPUs",
)


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "unit: CPU-only fast tests")
    config.addinivalue_line("markers", "gpu: single-GPU tests")
    config.addinivalue_line("markers", "distributed: multi-GPU tests")
    config.addinivalue_line("markers", "slow: long-running / large-model tests")


def pytest_collection_modifyitems(config, items) -> None:
    for item in items:
        path = str(item.fspath).replace("\\", "/")
        if "/tests/unit/" in path:
            item.add_marker(pytest.mark.unit)
        elif "/tests/gpu/" in path:
            item.add_marker(pytest.mark.gpu)
        elif "/tests/distributed/" in path:
            item.add_marker(pytest.mark.distributed)
