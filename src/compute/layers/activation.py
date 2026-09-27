from __future__ import annotations
import torch
from torch import nn


class SiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.silu_and_mul(x)
