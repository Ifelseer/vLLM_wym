from __future__ import annotations

import torch

# ==============================================================================
# PP 层区间、中间激活、占位层。
#
# get_pp_indices 给 Qwen3 决定本 rank 加载哪些层。
# IntermediateTensors / PPMissingLayer 给跨 stage 的 hidden 和没分到的层。
# ==============================================================================

_PP_LAYER_RANGES: list[tuple[int, int]] | None = None


def ensure_divisibility(numerator: int, denominator: int) -> None:
    assert numerator % denominator == 0, (
        f"{numerator} is not divisible by {denominator}"
    )


def divide(numerator: int, denominator: int) -> int:
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def _equal_split(
    num_hidden_layers: int, pp_rank: int, pp_size: int
) -> tuple[int, int]:
    n = num_hidden_layers // pp_size
    start = pp_rank * n
    end = num_hidden_layers if pp_rank == pp_size - 1 else start + n
    return start, end


def get_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int
) -> tuple[int, int]:
    """本 rank 的 [start, end)。set_pp_layer_ranges 之后走不均分。"""
    if _PP_LAYER_RANGES is not None:
        return _PP_LAYER_RANGES[pp_rank]
    return _equal_split(num_hidden_layers, pp_rank, pp_size)


def set_pp_layer_ranges(layer_counts: list[int] | None) -> None:
    """按每 stage 层数设区间，例如 12G+24G 的 12/28。None 回到等分。"""
    global _PP_LAYER_RANGES
    if layer_counts is None:
        _PP_LAYER_RANGES = None
        return
    ranges: list[tuple[int, int]] = []
    start = 0
    for c in layer_counts:
        ranges.append((start, start + c))
        start += c
    _PP_LAYER_RANGES = ranges


class IntermediateTensors(dict[str, torch.Tensor]):
    """PP 边界上的 hidden_states / residual。"""


class PPMissingLayer(torch.nn.Identity):
    """本 rank 没有的层，forward 原样返回第一个参数。"""

    def forward(self, *args, **kwargs):
        return args[0] if args else next(iter(kwargs.values()))
