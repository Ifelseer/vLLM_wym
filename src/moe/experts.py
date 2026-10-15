"""Experts weights, EP map, and fused forward.

FP8 量化（``quant.method == "fp8"``）时：
- ``w13_weight`` [E_local, 2i_local, h] / ``w2_weight`` [E_local, h, i_local]
  为 ``float8_e4m3fn``；布局沿用 cutlass 约定 ``[up; gate]``。
- ``w13_scale`` [E_local, 2i_local/128, h/128] / ``w2_scale``
  [E_local, h/128, i_local/128] 为 fp32 per-block 128×128 scale。
- 加载两态：预量化 checkpoint（fp8 权重 + 独立 ``*.weight_scale``）直写；
  BF16 checkpoint（``quantize_at_load``）逐 piece 在线 per-block 量化
  （gate/up 各自成块，无需全量 staging）。
- 前向走 Triton grouped GEMM（block scale 版，见 kernels/triton.py）；
  cutlass 后端的 block-scale FP8 仅 SM90 支持，SM89 上强制 Triton。

INT4 weight-only（``quant.method in ("awq", "gptq")``）时，内部统一
**in-major** 布局（与 dense 侧 ``quant/kernels/int4_gemm.py`` 同约定）：
- ``w13_qweight`` [E_local, h, 2i_local/8] int32（in=h，out=2i，[up; gate] 序）
- ``w13_scales`` [E_local, h/G, 2i_local] fp16；``w13_zeros`` uint8（asym）
- ``w2_qweight`` [E_local, i_local, h/8]（in=i，out=h）
- ``w2_scales`` [E_local, i_local/G, h]；``w2_zeros`` uint8（asym）
加载期把 AWQ/GPTQ checkpoint 归一化到该布局（解包/交织还原/转置），
kernel 内按 k-block 解包 + scale 校正，激活保持 bf16。
"""

from __future__ import annotations

import torch
from torch import nn

from ..distributed import divide, ensure_divisibility
from ..quant.kernels.fp8_gemm import FP8_DTYPE, quantize_fp8_per_block
from ..quant.kernels.int4_gemm import (
    INT4_SYM_BIAS,
    qweight_to_in_major,
    qzeros_to_in_major,
    scales_to_in_major,
    unpack_int4_packed_cols,
)
from .fused import fused_moe_forward
from .parallel import MoEParallelConfig

_SCALE = 128
_INT4 = ("awq", "gptq")


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def local_expert_range(num_experts: int, ep_size: int, ep_rank: int) -> tuple[int, int]:
    n_local = divide(num_experts, ep_size)
    start = ep_rank * n_local
    return start, start + n_local


def make_expert_map(
    num_experts: int,
    ep_size: int,
    ep_rank: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """``map[global_id] = local_id``; remote experts are ``-1``."""
    mapping = torch.full((num_experts,), -1, dtype=dtype, device=device)
    start, end = local_expert_range(num_experts, ep_size, ep_rank)
    mapping[start:end] = torch.arange(end - start, dtype=dtype, device=device)
    return mapping


def _hf_gate_up_to_cutlass(w13_hf: torch.Tensor) -> torch.Tensor:
    gate, up = w13_hf.chunk(2, dim=1)
    if w13_hf.dtype == FP8_DTYPE:
        # fp8 的 chunk/cat 走 uint8 view，避开部分版本对 fp8 cat 的限制
        u8 = torch.cat(
            [up.view(torch.uint8), gate.view(torch.uint8)], dim=1
        )
        return u8.view(FP8_DTYPE)
    return torch.cat([up, gate], dim=1)


def _w13_slice(weight_name: str, intermediate_size: int) -> slice:
    if weight_name.endswith(("up_proj.weight", "w3")):
        return slice(0, intermediate_size)
    if weight_name.endswith(("gate_proj.weight", "w1")):
        return slice(intermediate_size, 2 * intermediate_size)
    raise ValueError(f"unknown gate/up weight: {weight_name}")


def _normalize_fused_weight(
    loaded: torch.Tensor,
    expected_shape: tuple[int, int, int],
    name: str,
) -> torch.Tensor:
    assert loaded.ndim == 3, f"{name} must be 3D, got {tuple(loaded.shape)}"
    if loaded.shape == expected_shape:
        return loaded
    transposed = (expected_shape[0], expected_shape[2], expected_shape[1])
    if loaded.shape == transposed:
        loaded = loaded.transpose(1, 2).contiguous()
    assert loaded.shape == expected_shape, (
        f"{name} shape {tuple(loaded.shape)} != {expected_shape}"
    )
    return loaded


class Experts(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        parallel: MoEParallelConfig,
        quant=None,
    ) -> None:
        super().__init__()
        ensure_divisibility(intermediate_size, parallel.tp_size)
        ensure_divisibility(num_experts, parallel.ep_size)
        self.global_num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tp_size = parallel.tp_size
        self.tp_rank = parallel.tp_rank
        self.ep_size = parallel.ep_size
        self.ep_rank = parallel.ep_rank
        self.fused_backend = parallel.fused_backend
        self.expert_start, self.expert_end = local_expert_range(
            num_experts, parallel.ep_size, parallel.ep_rank
        )
        self.num_experts = self.expert_end - self.expert_start
        self.intermediate_size_per_partition = divide(intermediate_size, parallel.tp_size)
        i = self.intermediate_size_per_partition
        # 量化配置（None = 不量化；支持 fp8 与 INT4 weight-only）。
        # INT8 W8A8 不覆盖 MoE 专家：显式 raise 而非让专家层静默留在 BF16
        # （那会让 `--quantization int8` 的显存/精度收益变成不可预期的混合路径）。
        if quant is not None and quant.method == "int8":
            raise NotImplementedError(
                "INT8 W8A8 is not implemented for MoE experts; use fp8 / "
                "awq / gptq for MoE models (INT8 covers dense linear layers)"
            )
        self.quant = (
            quant if (quant is not None and quant.method in ("fp8", *_INT4)) else None
        )
        self.int4 = quant is not None and quant.method in _INT4
        if self.int4:
            g = int(quant.group_size)
            ensure_divisibility(i, g)
            ensure_divisibility(hidden_size, g)
            self.group_size = g
            asym = (quant.method == "awq") or (not quant.sym)
            self.int4_asym = asym
            # GPTQ sym（uint4b8）：存储值 = 量化值 + 8，无 qzeros 张量
            self.zero_bias = 0.0 if asym else INT4_SYM_BIAS
            # w13：in=h（量化组沿 h），out=2i（[up; gate] 序，本地）
            # 注意 requires_grad=False：int32/uint8/fp8 不能 requires_grad；
            # zeros 初始化：EP 过滤/覆盖断言才有确定语义（empty 是未定义内存）
            self.w13_qweight = nn.Parameter(
                torch.zeros(self.num_experts, hidden_size, (2 * i) // 8, dtype=torch.int32),
                requires_grad=False,
            )
            self.w13_scales = nn.Parameter(
                torch.zeros(self.num_experts, hidden_size // g, 2 * i, dtype=torch.float16),
                requires_grad=False,
            )
            if asym:
                self.w13_zeros = nn.Parameter(
                    torch.zeros(self.num_experts, hidden_size // g, 2 * i, dtype=torch.uint8),
                    requires_grad=False,
                )
            else:
                self.register_parameter("w13_zeros", None)
            # w2：in=i（量化组沿 i），out=h
            self.w2_qweight = nn.Parameter(
                torch.zeros(self.num_experts, i, hidden_size // 8, dtype=torch.int32),
                requires_grad=False,
            )
            self.w2_scales = nn.Parameter(
                torch.zeros(self.num_experts, i // g, hidden_size, dtype=torch.float16),
                requires_grad=False,
            )
            if asym:
                self.w2_zeros = nn.Parameter(
                    torch.zeros(self.num_experts, i // g, hidden_size, dtype=torch.uint8),
                    requires_grad=False,
                )
            else:
                self.register_parameter("w2_zeros", None)
            setattr(self.w13_qweight, "weight_loader", self.weight_loader_w13)
            setattr(self.w2_qweight, "weight_loader", self.weight_loader_w2)
        elif self.quant is not None:
            ensure_divisibility(i, _SCALE)
            ensure_divisibility(hidden_size, _SCALE)
            self.w13_weight = nn.Parameter(
                torch.empty(self.num_experts, 2 * i, hidden_size, dtype=FP8_DTYPE),
                requires_grad=False,
            )
            self.w2_weight = nn.Parameter(
                torch.empty(self.num_experts, hidden_size, i, dtype=FP8_DTYPE),
                requires_grad=False,
            )
            self.register_buffer(
                "w13_scale",
                torch.zeros(
                    self.num_experts,
                    _cdiv(2 * i, _SCALE),
                    _cdiv(hidden_size, _SCALE),
                    dtype=torch.float32,
                ),
                persistent=False,
            )
            self.register_buffer(
                "w2_scale",
                torch.zeros(
                    self.num_experts,
                    _cdiv(hidden_size, _SCALE),
                    _cdiv(i, _SCALE),
                    dtype=torch.float32,
                ),
                persistent=False,
            )
        else:
            self.w13_weight = nn.Parameter(
                torch.empty(self.num_experts, 2 * i, hidden_size)
            )
            self.w2_weight = nn.Parameter(torch.empty(self.num_experts, hidden_size, i))
            setattr(self.w13_weight, "weight_loader", self.weight_loader_w13)
            setattr(self.w2_weight, "weight_loader", self.weight_loader_w2)
        self.register_buffer(
            "expert_map",
            make_expert_map(
                num_experts,
                parallel.ep_size,
                parallel.ep_rank,
                device=torch.device("cpu"),
            ),
            persistent=False,
        )

    def _local_id(self, expert_id: int) -> int | None:
        if expert_id < self.expert_start or expert_id >= self.expert_end:
            return None
        return expert_id - self.expert_start

    # ------------------------------------------------------------------ #
    # per-expert 权重加载（HF 形态：experts.N.{gate,up,down}_proj.weight）
    # ------------------------------------------------------------------ #

    def weight_loader_w13(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        expert_id: int,
    ) -> None:
        local = self._local_id(expert_id)
        if local is None:
            return
        shard = self.intermediate_size_per_partition
        seg = _w13_slice(weight_name, shard)
        if self.quant is None:
            piece = loaded_weight.narrow(0, self.tp_rank * shard, shard)
            dest = param.data[local, seg]
            assert dest.shape == piece.shape, f"{dest.shape} != {piece.shape}"
            dest.copy_(piece)
            return
        piece = loaded_weight.narrow(0, self.tp_rank * shard, shard)
        self._write_w13_fp8(param, local, seg.start, piece)

    def weight_loader_w2(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        expert_id: int,
    ) -> None:
        local = self._local_id(expert_id)
        if local is None:
            return
        shard = self.intermediate_size_per_partition
        if self.quant is None:
            piece = loaded_weight.narrow(1, self.tp_rank * shard, shard)
            dest = param.data[local]
            assert dest.shape == piece.shape, f"{dest.shape} != {piece.shape}"
            dest.copy_(piece)
            return
        piece = loaded_weight.narrow(1, self.tp_rank * shard, shard)
        self._write_w2_fp8(param, local, piece)

    def _write_w13_fp8(
        self, param: nn.Parameter, local: int, row_off: int, piece: torch.Tensor
    ) -> None:
        """写一段 [shard, h]（up 或 gate）到 w13_weight；在线量化时顺带写 scale。"""
        if piece.dtype == FP8_DTYPE:
            param.data[local, row_off : row_off + piece.shape[0]].copy_(piece)
            return
        if not self.quant.quantize_at_load:
            raise ValueError(
                "fp8 MoE expert got non-fp8 weights but quantize_at_load is off; "
                "use a pre-quantized checkpoint or quantization=fp8"
            )
        q, s = quantize_fp8_per_block(piece)
        param.data[local, row_off : row_off + piece.shape[0]].copy_(q)
        go = q.shape[0] // _SCALE
        self.w13_scale[local, row_off // _SCALE : row_off // _SCALE + go].copy_(s)

    def _write_w2_fp8(
        self, param: nn.Parameter, local: int, piece: torch.Tensor
    ) -> None:
        """写一段 [h, shard] 到 w2_weight；在线量化时顺带写 scale。"""
        if piece.dtype == FP8_DTYPE:
            param.data[local].copy_(piece)
            return
        if not self.quant.quantize_at_load:
            raise ValueError(
                "fp8 MoE expert got non-fp8 weights but quantize_at_load is off; "
                "use a pre-quantized checkpoint or quantization=fp8"
            )
        q, s = quantize_fp8_per_block(piece)
        param.data[local].copy_(q)
        gi = q.shape[1] // _SCALE
        self.w2_scale[local, :, self.tp_rank * gi : (self.tp_rank + 1) * gi].copy_(s)

    def weight_loader_w13_scale(
        self,
        param,
        loaded_scale: torch.Tensor,
        weight_name: str,
        expert_id: int,
    ) -> None:
        """预量化 checkpoint 的段 scale：全局段网格 → 本地 w13_scale 段块。"""
        if self.quant is None:
            return
        local = self._local_id(expert_id)
        if local is None:
            return
        shard = self.intermediate_size_per_partition
        seg = _w13_slice(weight_name, shard)
        local_go = _cdiv(shard, _SCALE)
        src = loaded_scale.float()
        self.w13_scale[
            local, seg.start // _SCALE : seg.start // _SCALE + local_go
        ].copy_(src[self.tp_rank * local_go : (self.tp_rank + 1) * local_go])

    def weight_loader_w2_scale(
        self,
        param,
        loaded_scale: torch.Tensor,
        expert_id: int,
    ) -> None:
        """预量化 checkpoint 的 down scale：全局网格列 → 本地 w2_scale 列块。"""
        if self.quant is None:
            return
        local = self._local_id(expert_id)
        if local is None:
            return
        shard = self.intermediate_size_per_partition
        local_gi = _cdiv(shard, _SCALE)
        src = loaded_scale.float()
        self.w2_scale[
            local, :, self.tp_rank * local_gi : (self.tp_rank + 1) * local_gi
        ].copy_(src[:, self.tp_rank * local_gi : (self.tp_rank + 1) * local_gi])

    # ------------------------------------------------------------------ #
    # INT4 weight-only（AWQ / GPTQ）per-expert 加载
    # ------------------------------------------------------------------ #

    def _int4_param(self, field: str, which: str):
        # checkpoint 字段名 → 参数名：qzeros → zeros（qweight/scales 同名）
        suffix = "zeros" if field == "qzeros" else field
        return getattr(self, f"{which}_{suffix}")

    def weight_loader_int4(
        self,
        loaded_weight: torch.Tensor,
        weight_name: str,
        expert_id: int,
        field: str,
    ) -> None:
        """把 AWQ/GPTQ 的 per-expert 张量归一化并写入 INT4 参数。

        Args:
            weight_name: ``"gate_proj.weight"`` / ``"up_proj.weight"`` /
                ``"down_proj.weight"``（与 FP8 路径同口径，用于定位段）。
            field: ``"qweight"`` / ``"scales"`` / ``"qzeros"``。
        """
        local = self._local_id(expert_id)
        if local is None:
            return
        i = self.intermediate_size_per_partition
        is_w13 = weight_name in ("gate_proj.weight", "up_proj.weight")
        in_full, out_full = (
            (self.hidden_size, self.intermediate_size)
            if is_w13
            else (self.intermediate_size, self.hidden_size)
        )
        # 1) 归一化到内部 in-major（方向/交织/打包轴在此一次性处理）
        if field == "qweight":
            norm = qweight_to_in_major(
                loaded_weight,
                in_full=in_full,
                out_full=out_full,
                method=self.quant.method,
            )  # [in, out/8] int32
        elif field == "scales":
            norm = scales_to_in_major(
                loaded_weight,
                in_full=in_full,
                out_full=out_full,
                group_size=self.group_size,
            )  # [in/G, out] fp32
        else:
            packed = qzeros_to_in_major(
                loaded_weight,
                in_full=in_full,
                out_full=out_full,
                group_size=self.group_size,
            )
            norm = unpack_int4_packed_cols(packed, out_full)  # [in/G, out] uint8

        # 2) TP 切分 + 写入
        if is_w13:
            # out=2i（[up | gate] 序）按 TP 切；qweight 的列是打包列（out/8）
            seg_out = 0 if weight_name == "up_proj.weight" else i
            if field == "qweight":
                c0 = seg_out // 8
                cl = i // 8
                src = norm[
                    :, self.tp_rank * cl : (self.tp_rank + 1) * cl
                ]  # [h, i/8]
                self.w13_qweight.data[local, :, c0 : c0 + cl].copy_(src)
            else:
                src = norm[:, self.tp_rank * i : (self.tp_rank + 1) * i]  # [h/G, i]
                p = self._int4_param(field, "w13")
                p.data[local, :, seg_out : seg_out + i].copy_(src.to(p.dtype))
        else:
            # down：in=i 按 TP 切（量化组维随之切；i 整除 G）
            g_local = i // self.group_size
            if field == "qweight":
                src = norm[self.tp_rank * i : (self.tp_rank + 1) * i]  # [i, h/8]
                self.w2_qweight.data[local].copy_(src)
            else:
                src = norm[self.tp_rank * g_local : (self.tp_rank + 1) * g_local]
                p = self._int4_param(field, "w2")
                p.data[local].copy_(src.to(p.dtype))

    def int4_payload(self) -> dict:
        """INT4 前向所需的权重/scale/zeros 组（供 fused_moe_forward 透传）。"""
        return {
            "w13_qweight": self.w13_qweight,
            "w2_qweight": self.w2_qweight,
            "w13_scales": self.w13_scales,
            "w2_scales": self.w2_scales,
            "w13_zeros": self.w13_zeros,
            "w2_zeros": self.w2_zeros,
        }

    # ------------------------------------------------------------------ #
    # 融合专家张量加载（vLLM/SGLang 风格 3D）
    # ------------------------------------------------------------------ #

    def load_fused_w13(self, loaded: torch.Tensor) -> None:
        loaded = _normalize_fused_weight(
            loaded,
            (self.global_num_experts, 2 * self.intermediate_size, self.hidden_size),
            "gate_up",
        )
        loaded = _hf_gate_up_to_cutlass(loaded)[self.expert_start : self.expert_end]
        shard = self.intermediate_size_per_partition
        start = self.tp_rank * shard
        u = loaded[:, start : start + shard, :]
        g = loaded[
            :,
            self.intermediate_size + start : self.intermediate_size + start + shard,
            :,
        ]
        self.w13_weight.data.copy_(torch.cat([u, g], dim=1))

    def load_fused_w13_scale(self, loaded: torch.Tensor) -> None:
        """3D 融合 gate_up 的 scale：[E, 2i/128, h/128]（HF [gate;up] 序）。"""
        if self.quant is None:
            return
        loaded = loaded.float()
        e, gr, hc = loaded.shape
        # HF 段序 [gate; up] → cutlass [up; gate]
        gate_go = _cdiv(self.intermediate_size, _SCALE)
        up = loaded[:, :gate_go, :]
        gate = loaded[:, gate_go:, :]
        reordered = torch.cat([up, gate], dim=1)[self.expert_start : self.expert_end]
        shard = self.intermediate_size_per_partition
        local_go = _cdiv(shard, _SCALE)
        start = self.tp_rank * local_go
        self.w13_scale.copy_(reordered[:, start : start + 2 * local_go, :])

    def load_fused_w2(self, loaded: torch.Tensor) -> None:
        loaded = _normalize_fused_weight(
            loaded,
            (self.global_num_experts, self.hidden_size, self.intermediate_size),
            "down",
        )[self.expert_start : self.expert_end]
        shard = self.intermediate_size_per_partition
        start = self.tp_rank * shard
        self.w2_weight.data.copy_(loaded[:, :, start : start + shard])

    def load_fused_w2_scale(self, loaded: torch.Tensor) -> None:
        """3D 融合 down 的 scale：[E, h/128, i/128]。"""
        if self.quant is None:
            return
        loaded = loaded.float()[self.expert_start : self.expert_end]
        shard = self.intermediate_size_per_partition
        local_gi = _cdiv(shard, _SCALE)
        start = self.tp_rank * local_gi
        self.w2_scale.copy_(loaded[:, :, start : start + local_gi])

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if not hidden_states.is_cuda:
            raise RuntimeError("MoE fused kernels require CUDA tensors")
        mapping = self.expert_map.to(device=topk_ids.device, non_blocking=True)
        if self.int4:
            return fused_moe_forward(
                backend=self.fused_backend,
                hidden_states=hidden_states,
                topk_ids=mapping[topk_ids.long()],
                topk_weights=topk_weights,
                w13_weight=self.w13_qweight,
                w2_weight=self.w2_qweight,
                w13_scale=self.w13_scales,
                w2_scale=self.w2_scales,
                w13_zeros=self.w13_zeros,
                w2_zeros=self.w2_zeros,
                int4=True,
                zero_bias=self.zero_bias,
            )
        return fused_moe_forward(
            backend=self.fused_backend,
            hidden_states=hidden_states,
            topk_ids=mapping[topk_ids.long()],
            topk_weights=topk_weights,
            w13_weight=self.w13_weight,
            w2_weight=self.w2_weight,
            w13_scale=getattr(self, "w13_scale", None),
            w2_scale=getattr(self, "w2_scale", None),
        )
