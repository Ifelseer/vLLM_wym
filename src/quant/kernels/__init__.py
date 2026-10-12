"""量化 kernel 层：Triton / torch 原语封装。

- ``fp8_gemm``：per-tensor（``torch._scaled_mm``）/ per-block 128×128（Triton）
- ``int4_gemm``：AWQ/GPTQ packed 布局 fused dequant-GEMM（Triton）
- ``int8_gemm``：``torch._int_mm`` + scale 融合

kernel 的 workspace 一律按容量池化（同 ``src/moe/fused.py`` 的 ``_scratch``
模式），保证 Decode CUDA Graph 录图/replay 期间无逐帧 cudaMalloc。
"""
