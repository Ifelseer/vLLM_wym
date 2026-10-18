# 基准

本目录是性能基准脚本：本引擎端到端延迟、与官方 vLLM 对比、MoE TP / EP / CUDA Graph 矩阵。

## 安装官方 vLLM（用于对比）

官方 vLLM 装在**独立的**环境里，不要与本项目的 Torch / FlashInfer 栈混装：

```bash
uv venv ~/venvs/vllm-official
uv pip install vllm --python ~/venvs/vllm-official/bin/python
```

## 运行

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 本引擎
PYTHONPATH=. python bench/moe_e2e.py --devices 1

# 与官方 vLLM 对比
PYTHONPATH=. python bench/vs_vllm.py --model dense --devices 1 \
    --vllm-python ~/venvs/vllm-official/bin/python
PYTHONPATH=. python bench/vs_vllm.py --model moe --devices 1 \
    --vllm-python ~/venvs/vllm-official/bin/python
```

指标（两边同 prompt / max_tokens / greedy）：

```
ttft_ms, decode_tok_s, e2e_tok_s, tokens（正确性抽查）
```

## MoE 矩阵（TP / EP / CUDA Graph）

```bash
bash bench/run_moe_matrix.sh
# 或单格：
PYTHONPATH=. python bench/moe_e2e.py --devices 0 --tp 1 --cg --moe-backend triton
PYTHONPATH=. python bench/moe_e2e.py --devices 0,1 --tp 2 --ep --eager
```

## 量化对比（none / fp8 / int8 / 预量化 auto）

```bash
# 在线量化：同一 BF16 模型多档对比（首行为基线）
PYTHONPATH=. python bench/quant_bench.py --model ~/huggingface/Qwen3-0.6B \
    --quantizations none,fp8,int8 --cg --n 64 --json quant.json

# 预量化 checkpoint：与 BF16 兄弟模型对比
PYTHONPATH=. python bench/quant_bench.py --model ~/huggingface/Qwen3-0.6B \
    --quantized-model ~/huggingface/Qwen3-0.6B-FP8 --cg

# 与官方 vLLM 对比时指定同精度
PYTHONPATH=. python bench/vs_vllm.py --model ~/huggingface/Qwen3-0.6B-FP8 \
    --quantization auto --vllm-python ~/venvs/vllm-official/bin/python
```

`quant_bench.py` 额外输出 `peak GB`（引擎生命周期峰值显存）、相对基线的显存比、
以及与基线逐 token 的一致率。**口径纪律**：`--cg` 与默认 eager 可能给出相反
结论（量化路径的 kernel launch 开销在 CG 下被摊掉），报数必须注明口径。

预量化 checkpoint 的注册表与有效显存估算见
`python run.py --list`（`QUANT_REGISTRY`，来源核验日 2026-10-12）。
