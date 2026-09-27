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
