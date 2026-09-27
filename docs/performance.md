# 性能基准

本文说明性能测试的口径、本引擎的端到端结果，以及与官方 vLLM 的对比方式。

- [1] 基准口径
- [2] 本引擎端到端结果
- [3] 与官方 vLLM 对比
- [4] 基准脚本

## 1. 基准口径

统一口径，保证结果可比：

- **采样**：`greedy + ignore_eos`，固定 prompt / max_tokens / seed。
- **指标**：TTFT（ms，稳态 prefill + 首 token）、decode tok/s、端到端 tok/s。
- **负载**：单请求与小 batch（默认 1 与 8）分开报；MoE 用 16 条变长请求（prompt 100~400 / out 32~64）。
- **对齐**：同 GPU、同 dtype、同 max_model_len；对比 CUDA Graph 时两边都开 `--cg` / 关 eager。
- **预热**：compile 与 CUDA Graph capture 计入 warmup，不计入计时。

## 2. 本引擎端到端结果

单卡，Qwen3-0.6B，4090 D，PyTorch 2.13 / FlashInfer 0.6.15 / Triton 3.7.1：

| 档位 | 吞吐 (tok/s) |
|------|------------:|
| eager（关闭 Decode CUDA Graph） | 752.4 |
| cg（Decode CUDA Graph） | 2122.0 |

Decode CUDA Graph 带来约 2.8× 的吞吐提升：decode 阶段从逐 kernel 启动变为整图 replay，消除了反复 launch 的开销。

## 3. 与官方 vLLM 对比

`bench/vs_vllm.py` 在**独立的 Python 解释器**里加载官方 vLLM（避免与本项目的 Torch / FlashInfer 栈冲突），两边跑同样的 greedy 负载，输出：

```
ttft_ms / decode_tok_s / e2e_tok_s / tokens（正确性抽查）
```

对比时官方与本引擎都用相同的 prompt / max_tokens / greedy，且对比 CUDA Graph 时两边同开 `--cg` / 关 eager。

## 4. 基准脚本

| 脚本 | 用途 |
|------|------|
| `bench/moe_e2e.py` | 本引擎 MoE / dense 端到端延迟与吞吐 |
| `bench/vs_vllm.py` | 与官方 vLLM 同负载对比（`--model dense|moe`，`--tp` / `--ep`） |
| `bench/run_moe_matrix.sh` | TP / EP / CUDA Graph 的组合延迟矩阵（Qwen1.5-MoE-A2.7B） |

运行：

```bash
# 本引擎 e2e
PYTHONPATH=. python bench/moe_e2e.py --devices 1

# 与官方对比（需另装官方 vLLM，见 bench/README.md）
PYTHONPATH=. python bench/vs_vllm.py --model dense --devices 1 --vllm-python ~/venvs/vllm-official/bin/python
PYTHONPATH=. python bench/vs_vllm.py --model moe  --devices 1 --vllm-python ~/venvs/vllm-official/bin/python

# MoE TP / EP / CG 矩阵
bash bench/run_moe_matrix.sh
```

安装官方 vLLM 建议单独建 venv，不要与本项目混装：

```bash
uv venv ~/venvs/vllm-official
uv pip install vllm --python ~/venvs/vllm-official/bin/python
```
