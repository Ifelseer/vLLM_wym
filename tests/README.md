# 测试与性能对比

## 覆盖目标

- **Dense**：Qwen3-0.6B（回归引擎 / FlashAttention / 调度）
- **MoE**：Qwen1.5-MoE-A2.7B（TP / EP）、Qwen3-30B-A3B（大模型冒烟）
- 不含 VL / OpenAI serving / DeepEP

## 一、应该测什么

### A. 正确性（tests/）

| 层 | 覆盖 |
|----|------|
| `unit` | 无 GPU：routing / backend / EP map / EngineConfig 的 MoE 开关 |
| `gpu` | 单卡：dense greedy；fused vs naive；A2.7B greedy 冒烟 |
| `distributed` | ≥2 张同型号卡：MoE tp=1 / tp=2 / tp=2+ep 贪心 token 一致 |

### B. MoE 专项

| 项 | 目的 | 入口 |
|----|------|------|
| fused vs naive | Triton / cutlass 数值对齐 | `tests/gpu/test_moe_kernel.py` |
| EP map 分片求和 | `expert_map` 切分语义 | 同上 |
| tp / ep 对齐 | 并行不改 greedy 结果 | `tests/distributed/test_moe_parallel.py` |
| backend 选择 | auto→cutlass\|triton 规则 | `tests/unit/` |
| TTFT / tok/s | MoE 端到端延迟吞吐 | `bench/moe_e2e.py` |
| vs 官方 vLLM | 同负载对比延迟 / 吞吐 / token | `bench/vs_vllm.py` |

### C. 性能对比口径（与官方 vLLM 对齐）

- `greedy + ignore_eos`，固定 prompt / max_tokens / seed
- 指标：TTFT (ms)、decode tok/s、e2e tok/s
- 单请求与小 batch（默认 1 与 8）分开报
- 同 GPU、同 dtype、同 max_model_len；对比 CG 时两边都开 `--cg` / 关 eager

## 二、如何跑

```bash
cd <项目根目录>
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.
# 建议避开默认 cuda:0：用物理卡号
export CUDA_VISIBLE_DEVICES=1   # 例：3090

# 正确性
pytest tests/unit -q
pytest tests/gpu -q --tb=short -m 'not slow'
# 多卡单独跑（避免与其它 GPU 作业争用导致 NCCL 挂起）
pytest tests/distributed -q --tb=short

# MoE 端到端性能（本引擎）
PYTHONPATH=. python bench/moe_e2e.py --device 1

# 与官方 vLLM 对比（需另装官方包，见 bench/README.md）
PYTHONPATH=. python bench/vs_vllm.py --model dense --device 1
PYTHONPATH=. python bench/vs_vllm.py --model moe --device 1
PYTHONPATH=. python bench/vs_vllm.py --model moe --tp 2 --ep --devices 1,2
```

模型默认路径：`~/huggingface/{Qwen3-0.6B, Qwen1.5-MoE-A2.7B, Qwen3-30B-A3B}`。
