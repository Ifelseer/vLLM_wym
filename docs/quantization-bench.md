# 量化性能基准：单卡 Qwen3-8B · batch-8

本文记录量化前后（BF16 / FP8 W8A8 / INT8 W8A8）的端到端性能与显存对比，并解释每组数字的成因。测试沿用 [`parallel-benchmark.md`](parallel-benchmark.md) 的 batch8 口径。

- [1] 测试口径
- [2] 结果
- [3] 数据解读
- [4] 与既有并行报告的一致性
- [5] 复现方式
- [6] 局限

## 1. 测试口径

**硬件 / 软件**：单卡 RTX 4090 24 GB（sm_89，非 4090 D）；PyTorch 2.13.0+cu132、Triton 3.7.1、FlashInfer 0.6.15、CUDA 13.2；计算精度 bfloat16。

**模型**：Qwen3-8B（8.19 B 参数；36 层，hidden 4096，32 q heads / 8 kv heads，head_dim 128，intermediate 12288，vocab 151936，embed 与 lm_head 不共享）。

**负载**：256-token prompt（重复的技术段落文本）、greedy、`ignore_eos`、`max_tokens=128`、**batch=8 并发**。引擎配置固定为 `max_num_seqs=8`、`context_len=2048`、`num_kvcache_blocks=64`（64 × 256 = 16,384 token，bf16 下 2.25 GiB）、`gpu_memory_utilization=0.85`、`prefix_backend=none`。

| 指标 | 定义 |
|---|---|
| TTFT (ms) | `add_request` 到 batch 内首个 completion token 的墙钟。8 路 prompt 在同一步一起 prefill，因此是 **2048 token 的 prefill 时间**（不是单流 TTFT）。 |
| decode (tok/s) | `batch × (n−1) / (e2e − TTFT)`，聚合吞吐 |
| e2e (tok/s) | `batch × n / 墙钟`，即并行报告里的 batch8 口径 |
| peak (GiB) | `torch.cuda.max_memory_allocated`，引擎生命周期峰值（权重 + KV + 激活 + CUDA Graph 池） |
| token match | batch 内第一条流 128 个贪心 token 与 BF16 基线的逐位一致率 |

**量化方式**：在 BF16 checkpoint 加载时在线量化（`quantize_at_load`），不依赖预量化权重。`lm_head` / `embed_tokens` / `mlp.gate` 按配置保持 bf16。

- **FP8 W8A8**：权重 per-block 128×128 静态 scale；激活 per-token × 128-group 动态量化；Triton fused kernel。
- **INT8 W8A8**：权重 per-channel（`[out]`）静态 scale；激活 per-token 动态量化；`M > 16` 走 `torch._int_mm`（cuBLASLt int8），decode 小 M 走 Triton 内核。
- **FP8 KV cache**（仅额外一组）：per-layer 固定 scale（首次写入校准后冻结），FlashInfer paged attention。

每档单次运行；TTFT 约有 5–10% 晃动，decode / e2e 是 100+ token 的平均，较稳。模型经 ModelScope 下载（官方 HF 源当时的实测速率只有约 1.4 MB/s）。

## 2. 结果

### 2.1 CUDA Graph（部署口径）

| 档位 | TTFT (ms) | decode (tok/s) | e2e (tok/s) | peak (GiB) | 显存 vs BF16 | token match |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 184.7 | 435.0 | 406.3 | **18.48** | 1.00× | — |
| FP8 W8A8 | 184.2 | **711.0** | 634.7 | **11.96** | **0.65×** | 100% |
| INT8 W8A8 | 160.9 | **724.6** | 655.1 | **12.47** | **0.67×** | 100% |
| FP8 + FP8 KV | 184.3 | 667.0 | 599.7 | **11.00** | **0.60×** | 100% |

- FP8：decode **1.63×**，e2e **1.56×**
- INT8：decode **1.67×**，e2e **1.61×**
- FP8 KV：KV 池 2.25 → **1.12 GiB**；decode 相对 FP8 组 −6%

### 2.2 eager 对照

| 档位 | TTFT (ms) | decode (tok/s) | e2e (tok/s) | peak (GiB) | decode vs BF16 |
|---|---:|---:|---:|---:|---:|
| BF16 | 183.7 | 300.7 | 287.5 | 18.42 | 1.00× |
| FP8 W8A8 | 189.5 | 235.1 | 227.0 | 11.96 | **0.78×** |
| INT8 W8A8 | 160.9 | 220.1 | 214.4 | 12.47 | **0.73×** |

eager 下结论与 CG **相反**：量化后 decode 反而慢 22% / 27%。BF16 eager 的数字重跑一次为 295.3 tok/s / 282.5 e2e，复现一致。

### 2.3 显存拆解

| | 量化后权重 | embed + lm_head (bf16) | KV 池 | 激活 / 图池 / 采样 | peak |
|---|---:|---:|---:|---:|---:|
| BF16 | 15.26 | —（含在权重内） | 2.25 | ~0.97 | 18.48 |
| FP8 / INT8 | 6.47 | 2.32 | 2.25 | ~0.92 | 11.96 / 12.47 |
| FP8 + FP8 KV | 6.47 | 2.32 | 1.12 | ~0.9 | 11.00 |

单位 GiB。权重字节数按参数量折算：BF16 = 8.19 B × 2 B；量化档 = 6.95 B × 1 B + 1.24 B × 2 B（未量化模块）。Qwen3-8B 每 token 的 KV 为 `2 × 36 × 8 × 128 × 2 B` ≈ 144 KiB。

## 3. 数据解读

### 3.1 decode 为什么快 1.6–1.7×

batch-8 的 decode 每步要把全部权重读一遍，是**权重带宽受限**。权重从 15.26 GiB 降到 8.79 GiB，理论上限 `15.26 / 8.79 ≈ 1.74×`；实测 FP8 1.63×（达到上限的 94%）、INT8 1.67×（96%）。

剩下的 4–6% 是量化路径自身的开销：激活量化 kernel、fp32 scale 校正、以及自写 Triton GEMM 与 cuBLAS 的效率差（INT8 在大 M 用 cuBLASLt，效率更高，所以更接近上限）。

同一步内 KV 的读取量约为 `8 路 × 384 token × 144 KiB ≈ 0.43 GiB`，只占权重流量的 3–5%，所以加速比基本由权重字节决定，与 KV 量化关系不大。

### 3.2 TTFT 为什么基本不变

8 路 × 256 token 的 prefill 约 `2 × 8.19 G × 2048 ≈ 33.5 TFLOP`，是**算力瓶颈**；同一份权重在这一步被复用 2048 次（算术强度高），权重字节量不主导。FP8 省下的权重带宽被激活量化与 scale 校正的额外 kernel 抵消，净效果约为 0（184.7 → 184.2 ms）。

INT8 的 TTFT 低 13%（160.9 ms）是单次测量，方向合理：它的 prefill 走 `torch._int_mm`（cuBLASLt int8），比自写的 Triton per-block FP8 内核更成熟；但幅度可能有测量波动，不宜当成稳定结论。

### 3.3 eager 为什么反而更慢

量化后每个线性层从「1 次 cuBLAS」变成「激活量化 kernel + GEMM + scale 反量化 epilogue」2–3 个 kernel。batch-8 decode 时每层 GEMM 只有 M=8，计算很小，**kernel launch 开销占主导**；没有 CUDA Graph 把 launch 摊掉，带宽收益补不回来，于是 decode 从 300.7 掉到 235.1 / 220.1。

这正是 [`bench/README.md`](../bench/README.md) 里强调的口径纪律：**量化收益必须在 CG replay 口径下评估**；eager 数字只用于说明"不开图会适得其反"。

### 3.4 峰值显存为什么只降 35%（权重本身降 42%）

权重：15.26 → 8.79 GiB（−42%）。但峰值里还有两块钱不随权重缩放：

- `lm_head` 与 `embed_tokens` 按配置保持 bf16（1.24 B 参数 = 2.32 GiB）；
- KV 池 2.25 GiB、激活 / CUDA Graph 池约 0.9 GiB。

所以 `18.48 → 11.96 GiB`（−35%）。INT8 的 12.47 GiB 比 FP8 高约 0.5 GiB，但两者权重字节完全相同（int8 与 fp8 都是 1 B/元素，per-channel scale 只有几 MB），差异来自分配器与图池的瞬态布局，不是方案固有开销；以单次测量为准。

BF16 峰值 18.48 GiB，24 GB 卡上还剩约 5.5 GB，实际部署可以把 KV 池从 64 blocks 继续放大；但本文为与并行报告对齐，固定 64 blocks。

### 3.5 FP8 KV 的收益与代价

- **收益**：KV 池 2.25 → 1.12 GiB（同 blocks），峰值再降 0.96 GiB；同显存预算下 blocks 可以翻倍（0.6B 实测 629 → 1258 blocks）。
- **代价**：decode 711 → 667 tok/s（−6%）。原因有三：attention 从 FA2 换成 FlashInfer paged；每个 decode step 在图外执行一次 `plan`（内部含 D2H 同步）；fp8 KV 在 attention epilogue 需要反量化。当前 384-token 上下文下 KV 流量占比很小（§3.1），省下的带宽抵不过这条通路的额外开销。
- 因此 FP8 KV 应作为**容量手段**（长上下文 / 大 batch / 显存吃紧）而不是吞吐手段。

### 3.6 token match 为什么是 100%

三档量化的第一条流 128 个贪心 token 与 BF16 完全一致。这是**这个负载**的结论：256-token 的重复段落文本，续写置信度高。此前的 teacher-forced 实测（[`quantization.md`](quantization.md)）表明，量化引入的分歧只出现在 BF16 自身 top1–top2 logit 间隔 ≤ 0.5 的近似并列处——量化噪声（~0.25）会翻转 argmax，随后级联改变后续 token。因此自由生成的逐 token 一致率不能作为通用验收口径；本文的 100% 只说明该 workload 下没有触及近似并列。

## 4. 与既有并行报告的一致性

| 指标 | 本文 | `parallel-benchmark.md` A.2（4090 D 单卡） | 说明 |
|---|---:|---:|---|
| BF16 CG batch8 (tok/s) | 406.3 | 400.2 | +1.5%，一致 |
| BF16 CG TTFT (ms) | 184.7 | 40.8 | **不可直接比**：本文是 batch-8（2048 token）prefill，报告里是单流（256 token）prefill |
| BF16 eager batch8 (tok/s) | 282.5–287.5 | 363.0 | 差约 25%，重跑可复现；CG 口径一致说明模型与配置对齐，差异更可能来自引擎版本与单次测量，本轮未定位 |

## 5. 复现方式

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.

# CG（主口径）
python bench/quant_bench.py --model ~/huggingface/Qwen3-8B --devices 0 \
    --quantizations none,fp8,int8 --cg --batch 8 --n 128 --reps 3 \
    --prompt "$(python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/root/huggingface/Qwen3-8B")
base = ("Machine learning systems have evolved rapidly over the past decade. "
        "Large language models now serve billions of requests every day, and the "
        "cost of inference is dominated by memory bandwidth and attention latency. "
        "Quantization reduces the number of bits used to store weights and caches, "
        "which lowers memory traffic and increases the number of sequences that fit "
        "on a single device. ")
text = base
while len(tok(text).input_ids) < 256:
    text += base
print(tok.decode(tok(text).input_ids[:256]))
PY
)" --json quant8b.json

# eager 对照：去掉 --cg
# FP8 KV：--quantizations fp8 --quant-kv-cache --cg
```

`--batch` 是本次为 batch-8 口径新增的参数（此前脚本固定单流）：decode / e2e 按 `batch × token` 计算，TTFT 取 batch 首个 token。

## 6. 局限

- 每档单次运行（与并行报告一致）；TTFT 有 5–10% 晃动。
- 固定 64 KV blocks；未测 KV 池放大后的表现。
- greedy、无前缀缓存（`prefix_backend=none`）；未测采样模式与真实多轮负载。
- 未测长 prompt（8K）的 prefill 主导场景；FP8 在长 prefill 下的表现可能不同于本文的 256-token 口径。
- 未测 INT4（AWQ/GPTQ）：需要预量化 checkpoint，本轮只覆盖可在线量化的 FP8 / INT8。
