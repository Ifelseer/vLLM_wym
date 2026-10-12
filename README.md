# vLLM_wym

<div align="center">

**从零实现的高性能 LLM 推理引擎**，面向 Qwen3（稠密）与 Qwen3-MoE（稀疏专家），
覆盖 PagedAttention、Continuous Batching、CUDA Graph 与 TP / PP / DP / EP 多卡并行。

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12"></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.13-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.13"></a>
<a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-13.0-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA 13.0"></a>
<a href="https://flashinfer.ai/"><img src="https://img.shields.io/badge/FlashInfer-0.6.15-6366f1?style=flat-square" alt="FlashInfer"></a>

</div>

---

## 特性

| 能力 | 说明 |
|------|------|
| **FlashAttention Paged Attention** | prefill 走 varlen paged kernel，decode 用 `flash_attn_with_kvcache` 一次 kernel 处理整批，KV 不再 gather 成连续张量 |
| **Paged KV Cache + 前缀缓存** | 物理 block 池（free / used / cached 三态），hash / radix 两种前缀索引跨请求复用公共前缀 |
| **Continuous Batching** | waiting / running 队列，chunked prefill 切分长 prompt，decode 优先 + 剩余预算给 prefill 的 PD 混跑 |
| **Decode CUDA Graph** | 按 batch size 列表（1/2/4/8/16…）捕获 decode 子图，replay 消除 launch 开销；PP 走 piecewise（图外 P2P） |
| **Prefill torch.compile** | 首 chunk 全图编译，符号形状一次编译覆盖任意长度 |
| **多卡并行** | 张量并行 TP（列/行并行 + 词表并行）、流水线并行 PP（可不均分）、数据并行 DP（独立副本 + ZMQ）、专家并行 EP（复用 TP 组切专家） |
| **MoE 稀疏专家** | SparseMoeBlock 路由 + grouped GEMM（FlashInfer cutlass / 自研 Triton 核）+ 共享专家，MoE 进 CUDA Graph |
| **快速权重加载** | `skip_param_init` + 目标 dtype 直接构建 + safetensors 分片预取，QKV / Gate-Up 融合加载 |

## 架构总览

引擎采用「控制面 / 数据面 / 计算面」三面分离：

```
控制面 src/control/    Engine → EngineCoreClient → EngineCore → Executor → Worker
                       （请求门面 / 调度 / 执行桥 / 每卡 Worker）
数据面 src/data/       KV 账本（BlockManager / KVManager）、Sequence、批数据
                       （纯 CPU，不 import torch，管逻辑页号与前缀索引）
计算面 src/compute/    Worker / ModelRunner / KVPool / CUDA Graph / Sampler / 算子层 / 模型
                       （真正持有 GPU 权重与 KV 张量）
```

一步推理 = **调度 → 前向 → 采样 → 后处理**。请求生命周期、KV 分页、prefill/decode 双路径等详见 [docs/architecture.md](docs/architecture.md)。

## 快速开始

### 1. 安装

```bash
# Python 3.12 虚拟环境
uv venv .venv-py312 --python 3.12
source .venv-py312/bin/activate

# PyTorch（以 torch 2.13 / CUDA 13.0 为准）
uv pip install torch==2.13.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# FlashInfer（paged attention + cutlass MoE）
uv pip install flashinfer-python flashinfer-cubin \
    --index-url https://flashinfer.ai/whl \
    --extra-index-url https://pypi.org/simple
uv pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu130

# 其余依赖
uv pip install transformers huggingface_hub safetensors tqdm numpy \
    Pillow pyyaml einops xxhash triton pyzmq fastapi uvicorn pydantic
```

### 2. 下载模型

首次运行会自动从 HuggingFace 拉取权重；也可手动下载到 `~/huggingface/`：

```bash
hf download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B
```

默认模型注册表见 `utils/model_hub.py`（`QWEN3_REGISTRY`）。

### 3. 运行

```bash
# 单条推理（eager，启动快）
python run.py --model Qwen3-0.6B --prompt "Hello, my name is"

# 启用 Decode CUDA Graph（吞吐档）
python run.py --model Qwen3-0.6B --cg

# 批量推理演示
python run.py --model Qwen3-0.6B --batch

# 管理命令
python run.py --list          # 列出可下载模型与显存预算
python run.py --download      # 预下载全部模型
```

MoE 与多卡：

```bash
# Qwen3-MoE（单卡，自动选 cutlass / triton 后端）
python run.py --model ~/huggingface/Qwen1.5-MoE-A2.7B --cg

# 张量并行（机内多卡）
python run.py --model Qwen3-0.6B --tp 2 --cg

# 专家并行（TP 组上按 expert 切分，token 不搬家）
python run.py --model ~/huggingface/Qwen1.5-MoE-A2.7B --tp 2 --ep --cg

# 数据并行（独立副本 round-robin）
python run.py --model Qwen3-0.6B --dp 2 --batch --cg
```

常用参数：`--cg` `--compile` `--tp` `--pp` `--dp` `--ep` `--moe-backend` `--pp-layer-counts` `--device-ids` `--prefix-backend`。完整见 `config.yaml`。

## 性能

单卡（Qwen3-0.6B，4090 D，16 条变长请求 greedy，PyTorch 2.13 / FlashInfer 0.6.15 / Triton 3.7.1）：

| 档位 | 吞吐 (tok/s) |
|------|------------:|
| eager | 752.4 |
| cg（Decode CUDA Graph） | 2122.0 |

对比官方 vLLM 的基准脚本见 `bench/vs_vllm.py`，测试口径与完整结果见 [docs/performance.md](docs/performance.md)。

## 目录结构

```
.
├── run.py                 # 统一 CLI 入口（单条 / 批量 / MoE / 多卡）
├── run_torchrun.py        # 多机 torchrun 入口（TP × PP）
├── config.yaml            # 默认参数
├── model_hub.py           # 模型注册表 re-export
├── utils/                 # 共享工具（模型下载 / 配置 / 计时）
├── src/
│   ├── control/           # 控制面：Engine / EngineCore / Scheduler / Executor / Client
│   ├── data/              # 数据面：KV 账本 / Sequence / 批数据 / 前缀索引
│   ├── compute/           # 计算面：Worker / ModelRunner / KVPool / CUDA Graph / 算子层 / 模型
│   ├── moe/               # 稀疏专家：SparseMoeBlock / Experts / 后端 / 并行配置
│   ├── distributed/       # NCCL / Gloo / ZMQ 通信与进程组
│   └── config.py          # EngineConfig
├── bench/                 # 性能基准（本引擎 e2e / vs 官方 vLLM / TP-EP 矩阵）
├── tests/                 # 单元 / GPU / 分布式测试
└── docs/                  # 设计文档
```

## 文档

| 文档 | 内容 |
|------|------|
| [docs/architecture.md](docs/architecture.md) | 三面架构、请求生命周期、KV 分页、prefill/decode、CUDA Graph、权重加载 |
| [docs/moe.md](docs/moe.md) | 稀疏专家、路由与 grouped GEMM 融合、权重布局、后端选择 |
| [docs/parallelism.md](docs/parallelism.md) | TP / PP / DP / EP 与 NCCL·Gloo·ZMQ 通信 |
| [docs/performance.md](docs/performance.md) | 基准方法、本引擎与官方 vLLM 对比 |
| [docs/parallel-benchmark.md](docs/parallel-benchmark.md) | 两套机器多卡并行实测：Graph / compile 收益、TP / PP / DP 切分选择、与官方 vLLM 对照 |

## 测试

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.

pytest tests/unit -q                 # 无 GPU：路由 / 后端 / EP map
pytest tests/gpu -q --tb=short       # 单卡：dense greedy / fused vs naive / MoE 冒烟
pytest tests/distributed -q --tb=short   # 多卡：tp / tp+ep 贪心 token 一致
```

测试模型默认路径与覆盖矩阵见 [tests/README.md](tests/README.md)。

## 参考

- [vLLM](https://github.com/vllm-project/vllm)：PagedAttention 与 Continuous Batching 的参考实现
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) / [FlashInfer](https://github.com/flashinfer-ai/flashinfer)：paged attention 与 fused MoE kernel
- [Qwen3 / Qwen3-MoE](https://huggingface.co/Qwen)：目标模型
