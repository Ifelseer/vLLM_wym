# 文档

从零实现的 LLM 推理引擎（vLLM_wym）设计文档。建议按顺序阅读：先看整体架构，再分别看 MoE 与并行，最后看性能。

| 文档 | 内容 |
|------|------|
| [architecture.md](architecture.md) | 三面架构、请求生命周期、KV 分页与 paged attention、prefill/decode、CUDA Graph、权重加载 |
| [moe.md](moe.md) | 稀疏专家：路由、grouped GEMM 融合、权重布局、后端选择、CUDA Graph 集成 |
| [parallelism.md](parallelism.md) | TP / PP / DP / EP 切分方式与 NCCL·Gloo·ZMQ 通信 |
| [performance.md](performance.md) | 基准方法、本引擎端到端结果、与官方 vLLM 对比 |
| [parallel-benchmark.md](parallel-benchmark.md) | 两套机器上的多卡并行实测：CUDA Graph / compile 收益、TP / PP / DP 切分选择、与官方 vLLM 对照 |
| [fp8-debug-report.md](fp8-debug-report.md) | FP8 W8A8 落地调试：6 个缺陷的现象 / 判定依据 / 修复取舍，以及验收容差的定界过程 |

## 代码地图

| 目录 | 职责 |
|------|------|
| [`src/control/`](../src/control/) | 控制面：`Engine`（门面）→ `EngineCoreClient`（DP 分叉）→ `EngineCore`（核心循环）→ `Executor`（执行桥）→ `Worker`（每卡） |
| [`src/data/`](../src/data/) | 数据面：`KVManager` / `BlockManager`（KV 账本）、`Sequence`、批数据、前缀索引（hash / radix） |
| [`src/compute/`](../src/compute/) | 计算面：`ModelRunner`、`KVPool`（GPU KV 张量）、`CUDAGraphRunner`、`Sampler`、算子层、模型 |
| [`src/moe/`](../src/moe/) | 稀疏专家：`SparseMoeBlock`、`Experts`、`fused_moe_forward`（cutlass / triton）、`MoEParallelConfig` |
| [`src/distributed/`](../src/distributed/) | 进程组（TP/PP）、NCCL 集合通信、PP 点对点、控制通道 |

配图位于 [`posts/`](posts/)，均为本实现生成的示意图，可用 `python docs/posts/gen_moe_figs.py` 重新生成；[`figures/`](figures/) 存放 [parallel-benchmark.md](parallel-benchmark.md) 的实测图，用 `python docs/figures/gen_ieee_figs.py` 重新生成。

测试默认模型与覆盖矩阵见 [`tests/README.md`](../tests/README.md)；性能基准见 [`bench/README.md`](../bench/README.md)。
