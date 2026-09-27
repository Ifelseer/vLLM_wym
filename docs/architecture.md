# 架构总览

本文描述引擎的整体架构：三面分离、一步推理的完整链路、KV 分页与 paged attention、prefill/decode 双路径、Continuous Batching 调度、Decode CUDA Graph 与权重加载。

- [1] 三面分离
- [2] 一步推理的完整链路
- [3] 控制面：分层与 DP 分叉
- [4] 数据面：KV 账本与前缀索引
- [5] 计算面：Worker 与 ModelRunner
- [6] KV Cache 与 PagedAttention
- [7] prefill / decode 双路径
- [8] Continuous Batching 调度
- [9] Decode CUDA Graph
- [10] 采样
- [11] 权重加载

## 1. 三面分离

引擎按「控制面 / 数据面 / 计算面」三层解耦，各层职责与依赖方向固定：

| 面 | 目录 | 职责 | 是否碰 GPU |
|----|------|------|-----------|
| 控制面 | `src/control/` | 请求门面、调度、执行桥 | 否（只发指令） |
| 数据面 | `src/data/` | KV 逻辑账本、Sequence、批数据 | 否（`KVManager` 不 import torch） |
| 计算面 | `src/compute/` | 模型权重、GPU KV 张量、前向、采样 | 是 |

控制面拿到一个批（`ScheduledBatch`），序列化后交给计算面执行；计算面把采样结果回传，控制面据此推进序列状态。数据面夹在中间：控制面用它在 CPU 侧管理「逻辑页号 → 物理 slot」的映射，计算面按这些映射读写 GPU 上的 KV 张量。

## 2. 一步推理的完整链路

引擎对外暴露 `step()`，每一步都是：

```
调度 schedule → 执行 run → 读回 read_samples → 后处理 update
```

1. **调度**：`Scheduler.schedule()` 从 waiting / running 队列选出本步的 prefill 与 decode 序列，产出 `ScheduledBatch`。
2. **执行**：`Executor.run()` 把 `ScheduledBatch.to_wire()` 得到的 `SchedulerOutput` 交给 `Worker.execute_model()`，由 `ModelRunner` 完成前向，采样结果异步 D2H。
3. **读回**：`Executor.read_samples()` 取回采样出的 token id。
4. **后处理**：`Scheduler.update()` 逐序列追加 token、判断是否结束、回收已完成序列的 KV 块，产出 `EngineCoreOutput`。

`EngineCore.step()` 是这一循环的驱动者，位于 `Engine` 与 `Executor` 之间。

## 3. 控制面：分层与 DP 分叉

```
Engine (src/control/engine.py)
  │  add_request / step / generate_batch / generate
  ▼
EngineCoreClient (src/control/client.py)      ← DP 分叉点
  │  dp=1 → InprocClient（同进程直接转发）
  │  dp>1 → MPClient（ZMQ 路由到多个副本子进程）
  ▼
EngineCore (src/control/core.py)              ← 核心循环
  │  step = schedule → run → read_samples → update
  ▼
Executor (src/control/executor.py)            ← 执行桥
  │  tp×pp=1 → UniprocExecutor（同进程直接调用）
  │  否则    → MultiprocExecutor（spawn 子进程 + Gloo 广播）
  ▼
Worker (src/compute/worker.py)                ← 每卡一个，持有模型/KV/采样器
```

分层动机：**DP 的分叉只在 Client 层发生，TP/PP 的多进程展开只在 Executor 层发生**。`Engine` 不感知任何并行策略，同一套接口覆盖单卡与多卡。

- `dp=1`：`Engine` 与 `Scheduler` 共享同一个 `Sequence` 对象，调度器原地更新，门面侧无需同步。
- `dp>1`：每个副本是一个独立进程（各有一份 Scheduler），`MPClient` 用 ZMQ 把请求 round-robin 到副本，门面侧持有镜像 `Sequence`，根据 step 返回的增量手动推进。

`Executor` 的 `MultiprocExecutor` 中，rank0 同时扮演 driver 与本地 Worker：通过 Gloo（CPU 后端）把 `ControlCommand`（execute / reset_kv / shutdown）广播给所有 rank，各 rank 独立前向，采样结果由采样 rank 回传 rank0。采样固定落在末 PP stage 的 tp0（只有它有完整 logits）。

## 4. 数据面：KV 账本与前缀索引

数据面不碰 CUDA，只维护「逻辑页号」与序列状态：

- **`Sequence`**（`src/data/sequence.py`）：一条请求的完整状态——token ids、`block_table`（逻辑页表）、已缓存 token 数、本步调度 token 数、采样参数等。三种状态 `WAITING / RUNNING / FINISHED`。
- **`BlockManager`**（`src/data/block_manager.py`）：物理块池的三态管理——`free_ids`（空闲）、`used_ids`（使用中）、`cached_ids`（前缀缓存暂留，LRU）。缺页时从 cached 淘汰或报错。
- **`KVManager`**（`src/data/kv.py`）：调度器与 block 池之间的入口，提供 `can_allocate / allocate / may_append / deallocate`，并在分配时做前缀复用预检查（`_fit_prefix`）。
- **前缀索引**（`src/data/prefix/`）：两种后端——`hash`（链式 hash 字典）与 `radix`（压缩前缀树），`none` 为空实现。`sync_prefix` 在每步把写满的新块登记进索引，供后续请求命中。

`KVManager` 与 GPU 上真正的 KV 张量是分离的：前者管账，后者（`compute/kv_pool.py`）管显存，两者通过「块号」一一对应。

## 5. 计算面：Worker 与 ModelRunner

**`Worker`**（`src/compute/worker.py`）是每卡的入口：

- `init_environment()`：设 CUDA 环境（FlashInfer arch 列表、NCCL 参数）、初始化分布式环境与 TP/PP 进程组。
- `initialize()`：加载模型 → 建采样器 → 建 KV 池 → 建 `ModelRunner`。
- `_make_kv()`：按 `num_kvcache_blocks` 或显存预算自动计算块数；开 CG 时先探测 CG footprint 再倒推剩余显存给 KV。

**`ModelRunner`**（`src/compute/model_runner.py`）负责单步前向编排：

- 持有常驻 GPU buffer（`Batch`），`build_decode` / `build_prefill` 填充输入。
- `run_forward`：先 decode 再 prefill，PP 为 lockstep 切层（非 1F1B）；非末 stage 只做本段计算 + P2P 发送，末 stage 异步 D2H 采样 id。
- 开 CG 时 decode 走 `CUDAGraphRunner.replay`，否则 eager 直跑。

**`KVPool`**（`src/compute/kv_pool.py`）在 GPU 上分配 KV 张量，形状 `[2, L, num_blocks, block_size, n_kv, head_dim]`（2 为 K/V，L 为本 PP stage 层数，NHD 布局对齐 FlashAttention paged）。`auto_num_blocks` 按 `gpu_memory_utilization` 与 CG 预留算块数，并跨 TP rank 取最小值。

## 6. KV Cache 与 PagedAttention

KV cache 以固定大小 block（`kvcache_block_size`，默认 256，FA2 paged 要求 256 的倍数）分页：

```
逻辑 token i  →  逻辑块 b = i // block_size，块内偏移 off = i % block_size
物理 slot     =  block_table[b] * block_size + off
```

- **写**：prefill 时把新计算的 K/V 按 `slot_mapping` 散射写入对应物理 slot。
- **读**：attention 拿着 `block_table`（页表）直接在 paged 布局上计算，不需要先把历史 KV gather 成连续张量。

注意力计算用 FlashAttention paged kernel：

- prefill：`flash_attn_varlen_func(block_table=...)`，varlen paged，内核内因果 mask，首 chunk / 续 chunk 统一处理。
- decode：`flash_attn_with_kvcache`，一次 kernel 处理整批 decode，GQA 在内核层处理（不材料化 K/V）。

因为 KV 布局 `[2, L, blocks, 256, n_kv, hd]` 直接对齐 paged attention，写路径用 Triton fused scatter（`src/compute/kv_ops.py`）替代 `index_copy_`。

## 7. prefill / decode 双路径

- **prefill**：读入 prompt 的所有 token，一次性计算，产出自回归所需的 KV；长 prompt 被 chunked prefill 切成多块，只有最后一块覆盖完整 prompt 时触发采样（`need_sample`）。
- **decode**：每个运行中的序列只推进一个 token，只对最后一个位置做 attention。

两者在同一批里合跑（PD 混跑）：`ModelRunner.run_forward` 先 `_run_decode` 再 `_run_prefill`。prefill 可选走 `torch.compile` 编译的 `prefill_fn`（符号形状，一次编译覆盖任意长度）。

## 8. Continuous Batching 调度

`Scheduler` 维护 `waiting` / `running` 两个队列，`mix_prefill_decode=True` 时：

1. **decode 优先**：从 running 队列拉 decode 序列，每个占 1 个 token 预算，直到 `max_num_batched_tokens` 或 `max_num_seqs` 打满。
2. **剩余预算给 prefill**：从 waiting 队列拉新请求，chunked prefill 切成不超过剩余预算的块；一个 chunk 若未覆盖完整 prompt，则 `need_sample=False`、留在 waiting。
3. KV 不够时**抢占**（`preempt`）：把 running 里最晚的序列还页、退回 waiting。

`mix_prefill_decode=False` 则退化为主 prefill 次 decode 的串行。每步 metric = prefill token 数 + decode 请求数，供吞吐统计。

## 9. Decode CUDA Graph

`CUDAGraphRunner` 按一组 batch size（`[1, 2, 4, 8, 16, ...]`）捕获 decode 子图，实际 batch 向上对齐到最近的捕获点后 replay，消除 decode 阶段反复的 kernel launch 开销。

- **capture**：先 warmup 一次初始化算子与 NCCL 集合通信顺序，再 `torch.cuda.graph` 录图；`graph_capture` 上下文在录图前后对 TP/PP 做 barrier，保证各 rank 步调一致。
- **piecewise**：PP 的 recv/send 在图外，图内只做本地 forward 读写静态 buffer，P2P 更稳、调度与图解耦。
- **MoE 进图**：align / silu / sum / GEMM 的 workspace 按容量池化，token 数按 captured BS 垫齐，路由内容可变但 launch 形状不变，因此可整图 replay（详见 [moe.md](moe.md)）。

## 10. 采样

`Sampler`（`src/compute/sampler.py`）做批量采样，全程无 GPU→CPU 同步：

- 全 greedy：`logits.argmax`。
- 仅温度：Gumbel-max 采样（GPU 端）。
- top-k / top-p / min-p：在降序候选空间内做截断后 `multinomial`。
- 混合 greedy + 采样：标量开关 `any_greedy` / `use_top_k` 等在 CPU 侧打包时判定，采样器内用 `torch.where` 覆盖。

decode 的 batch size 随请求完成动态递减，因此采样器不做 `@torch.compile`（避免按形状特化导致每步重编译）。

## 11. 权重加载

- `skip_param_init` + 目标 dtype 直接构建 + safetensors 分片预取，跳过随机初始化与逐分片重排，显著缩短加载时间。
- 融合加载：HF 的 `q/k/v_proj` 融合进 `qkv_proj`，`gate/up_proj` 融合进 `gate_up_proj`，由各并行层的 `weight_loader` 按 TP rank 切分写入。
- PP 层索引过滤：非本 stage 的层权重直接跳过，本 stage 内层号重映射为局部下标。
- 权重绑定：`lm_head` 未显式加载时复用 `embed_tokens` 权重（跳过 padding 行）。
