# 用 nsys 定位量化推理的优化点：实测、方法与面试问答

本文记录一次真实的 Nsight Systems 采集与分析：Qwen3-0.6B、FP8 W8A8、batch-8、CUDA Graph。内容包括完整数据、根因定位链、按 ROI 排序的优化清单、可复用的 profiling 方法论与坑点，以及面试中可能被追问的问题与回答。

- [1] 结论摘要
- [2] 采集口径与命令
- [3] 第一手数据
- [4] 数据解读：bound 判断与 roofline
- [5] 优化优先级（按 ROI）
- [6] 定位优化点的标准流程
- [7] Profiling 坑点清单
- [8] 面试追问与回答
- [9] 后续可执行的 A/B 实验
- [10] 附录：可复现的查询

---

## 1. 结论摘要

1. **全进程 profile 的第一名是陷阱**：91.7% 的 GPU kernel 时间花在 `FillFunctor<int>`——它不是模型计算，而是 Triton autotune 的 `do_bench` 每轮清 256 MB L2 缓冲。剔除一次性成本后，稳态的 56% 才是 FP8 GEMM。
2. **最大优化点不是量化本身，而是小模型 decode 的 GEMM 并行度**：N=1024 的 o_proj/down_proj 在 decode 时只有 **8 个 CTA**（128 SM 的 4090 用不到 6%），这部分占 FP8 GEMM 时间的 55%。
3. **lm_head 未量化是一个确定但需谨慎的杠杆**：每步 311 MB、920 GB/s（已带宽饱和），0.6B 下占 decode 权重流量约 **43%**（Qwen3-0.6B 的 embed/lm_head 是 tied），8B 约 15%。
4. **autotune 的 5.5 s 不只是一次性慢**：线上遇到未预热的 (M,N,K) 会在请求路径上触发 `do_bench`，形成延迟尖刺。需要裁剪配置或按服务的形状分布预热。
5. ncu 在容器里被 `ERR_NVGPUCTRPERM` 挡住；用 nsys 的 grid/耗时 + 字节数算 achieved bandwidth 可以替代大部分判断。

## 2. 采集口径与命令

- 模型：Qwen3-0.6B（28 层，hidden 1024，32 q / 8 kv heads，head_dim 128，intermediate 3072）
- 量化：FP8 W8A8（per-block 128×128 权重 + per-token×128-group 动态激活）
- 负载：greedy，batch=8，CG（decode CUDA Graph），单卡 RTX 4090
- 工具：Nsight Systems 2025.6.3；CUDA 13.2；PyTorch 2.13

```bash
nsys profile -o nsys_fp8 --force-overwrite true \
    --trace=cuda,nvtx,osrt \
    --cuda-graph-trace=node \
    --sample=none --cpuctxsw=none \
    python bench/quant_bench.py --model ~/huggingface/Qwen3-0.6B --devices 0 \
        --quantizations fp8 --cg --batch 8 --n 16 --warmup 1 --reps 1
```

三个关键参数：

- `--cuda-graph-trace=node`：**必须**。否则 decode 的整张 CUDA Graph 在时间线上只是一个 `cudaGraphLaunch` 节点，看不到里面的 kernel，会误判"GPU 空转"。
- `--sample=none --cpuctxsw=none`：关闭 CPU 采样与上下文切换，降低开销与噪声。
- `--trace=cuda,nvtx`：CUDA 活动 + 可选的 NVTX 区间。

> 本文的数据来自**一次**包含模型加载、在线量化、autotune、warmup 与 16 步 decode 的完整进程。这也正是"全进程 profile 会被一次性成本污染"的现场教材。

## 3. 第一手数据

### 3.1 全进程 kernel 排名（陷阱现场）

| kernel | 时间占比 | 次数 | 平均 |
|---|---:|---:|---:|
| `at::native::vectorized_elementwise_kernel<4, FillFunctor<int>>` | **91.7%** | 20,253 | 273.6 µs |
| `_w8a8_block_fp8_kernel` | 8.0% | 28,016 | 17.2 µs |
| `flash_fwd_splitkv_kernel` | 0.1% | 616 | 5.6 µs |
| `FusedAddRMSNormKernel` | 0.03% | 1,344 | 1.6 µs |
| `_store_kvcache_kernel` | 0.01% | 672 | 1.0 µs |

同时 `cudaMemcpyAsync` 里有一次 40 ms 的 H2D（设备权重首次搬运），以及 HtoD 总计 171 ms——都在加载期。

### 3.2 根因定位链

1. **看时间线**：fill 集中在进程开始后的 9.1 s 内，最后一批是背靠背的 273 µs kernel（时间戳间隔 0.3 ms，持续 274 µs）。
2. **看 shape**：`torch.profiler(record_shapes=True)` 显示 `aten::zero_` 作用在 `[67108864]` 的 int32 上（256 MB），`group_by_input_shape` 汇总为 20,253 次、5.54 s。
3. **对源码**：`triton/backends/nvidia/driver.py:370-380`：

```python
def get_empty_cache_for_benchmark(self):
    cache_size = 256 * 1024 * 1024          # 256 MB
    return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')

def clear_cache(self, cache):
    cache.zero_()                            # 每次 do_bench 迭代前清 L2
```

4. **对上号**：FP8 kernel 的 `@triton.autotune` 对每个新的 `(M,N,K)` 跑 `do_bench`，每个候选配置在 warmup/rep 窗口内反复调用 kernel；**每次迭代前 zero 256 MB**。20,253 次 × 273 µs ≈ 5.5 s。

这就是端到端基准里 `Warmup: 1.42 s`（FP8）与 `0.09 s`（BF16）差异的来源。预热本身被排除在计时外（`drain_engine` 修复之后），但**如果线上出现未预热的形状，这部分成本会落在请求路径上**。

### 3.3 稳态构成（剔除 autotune 后）

过滤掉最后一个 fill 之后（约 18 次 decode forward + 若干 prefill），GPU kernel 共 42.4 ms：

| 组成 | 时间 | 占比 | 备注 |
|---|---:|---:|---|
| `_w8a8_block_fp8_kernel` | 23.71 ms | 56% | 2,238 次，平均 10.6 µs |
| lm_head（cutlass bf16 GEMM） | 6.76 ms | 16% | 20 次，每次 338 µs |
| attention（FA2 splitkv + combine） | 4.48 ms | 11% | 504 + 504 次 |
| 逐元素 copy | 2.58 ms | 6% | 1,732 次（约 3 次/层） |
| RMSNorm（含 fused_add） | 2.93 ms | 7% | 两条 kernel 各约 1,120 次 |
| RoPE / silu / KV store | 1.97 ms | 5% | 各约 560 次 |

### 3.4 FP8 GEMM 的并行度（关键表）

按 kernel 的 grid 维度（CTA 数）聚合稳态调用：

| CTAs | 调用数 | 平均耗时 | 对应层 | 每次权重字节 |
|---:|---:|---:|---|---:|
| **8** | 1,008 | **12.9 µs** | o_proj、down_proj（N=1024） | 1–3 MB |
| 24 | 111 | 13.2 µs | prefill 小形状 | — |
| 32 | 504 | 7.1 µs | qkv（N=4096） | 4 MB |
| 48 | 504 | 9.2 µs | gate_up（N=6144） | 6 MB |
| 96 / 144 | 110 | 7.4 / 9.9 µs | prefill 形状 | — |

0.6B 每层输出维：qkv 4096、o_proj 1024、gate_up 6144、down 1024。kernel 的 `BLOCK_N` 固定 128，所以：

- N=1024 → 8 个 N 块 → **8 个 CTA**；
- N=4096 → 32 个；
- N=6144 → 48 个。

4090 有 128 个 SM。decode 时 o/down 两层各自只用到 8 个 SM，**占用率 6%**；1,008 次调用合计 13.0 ms，占 FP8 GEMM 时间的 55%，但每次只搬 1–3 MB（约 80–230 GB/s）。

## 4. 数据解读：bound 判断与 roofline

### 4.1 先判断 decode 是带宽还是算力受限

算术强度（每读 1 字节权重能做的浮点运算）：

```
decode  batch B:  AI ≈ 2·P·B / weight_bytes
prefill 长度 T:    AI ≈ 2·P·T / weight_bytes ≈ T  (B=1 时)
```

- FP8、8B、batch 8：`AI ≈ 16 FLOP/Byte`，远低于 4090 的 ridge point（~100+ FLOP/Byte）→ **带宽受限**。
- 同一模型 prefill T=2048：`AI ≈ 4096 FLOP/Byte` → **算力受限**。

这解释了两个现象：

- **TTFT 几乎不受权重量化影响**（prefill 算力受限，权重复用 2048 次）；
- **decode 加速比 ≈ 权重字节比**：8B 从 15.26 GiB 降到 8.79 GiB，理论上限 1.74×，实测 1.63×（达到 94%）。

### 4.2 为什么 0.6B 的 FP8 几乎不提速

单流 0.6B 的实测：FP8 decode 430.8 tok/s vs BF16 440.2 tok/s（基本持平）；而 8B batch-8 是 1.63×。原因有三层：

1. **lm_head 占比高**：0.6B 的 lm_head 是 311 MB bf16（tied embed，155.58 M 参数），每步都读；FP8 量化后层权重只有约 411 MB，lm_head 占 decode 权重流量的 **43%**（8B 约 15%）。不量化 lm_head，理论加速比只有 `1133/722 ≈ 1.57×`（未计 attention/KV）。
2. **GEMM 并行度不足**：见 §3.4，N=1024 的层只有 8 个 CTA，达不到带宽峰值，实际只有 ~80–230 GB/s。
3. **固定开销**：attention、norm、RoPE、copy 在 0.6B 的每步时间中占比更高（8B 的 GEMM 时间更长，能把它们摊薄）。

### 4.3 lm_head 为什么已经"没得优化"

lm_head `[151936, 1024]` bf16 = 311 MB，单次 338 µs → **920 GB/s**，接近 4090 的 HBM 峰值（~1 TB/s）。它已经不是效率问题，而是字节问题；唯一的杠杆是量化（减半），代价是 logits 精度。

## 5. 优化优先级（按 ROI）

| 优先级 | 优化 | 依据 | 预期收益 | 风险/代价 |
|---|---|---|---|---|
| P0 | **FP8 decode 细分 N tile**：autotune 增加 `BLOCK_N=64/32`，或对 N=1024 用 split-K | 8 CTA × 12.9 µs × 1008 次 = GEMM 时间的 55% | 0.6B decode +15–25%（估计） | 更小的 tile 增加循环与 scale 加载次数；split-K 需要额外归约 |
| P1 | **量化 lm_head**（per-channel int8/fp8） | 每步 311 MB、占 0.6B decode 权重 43% | 权重流量 722→566 MB（−22%），预计 decode +10–20% | logits 精度敏感；**tied embedding（0.6B）下 lm_head 与 embed 共享权重**，需同时决定 embedding 的处理；8B 是 untied，较简单 |
| P1 | **autotune 抖动治理**：裁剪 configs、decode 固定 tile、按形状分布预热 | 5.5 s 一次性成本，新形状会落在请求路径 | 消除 p99 尖刺 | 放弃部分形状的最优 tile |
| P2 | **消除 attention 的 `.contiguous()`** | 1,732 次 copy、2.58 ms（6%） | decode +3–5% | 需要改 RoPE 的输出布局 |
| P2 | **核对 norm 路径** | `FusedAddRMSNorm` 与 `RMSNorm` 各约 1,120 次 | 潜在 2–4% | 需要确认是否重复 add/norm |
| P3 | **FP8 KV 的每步 plan** | decode −6%（相对 BF16 KV） | 若开 KV 量化，可回到 −2% 以内 | 需固定 plan 或自写 decode kernel |
| P3 | **MoE**（待 checkpoint） | 未测 | 未知 | grouped GEMM 的 block/naive 阈值需实测 |

> 注意优先级不是按"占比"排的：attention 占 11%，但 FA2 已经很成熟；FP8 GEMM 占 56%，其中 55% 又集中在 8-CTA 的层上——**占比 × 可改进空间**才是 ROI。

## 6. 定位优化点的标准流程

1. **先定性（5 分钟）**：算 roofline，确认是带宽还是算力受限；确认瓶颈在 prefill 还是 decode。
2. **再看时间线（nsys）**：
   - 剔除一次性成本（autotune/JIT/capture/首次 cuBLAS），只看稳态窗口；
   - 按 kernel 总时长排序；看 GPU busy%、kernel 间的 gap、memcpy、同步；
   - CG 必须 `--cuda-graph-trace=node`。
3. **对 top kernel 做算术**：从 nsys 拿 grid/block/耗时，从代码拿 tile 配置，算 CTA 数、每 CTA 读取字节、achieved bandwidth；判断"带宽不够"还是"并行度不够"。
4. **ncu 下钻（能用的环境）**：DRAM throughput、tensor active%、occupancy、L2；确认 memory-bound / latency-bound。
5. **一次只改一个变量**，同 bench（同 prompt/batch/预热/CG 口径）复测至少 3 次；看中位数与 p99，不只看均值。
6. **回到端到端**：kernel 省下的时间是否真的体现在 TTFT/decode？如果 kernel 之和 << 墙钟，问题在调度/同步/launch。

## 7. Profiling 坑点清单

1. **CUDA Graph 默认是一个节点**。不加 `--cuda-graph-trace=node` 会把整个 decode 看成一次 launch，误判 GPU 空闲。
2. **一次性成本污染全进程 profile**。JIT、autotune、`torch.compile`、CG capture、首次 cuBLAS handle 初始化都会进入排名；用 nsys capture range 或时间窗口过滤。本文 91.7% 的 fill 就是典型。
3. **ncu 权限**：容器里常见 `ERR_NVGPUCTRPERM`（本次实测踩到）。需要宿主机放开 perf counter 权限；否则退回 nsys + 算术 + 微基准。
4. **平均会骗人**：autotune 的 fill 平均 273 µs、稳态 GEMM 17 µs，混在一起会得出完全错误的结论。看中位数、分位数、分阶段。
5. **同步点表现为 GPU 空闲**：每步 `plan` 的 D2H、`.item()`、pinned buffer 同步，在时间线上是 gap，不是 kernel 慢。
6. **多进程/并行**：TP/PP/DP 下用 `--trace-fork-before-exec=true` 或 `nsys launch`；NCCL 通信与计算重叠，不能只看 compute kernel；按 rank 分开看。
7. **时钟与功耗**：4090 会 boost/throttle；对比前后要锁频或多跑几轮。单次 TTFT 有 5–10% 晃动。
8. **别用 wall-clock 反推 kernel 占比**：kernel 时间之和远小于墙钟时，问题在 launch/调度/同步——这正是 eager 与 CG 的分野（同一量化配置，CG 下 decode 1.63×，eager 下 0.78×）。
9. **基准口径不一致**：prompt 长度、batch、`max_num_seqs`、KV blocks、prefix cache、是否 CG，任何一个不同都会让数字不可比。

## 8. 面试追问与回答

**Q1：nsys 和 ncu 的区别？**
nsys 是系统级时间线：CPU/GPU 活动、kernel 排布、memcpy、同步、CUDA Graph 结构，开销小，用来找"哪里花时间"；ncu 是单 kernel 的硬件计数器：DRAM/L2 吞吐、tensor core 利用率、occupancy、指令统计，用来解释"为什么慢"。流程是先 nsys 定位，再 ncu 下钻。

**Q2：CUDA Graph 下怎么 profile？**
nsys 加 `--cuda-graph-trace=node` 把图展开成节点；ncu 也需要 node 级图追踪（或对捕获前的 eager 版本做单 kernel 分析）。否则只能看到一个 graph launch。

**Q3：为什么 autotune 不在稳态里？线上怎么办？**
autotune 按 `(M,N,K)` 缓存，首次遇到某个形状才跑 `do_bench`。我们把它放在 warmup 里，所以计时排除了它。但线上如果出现没预热过的 batch size / prompt 长度，会在请求路径上跑，形成 1 秒级尖刺。治理：裁剪候选配置、对 decode 固定 tile（INT4 就是这么做的）、按服务的形状分布做 warmup，或把 autotune 结果持久化。

**Q4：怎么判断 memory-bound 还是 compute-bound？**
算算术强度并与 GPU 的 ridge point 比。decode 时权重只复用 batch 次，AI 只有十几 FLOP/Byte，远低于 4090 的 ~100，所以带宽受限；prefill 时同一权重被 T 个 token 复用，AI 上千，所以算力受限。实测也吻合：量化权重后 decode 提速接近字节比，TTFT 几乎不变。

**Q5：为什么 0.6B 的 FP8 不提速？**
三层原因：lm_head 未量化且占 decode 权重流量 38%；N=1024 的层只有 8 个 CTA，带宽打不满；attention/norm 等固定开销在小模型里占比更高。8B 的层更大、lm_head 占比更低，所以能到 1.63×。

**Q6：为什么先优化 GEMM 的 tile，而不是先量化 lm_head？**
因为数据：8-CTA 的 GEMM 占 FP8 GEMM 时间的 55%，且 achieved bandwidth 只有峰值的 8–25%，改进空间大；lm_head 已经 920 GB/s，贴着峰值，只能靠量化减字节，而量化 lm_head 有精度风险。ROI 上先做前者。

补充一个容易踩的点：Qwen3-0.6B 的 `tie_word_embeddings=true`，lm_head 与 `embed_tokens` 是同一份权重。量化它不只是"多量化一个 Linear"——embedding lookup 也需要决定是保留一份 bf16 副本（那 311 MB 就省不掉），还是做量化 embedding（查表后反量化，多一次 kernel）。8B 是 untied，可以直接量化 lm_head。

**Q7：BLOCK_N 减小 / split-K 的代价？**
BLOCK_N 变小会增加每个 CTA 的循环次数与 scale 加载次数（per-block scale 按 N/128 索引，N tile 小于 128 时同一 scale 被多个 CTA 重复读）；split-K 需要额外的部分和缓冲与归约 kernel，且会改变数值累加顺序。需要实测权衡，不能只看 CTA 数。

**Q8：为什么不用 torch.compile 解决 launch 开销？**
compile 主要优化 prefill 的图融合与 kernel 选择；decode 的 launch 开销用 CUDA Graph 解决更直接。而且 FP8 + prefill compile 在我们这里显式禁用（Dynamo × Triton autotune 未验证），INT4 也因为 autotune 不能在 capture 期间做而选择固定 tile。

**Q9：怎么确认优化是真实的，不是噪声？**
固定所有变量（同 prompt/batch/KV/预热），同一天同一张卡，至少 3 次重复看中位数；用 CUDA events 而不是纯 wall clock；对 p99 单独看。GPU 会 boost/throttle，必要时锁频。eager 与 CG 的结论可能相反，必须注明口径。

**Q10：多卡怎么 profile？**
nsys 用 `--trace-fork-before-exec=true` 或 `nsys launch` 跟随子进程，按 rank 分别看；NCCL 通信和计算重叠，要同时看 NCCL kernel 和 gap；TP 的有效带宽收益取决于消息大小（本项目 0.6B 每步约 57 次 2 KB all-reduce、8B 约 73 次 8 KB，无 NVLink 时通信 ~5 ms/token，直接吃掉 decode 收益）。

**Q11：没有 ncu 权限怎么办？**
用 nsys 的 grid/block/耗时 + 代码里的 tile 配置算 CTA 数、每 CTA 字节、achieved bandwidth；再用微基准（单独跑该 kernel、扫 M/N/K）补足。本文的 P0 结论就是这样得出的。

**Q12：如果换到 H100 会怎样？**
H100 有 SM90 的原生 block-scaled FP8（cutlass），可以替掉自写 Triton per-block kernel；HBM 带宽更高、SM 更多，8-CTA 的并行度问题会缓解但不会消失（N=1024 仍是 8 个 CTA）；同时 ridge point 更高，decode 更偏带宽。优化顺序会变成：先换 cutlass block-scaled kernel，再看 lm_head 与 tile。

**Q13：怎么发现"请求级"的性能问题，而不只是平均吞吐？**
看 p50/p95/p99 与分阶段指标：TTFT 分布、每步 decode 时间、排队时间。常见来源是 autotune 新形状、KV 池不足导致的调度空转、前缀缓存未命中、以及每步 plan/同步。nsys 抓单请求的时间线可以看到 gap。

**Q14：为什么 FP8 KV 会慢 6%？**
FlashInfer paged attention 相对 FA2 多了一次每步的 `plan`（内含 D2H 同步），以及 fp8→bf16 的反量化 epilogue；在 384-token 上下文下 KV 流量占比小，省下的带宽抵不过通路开销。所以 FP8 KV 是容量手段（同显存 blocks 翻倍），不是吞吐手段。

## 9. 后续可执行的 A/B 实验

1. **FP8 decode tile**：`_block_configs` 增加 `BLOCK_N∈{64,32}`；对 N=1024 的层单独测；同 bench 对比。
2. **split-K**：只对 N≤1024 的层启用，K 方向切 2–4 份，测 CTA 数与归约开销。
3. **lm_head 量化**：per-channel int8（`w8a8_int8_matmul` 已有现成路径），验证 teacher-forced top-1/logit 偏差与 decode 吞吐。
4. **autotune 治理**：把配置裁剪到 1–2 个、按 decode 形状固定 tile，测 warmup 时间与 p99。
5. **attention 布局**：让 RoPE 直接输出连续布局，消掉 3 次/层的 `.contiguous()`。
6. **norm 合并**：确认 `FusedAddRMSNorm` 与 `RMSNorm` 是否重复，合并后测 2–4% 的收益。

## 10. 附录：可复现的查询

```bash
# 1) 顶层 kernel 排名
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum nsys_fp8.nsys-rep

# 2) 找出 fill 的 grid 维度与时间跨度（sqlite）
sqlite3 nsys_fp8.sqlite "
SELECT k.gridX,k.gridY,k.gridZ,k.blockX,COUNT(*),AVG(k.end-k.start)/1000.0
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
WHERE s.value LIKE '%FillFunctor%' GROUP BY 1,2,3,4 ORDER BY 5 DESC;"

# 3) 剔除 autotune 后的稳态构成
LAST=$(sqlite3 nsys_fp8.sqlite "
SELECT MAX(k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName=s.id WHERE s.value LIKE '%FillFunctor<int>%';")
sqlite3 nsys_fp8.sqlite "
SELECT ROUND(SUM(k.end-k.start)/1e6,2) ms, COUNT(*), s.value
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
WHERE k.start > $LAST GROUP BY s.value ORDER BY ms DESC LIMIT 12;"

# 4) 按 CTA 数看 FP8 GEMM 的并行度
sqlite3 nsys_fp8.sqlite "
SELECT k.gridX, k.blockX, COUNT(*), AVG(k.end-k.start)/1000.0
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
WHERE k.start > $LAST AND s.value='_w8a8_block_fp8_kernel'
GROUP BY 1,2 ORDER BY 1;"
```

> 说明：本文数据来自单次采集（0.6B、FP8、batch-8、CG、n=16）；数值用于说明方法与相对关系，不应当作绝对性能基准。绝对性能与显存对比见 [`quantization-bench.md`](quantization-bench.md)。
