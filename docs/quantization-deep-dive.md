# 量化实现全解析：从请求到 kernel

本文以代码为准，完整串一遍量化在本引擎中的实现：设计分层、一次请求从入队到出 token 的每个环节、五种量化方案（FP8 / INT8 / INT4 / MoE / FP8 KV）各自的通路、所有优化点的动机，以及调试过程中每个错误的发现方式与修复理由。性能与显存实测见 [`quantization-bench.md`](quantization-bench.md)，验收口径见 [`quantization.md`](quantization.md)。

- [1] 设计总览
- [2] 请求运行视角：一次请求的完整生命周期
- [3] FP8 W8A8（dense）
- [4] INT8 W8A8（dense）
- [5] INT4 weight-only（AWQ / GPTQ dense）
- [6] MoE 专家量化（FP8 / INT4）
- [7] FP8 KV cache
- [8] 优化清单
- [9] 错误复盘：怎么看出来、为什么这么改
- [10] 验收与实测结果
- [11] 已知限制

---

## 1. 设计总览

### 1.1 四层结构与依赖方向

```
src/quant/config.py      QuantConfig：格式解析、硬件门控、modules_to_not_convert
src/quant/methods/*      QuantMethod：参数创建、weight_loader、finalize、forward 分派
src/quant/kernels/*      kernel：Triton / torch 原语（FP8 / INT8 / INT4）
src/compute/layers/*     层：持有参数、决定 TP 通信；只做薄委托
```

依赖方向是单向的：`compute/*` 与 `moe/*` 可以依赖 `quant/*`，`quant/*` 不反向依赖计算层。这带来两个直接好处：

1. **量化逻辑集中**。层代码里只有 `if self.quant_method is not None: return self.quant_method.forward(x, self)`（如 `linear.py:85`），TP 语义（all-gather / all-reduce / input_is_parallel）与量化完全正交，新增量化方法不需要改任何层。
2. **注册表懒加载**。`methods/__init__.py::make_method` 按 method 名 `importlib.import_module`，未实现的格式报 `NotImplementedError` 而不是拖垮整个包导入；`awq` / `gptq` 共用 `int4.py`，在模块内按 `quant.method` 二次分派。

模型侧通过 `quant_for(quant, prefix)` 做**按参数名前缀**的过滤（`qwen3.py:55/66/101/115`）：命中的模块不量化。匹配是点分段感知的连续子序列（`config.py:77`），所以 `mlp.gate` 不会误伤 `mlp.gate_up_proj`，而 `lm_head` 仍能命中 `lm_head.weight`。

### 1.2 方案矩阵

| 方案 | 量化对象 | 权重粒度 | 激活 | 硬件门控 |
|---|---|---|---|---|
| `fp8` | dense 线性层 + MoE 专家 + KV cache | per-tensor 或 per-block 128×128 | per-token 动态（per-block 时按 128-group） | SM89+ |
| `int8` | dense 线性层 | per-channel `[out]` | per-token 动态 | SM61+ |
| `awq` / `gptq` | dense 线性层 + MoE 专家 | group=128，INT4 weight-only | 保持 bf16 | 任意（Triton dequant-GEMM） |
| `auto` | 按 checkpoint 的 `quantization_config` | — | — | 同对应方案 |

门控原则是**显式拒绝而非静默降级**（`config.py:309`）：SM86 上请求 fp8 会直接报错，避免"以为量化了、实际跑 bf16"的隐性混合路径。INT8 对 MoE 专家显式 `NotImplementedError`（`experts.py:135`），同样的理由。

### 1.3 三条贯穿全局的设计原则

1. **一次性成本放加载期，热路径只留最少的算子。** AWQ 的交织/转置、GPTQ 的打包轴转换、零点解包都在加载期完成（`int4_gemm.py:126` 的 `normalize_int4_to_in_major`），kernel 里只剩"读 packed int32 → 移位 → 乘 scale"。
2. **能融合就融合。** INT8 的 per-token 量化做成单个两遍过 K 的 Triton kernel；FP8 的激活量化直接融进 GEMM 的 k-block 循环；INT4 的解包融进 dequant-GEMM。理由统一：小 kernel 多、对 `[M,K]` 多次全量读写会把带宽吃光（INT8 注释里实测 165 µs vs 20 µs）。
3. **与 TP / PP / CG 正交。** `QuantMethod.forward` 不做通信；scale 随参数一起按 TP 规则切分；CUDA Graph 路径里禁止 autotune 与 host 同步（INT4 固定 tile、FP8 KV 的 plan 放图外）。

---

## 2. 请求运行视角：一次请求的完整生命周期

量化在"启动期"和"运行期"的介入点完全不同。先建立全局图，再逐方案展开。

### 2.1 启动期（只发生一次）

```
EngineConfig(quantization=…)
  └─ QuantConfig.from_name / from_hf_config        # 解析格式、硬件门控
       └─ Worker._load_model
            ├─ 构造模型：每个 Linear 层调 get_quant_method(...)
            │    └─ method.create_weights(layer)   # 建量化参数 + 绑 weight_loader
            ├─ load_model(...)                     # 内部 = load_weights + finalize_model
            │    ├─ load_weights：逐张量路由到 weight_loader
            │    │    ├─ 预量化 checkpoint：int8/fp8 直接切片写入
            │    │    └─ 在线量化：先写 CPU staging（bf16）
            │    └─ finalize_model：staging → 量化 → 释放
            └─ Worker._to_device(model)            # 只迁设备/转普通参数，保护量化参数
  └─ Worker._make_kv → KVPool(kv_dtype=…)
  └─ ModelRunner.__init__
       ├─ FP8 KV：Fp8KVManager + _calibrate_fp8_kv()   # 录图前校准 scale
       └─ CUDAGraphRunner.capture(...)                  # 按 bs 捕获 decode 图
```

关键点：

- **参数创建与权重加载解耦**。`create_weights` 先把参数按目标 dtype 建好（fp8 / int8 / int32 packed），并给每个参数挂 `weight_loader`。模型 `load_weights` 的既有路由（按参数名查表）不需要知道量化细节，拿到的 loader 已经是量化方法提供的 partial（`fp8.py:120-125`）。
- **在线量化的 staging 在 CPU**（`fp8.py:191`、`int8.py:174`）。safetensors 迭代器本身产出 CPU 张量，staging 顺水推舟留在 CPU：这样 GPU 上只驻留"量化后的权重"，不会在加载期同时存在 bf16 全量副本。8B 模型在线量化时 CPU 需要约 14 GB 额外内存（6.95 B 个被量化参数 × 2 B），但 GPU 峰值不翻倍——这是能在一张 24 GB 卡上在线量化 8B 的前提。
- **`finalize_model` 在全部 shard 到齐后运行**（`quant/__init__.py:57`）。原因：列并行层按输出维切分、行并行层按输入维切分，同一个 fused 参数（qkv / gate_up）由多个 checkpoint 张量拼成；必须等所有段写完才能算 per-block scale。
- **`_to_device` 不能整体 `.to(dtype)`**（`worker.py:147`）。量化参数（fp8 权重、fp32 scale、int32 packed）由方法自己管理 dtype，整体转换会把它们一并转成激活 dtype。实现上按 `quant_method.extra_params()` 的归属判断：属于量化方法的参数保持原 dtype，其余浮点参数转激活 dtype。
- **FP8 KV 的 scale 必须在录图前校准并冻结**（`model_runner.py:_calibrate_fp8_kv`）。捕获后的图里不能再写 scale，否则 replay 用新数据重算会和存量 KV 的写/读 scale 不一致。

### 2.2 运行期（每个 step）

```
Engine.step
 └─ EngineCore.step
     ├─ Scheduler.schedule()                    # waiting/running → prefill/decode 批
     ├─ Executor.run(sched) → Worker.execute_model
     │   └─ ModelRunner.run_forward
     │       ├─ _run_decode(decodes)            # 先 decode（CG replay 或 eager 前向）
     │       └─ _run_prefill(prefills)          # 再 prefill（varlen）
     │           每层：
     │             hidden = norm(hidden)
     │             attn_out = self_attn(q,k,v, attn_ctx)   # attention 自定义算子
     │                 └─ 写入 KV（bf16 或 fp8）→ paged attention
     │             hidden = hidden + attn_out
     │             hidden = hidden + mlp(hidden)
     │                 └─ quant_method.forward → 量化 GEMM
     │           末层：lm_head（不量化）→ argmax / sampler
     ├─ _stage_samples：采样 id 经 pinned buffer 异步 D2H
     └─ read_samples → Scheduler.update（追加 token、判结束、回收 KV）
```

量化在这一步的接入点只有两个：

1. **每个 Linear / Expert 的 `quant_method.forward`**：把 `x` reshape 成 2D，调用量化 GEMM，再 reshape 回去。层的 TP 通信（all-gather / all-reduce）在方法返回之后进行，与量化无关。
2. **attention 的 KV 写入/读取**：`fp8_kv` 非空时，写 KV 用 `store_kvcache_fp8`（量化写入），读用 FlashInfer paged attention（带 `k_scale/v_scale`）。

另外两个与量化间接相关的引擎级优化：采样 id 的**异步 D2H**（`_stage_samples` 在独立 copy stream 上拷到 pinned memory，`read_samples` 再同步，CPU 不阻塞在采样上）；decode 全程走 **CUDA Graph replay**（量化 kernel 必须能进图）。

### 2.3 量化接入点全图

| 阶段 | 代码 | 量化相关 |
|---|---|---|
| 配置解析 | `quant/config.py` | 格式、粒度、硬件门控、跳过模块 |
| 层构造 | `get_quant_method` + `create_weights` | 建量化参数、绑 loader |
| 权重加载 | 模型 `load_weights` + method 的 loader | 切片、在线 staging、预量化直写 |
| 收尾 | `finalize_model` | staging → scale + 量化权重 |
| 迁设备 | `worker._to_device` | 保护量化参数 dtype |
| KV 池 | `KVPool(kv_dtype=…)` | fp8 KV 的显存减半 |
| 前向 | `method.forward → kernel` | 量化 GEMM / dequant-GEMM |
| attention | `attention.py::_attention_fp8` | FP8 KV 写读 |
| 录图 | `CUDAGraphRunner.capture` | 量化 kernel 的图内约束 |

---

## 3. FP8 W8A8（dense）

### 3.1 数据格式与 scale 语义

支持两种权重粒度：

- **per-tensor**：权重 `[N,K]` fp8 + 标量 `w_scale`。激活 per-token 动态量化（每行 `amax/448`），走 `torch._scaled_mm`（cuBLASLt）。
- **per-block 128×128**：权重 `[N,K]` fp8 + `weight_scale` 网格 `[ceil(N/128), ceil(K/128)]` fp32。激活在 kernel 内按 `(token, 128-k-group)` 动态量化。社区 FP8 checkpoint 的主流格式，也是在线量化的默认。

为什么 per-block 是默认：e4m3 只有 3 位尾数，单个离群值会抬高整行/整张量的 scale。128×128 块把离群值限制在局部，实测 dense 端到端相对误差约 3.5–4%（这已是 e4m3 的物理下限，见 [`fp8-debug-report.md`](fp8-debug-report.md) §4）。

### 3.2 加载路径

**预量化 checkpoint**（`fp8.py:158`）：权重已是 fp8，按 TP 规则切片直写；scale 网格按块切（`_load_scale`）。row-parallel 只切输入维，scale 沿 k 维切；column/merged/qkv 沿输出维切，段偏移按本地网格计算（`_seg_grid`）。

**在线量化**（`quantize_at_load`）：BF16 权重写 CPU staging，`finalize` 时 `quantize_fp8_per_block`（`fp8_gemm.py:43`）：

```python
padded = zeros(ceil(M/128)*128, ceil(N/128)*128)   # 尾部补零
blocks = padded.view(mb, 128, nb, 128)
amax   = blocks.abs().amax(dim=(1,3))
scale  = amax / 448
q      = (blocks / scale).to(fp8).reshape(...)[:m,:n]
```

补零保证块边界不被截断；amax 取块内最大值，`clamp(min=1e-12)` 防止全零块除零。

预量化 checkpoint 的 `lm_head` / `embed_tokens` 若也是 fp8，由 `Fp8FallbackCatcher`（`utils/model_loader.py:40`）暂存 fp8 权重，等对应 `weight_scale` 到达后反量化回 bf16 再走普通 loader——这保留了 embedding 的 padding 行与 TP 切分逻辑。

### 3.3 前向 kernel

per-block kernel（`fp8_gemm.py:113`）：

```python
for k in range(0, K, BLOCK_K):            # BLOCK_K = 128 = scale 组宽
    x = load(X + ...)                      # [BM, BK]，带 M/N 掩码
    w = load(W + ...)                      # [BN, BK]
    w_scale = load(WS + n_start//128 * stride_wsn + (k//128)*stride_wsk)
    x_amax = max(|x|, axis=1)              # 每 token、每 128-k-group
    x_q    = (x / x_amax[:,None] * 448).to(fp8e4m3)
    a_s    = x_amax / 448
    acc   += tl.dot(x_q, tl.trans(w)) * (a_s[:,None] * w_scale)
```

三个设计点：

1. **`BLOCK_N` 与 `BLOCK_K` 必须整除 128**。kernel 用 `n_start // 128`、`k // 128` 直接索引 scale 网格——一个 program 的 n 块/k 块必须落在同一个 scale 块内。取 128（等于块宽）是能用的最大值；更小只是多几个 program 共享同一 scale，更大就会跨块出错。
2. **激活量化融进 GEMM 的 k 循环**。不额外开 kernel、不额外读写 `[M,K]`：每读入一个 k-block 就地量化，`a_s` 与 `w_scale` 在 fp32 累加后做一次外积校正。
3. **输出复用 buffer**。`out=` 参数让 CUDA Graph replay 复用同一块输出，避免每层每步分配。

per-tensor 路径（`w8a8_tensor_fp8_matmul`）用 `torch._scaled_mm`。torch 2.13 只接受两种 recipe：per-tensor × per-tensor，或 rowwise × colwise。我们的语义是 per-token 激活 × per-tensor 权重——属于混搭。解法是把标量权重 scale **广播成连续的 `[1,N]`**，落到 rowwise × colwise 上，数学等价（`scale_b` 必须是连续张量，stride-0 的 `expand` 视图会被拒绝）。

### 3.4 TP 切分

per-block scale 的切分不是简单的 `narrow`：block 网格的起点随切分偏移。例如 row-parallel 切输入维 `in_local = in_full/tp`，本地 scale 网格列偏移是 `tp_rank * ceil(in_local/128)`。`_seg_grid` / `_load_scale` 对 column / row / merged / qkv 四种语义分别处理；qkv 的 k/v 在 GQA 下按 `head_idx = tp_rank // replicas` 取重复头。

### 3.5 一次请求中的完整通路

**启动期**：`QuantConfig.from_name("fp8")` → 每个 Linear 调 `get_quant_method(kind=…)` → `Fp8Method.create_weights` 建 `weight`（fp8）与 `weight_scale`（fp32 网格）并绑 loader → `load_weights` 逐张量路由（预量化直写 / 在线写 staging）→ `finalize_model` 对 staging 做 per-block 量化 → `_to_device` 保护量化参数。

**prefill**：`x [T, H]` → 层内 `quant_method.forward` → reshape `[T, H]` → `w8a8_block_fp8_matmul`：每个 program 先读一个 `[BLOCK_M, 128]` 的 k-block，就地量化激活、`tl.dot` 走 fp8 tensor core、用 `a_s ⊗ w_scale` 校正 fp32 累加；M=T 较大时 autotune 倾向 `BLOCK_M=128`。输出 reshape 回 `[..., N]`，随后才做该层的 TP 通信。

**decode**：同一 kernel，M=batch；CG 下整层（GEMM + 残差 + norm）在 graph 内 replay，`out=` 复用静态缓冲。

**数值**：dense 实测 `rel ≈ 0.033–0.039`（e4m3 物理下限）；8B batch-8 CG 下 decode 1.63×、显存 0.65×。

---

## 4. INT8 W8A8（dense）

### 4.1 粒度选择

- **权重 per-channel**（`[out]` fp32）：离群值集中在少数输出通道，per-tensor 会被一个通道压垮全局精度。
- **激活 per-token 动态**：注意力/FFN 激活存在逐 token 离群值；动态量化无需校准。
- **对称、范围 [-127, 127]**：避开 -128 让表示域关于 0 对称，反量化无偏置项。

### 4.2 融合 per-token 量化

`_per_token_quant_kernel`（`int8_gemm.py:32`）一个 program 处理一行：第一遍过 K 求 `amax`，第二遍量化。两遍都走 L1/L2（同一行的数据第二次仍在缓存），比"abs → amax → clamp → div → round → cast"六个小 kernel 快约 8 倍（注释实测 165 µs → 20 µs，M=128/K=2048）。

一个易错点：CPU 参考实现和 Triton 实现都必须 **round**。`.to(torch.int8)` 是向零截断，会给所有权重引入约半个 LSB 的系统性收缩偏置——这正是我们在 INT8 代码里修掉的一个 bug（§9.7）。

### 4.3 双路径分派

`torch._int_mm`（cuBLASLt int8）硬性要求 `M > 16` 且 K、N 为 4 的倍数，于是：

- **M > 16**：`_int_mm(a_q, w.t())` → int32 累加 → 乘 `a_s ⊗ w_scale` 回激活 dtype。prefill 走这条。
- **M ≤ 16**：Triton 小 M 内核（`BLOCK_M = next_pow2(M)`）。decode 的 M=1 是带宽瓶颈的 GEMV，`BLOCK_M=1` 让 Triton 走 FMA 路径，实测比 `BLOCK_M=16+掩码` 快约 7 倍（267 µs → 36 µs），也比 bf16 cuBLAS 快 2 倍。

两条路径共享同一套量化参数，只是 GEMM 后端不同；`out=` 池化对两者都支持。

### 4.4 一次请求中的完整通路

**启动期**：`create_weights` 建 int8 权重 `[out_local, in_local]` 与 per-channel scale `[out_local]` → 预量化 checkpoint 按输出维切 scale 直写；在线量化写 CPU staging → `finalize` 做 per-channel 量化。

**prefill**：`M=T > 16` → 融合 per-token 量化 kernel 产出 `a_q [T,K] int8` 与 `a_s [T,1]` → `torch._int_mm(a_q, w.t())` int32 累加 → 乘 `a_s ⊗ w_scale` 回激活 dtype。

**decode**：`M=batch ≤ 16` → Triton 小 M 内核，`BLOCK_M=next_pow2(M)`，int8 tensor core（M=1 走 FMA），同一套 scale 校正；CG 下整段在 graph 内。

**数值**：实测 `rel ≈ 0.000–0.004`（明显优于 FP8，因为 int8 的尾数多 4 位）；8B batch-8 CG 下 decode 1.67×、显存 0.67×。

---

## 5. INT4 weight-only（AWQ / GPTQ dense）

### 5.1 内部统一布局与加载期归一化

两种 checkpoint 的打包方式不同：

- **GPTQ**：沿输入维打包 `[in/8, out]`，uint4b8（存值 = 量化值 + 8）。
- **AWQ**：可能是 `[in, out/8]`（in-major，vLLM 侧）或 `[out, in/8]`（AutoAWQ 原始），且 in-major 下输出通道在 int32 内按 `[0,2,4,6,1,3,5,7]` 交织。

内部统一成 **in-major**：`qweight [in, out/8] int32`、`zeros uint8 [in/G, out]`、`scales fp16 [in/G, out]`。归一化在加载期完成（`normalize_int4_to_in_major`）：

- 形状可无歧义判定布局（`[in,out/8]+[in/G,out]` vs `[out,in/8]+[out,in/G]`）；
- out-major 走"沿末维解包 → 转置 → 重新按行打包"；
- AWQ 的交织用逆置换 `_inverse_perm()` 还原；
- qzeros 直接解包成 uint8，热路径不再做位运算。

**为什么不在 kernel 里适配**：交织/转置是"每个权重一次性"的 CPU 成本，而热路径每步都跑。用一次性成本换单一布局 + 单一内核，AWQ/GPTQ 只在偏置（`zero_bias`）与零点语义上分叉。

### 5.2 Fused dequant-GEMM

`_int4_gemm_kernel`（`int4_gemm.py:345`）的权重读取是核心技巧：

```python
w32 = tl.load(q_ptr + offs_k[:,None]*stride_qk + offs_n8[None,:]*stride_qn)  # [BK, BN/8]
q   = (w32[:,:,None] >> shifts) & 0xF                                        # [BK, BN/8, 8]
q   = tl.reshape(q, (BLOCK_K, BLOCK_N)).to(fp32)
w   = (q - z[None,:]) * s[None,:]
acc += tl.dot(a, w.to(a.dtype))
```

- **三维 load + reshape**：一个 int32 的 8 个 nibble 正好是 8 个连续输出通道，`[BK, BN/8, 8]` 的访存无重复、无 bank conflict，reshape 后直接是 `[BK, BN]`。
- **`BLOCK_K == group_size == 128`**：一个 k-block 恰好一个 scale 组，循环里只需取一行 `s/z`，不需要组内索引。
- **固定 tile，不用 autotune**（`BLOCK_M=32, BLOCK_N=64`）。两个原因：decode 的 M 每步变化，autotune 按 M 重调会累积编译与 benchmark 开销；更关键的是 **CUDA Graph capture 期间做 `do_bench` 属于非法同步**。固定配置 + 掩码覆盖 M=1..prefill 全区间，行为确定。

### 5.3 零点语义

- AWQ：非对称，`w = (q - z) * s`，z 来自 qzeros。
- GPTQ `sym=true`：uint4b8，`w = (q - 8) * s`，无 qzeros 张量（`zero_bias=8`）。
- GPTQ `sym=false`：非对称，z 来自 qzeros。

判断依据是 **`sym` 而不是 `zero_point`**：HF 的 GPTQ config 只给 `sym`，`zero_point` 会停在 dataclass 默认值，据此判断会走错分支（`int4.py:75` 的注释）。

### 5.4 TP 切分

- column / merged / qkv：按输出维切。打包列是 `out/8`，所以切片位置与长度都要除以 8（`src_out // _PACK`）；merged 的各段先在本地参数里算出段偏移。
- row：按输入维切 qweight 的行；scales/qzeros 按组切（`g_local = in_local / group`）。
- 打包维度要求切分点对齐 8（输出维）或 group（输入维），由 `ensure_divisibility` 与模型维度保证。

### 5.5 一次请求中的完整通路

**启动期**：只接受预量化 checkpoint（`from_name` 对 `awq/gptq + quantize_at_load` 直接报错）→ `create_weights` 建 packed 参数（`qweight` int32、`scales` fp16、asym 时 `qzeros` uint8）→ 每个 checkpoint 张量先归一化到 in-major，再按 TP 切片写入（打包列按 `/8`、组维按 group）。

**prefill / decode**：同一套 kernel，M 由 batch 决定；`int4_weight_only_matmul` 固定 `BLOCK_M=32/BLOCK_N=64`，k 循环内三维 load 解包、`(q-z)*s` 反量化、bf16 tensor core 累加。激活全程保持 bf16（weight-only），因此与 FP8 KV、TP 通信、CUDA Graph 全部正交。

**数值**：dense 实测 `rel < 0.02`；显存约为 bf16 的 1/4（不含未量化模块）。

---

## 6. MoE 专家量化（FP8 / INT4）

### 6.1 权重布局

- **FP8**：`w13_weight [E_local, 2i_local, h]` fp8（cutlass 约定 `[up; gate]`），`w2_weight [E_local, h, i_local]`；scale 为 `[E_local, 2i/128, h/128]` / `[E_local, h/128, i/128]` fp32。
- **INT4**：`w13_qweight [E_local, h, 2i_local/8]` int32（in=h，out=2i，`[up; gate]` 序），`w2_qweight [E_local, i_local, h/8]`；scales fp16、zeros uint8。

### 6.2 加载路由

MoE 的 checkpoint 有两种形态，`qwen3_moe.py` 都处理：

1. **per-expert 张量**（`experts.N.gate_proj.weight` 等）：`_load_expert_weight` 按 expert 定位本地槽位，gate/up 分别写到 w13 的 `[i,2i)` / `[0,i)` 段（cutlass `[up; gate]`）。预量化的 `weight_scale` 走 `weight_loader_w13_scale`（段网格 → 本地段块）。
2. **3D 融合张量**（`experts.gate_up_proj` / `down_proj`）：`_load_fused_expert_weight` → `load_fused_w13`。融合权重的 HF 段序是 `[gate; up]`，内部是 `[up; gate]`，**权重与 scale 必须做同一个置换**。
3. **覆盖率校验**：`_assert_expert_coverage` 检查每个本地 expert 的 gate/up/down 是否都到齐，缺张量直接报错，而不是静默留下未初始化显存。

在线量化时，`_write_w13_fp8` / `_write_w2_fp8` 对每个 piece 直接 `quantize_fp8_per_block` 并写 scale——**逐段量化而不是全量 staging**：gate/up 各自成块，段边界与 128 块对齐，逐段量化与整体量化等价，省掉一份全量 bf16 staging。

### 6.3 Fused grouped GEMM

前向（`moe/kernels/triton.py::forward`）：

1. **token 分组**：`M*top_k*4 <= E` 时走 naive 路径（每个 token-expert 对一个块，跳过 `moe_align_block_size`）；否则把 token 按 expert 排序、按 `BLOCK_M` 对齐。
2. **grouped GEMM**：一个 program 处理 (expert, m-block, n-block)。`expert_id < 0` 的块直接返回；`pid_m*BLOCK_M >= n_post` 的块跳过。
3. **scale 校正**：FP8 路径按 k-block 加载 `a_s[:,None] * w_s[None,:]`；INT4 路径在 k 循环内解包并乘 `(q-z)*s`。
4. **加权求和**：`topk_weights` 可在 GEMM 后乘（`mul_w`），最后 `moe_sum` 归约 `[M*top_k, H] → [M, H]`。

输出 buffer 按 `(形状, dtype)` 池化复用，与 dense 侧一致。注意返回的是**池化缓冲**：调用方必须在下一次 MoE 调用前消费（引擎内所有调用都是立即消费；测试里比较两次结果必须先 clone）。

### 6.4 EP / TP 切分

- **EP**：`local_expert_range` 按 expert 维度均分，`expert_map[global_id] = local_id`（远程为 -1）。前向先把全局 id 映射成本地 id，再进 grouped GEMM。
- **TP**：`intermediate_size` 切分，w13 的 `2i` 与 w2 的 `i` 都按本地段处理；w2 的 scale 沿输入组维切，w13 的 scale 沿输出块切。

### 6.5 一次请求中的完整通路

**启动期**：`Experts.__init__` 按 EP 过滤本地专家、按 TP 切 intermediate；加载时 per-expert 或 3D 融合张量分别路由（FP8 在线量化逐段量化写 scale；INT4 归一化到 in-major 后写入 packed 参数）；`_assert_expert_coverage` 校验每个本地专家的 gate/up/down 齐全。

**运行期**（`Experts.forward`）：

```
topk_ids ──expert_map──> 本地 expert id
  └─ fused_moe_forward(int4=False/True)
       ├─ token 分组：稀疏时 naive（token-expert 一对一），否则 moe_align_block_size
       ├─ grouped GEMM #1（w13，[up; gate]，FP8 带 block scale 校正）
       ├─ silu_and_mul → 中间激活（FP8 路径再次 per-token-group 量化）
       ├─ grouped GEMM #2（w2，epilogue 已乘 topk_weights）
       └─ moe_sum：归约 [M*top_k, H] → [M, H]
```

decode 的 `M*top_k*4 <= E` 时走 naive 路径，省掉排序；CG 下整段在 graph 内，权重/输出/scale 缓冲全部预分配。

**数值**：MoE FP8 实测 `rel 0.052–0.060`（两次 GEMM + 中间激活量化叠加，见 §9.3）；INT4 MoE 的误差主要由 4-bit 权重量化决定。

---

## 7. FP8 KV cache

### 7.1 为什么 scale 必须固定

decode 是增量的：同一个 KV slot 今天写、以后每一步读。如果写入时用 per-step 动态 scale，读取时就不知道历史 slot 用的是什么 scale。因此 scale 必须**每层固定、写读一致**。同时，CUDA Graph 捕获后不能再写 scale（replay 会用新数据重算，导致存量 KV 不一致），所以校准必须发生在录图前。

实现（`fp8_kv.py`）：

- 每层首次写入时按 `max(|k|,|v|)._max` 校准：`scale = amax/448 × 2`（2 倍头部余量抗离群值），然后冻结。
- 存两套表示：GPU Tensor（给 `store_kvcache_fp8` 内核读）+ Python float（FlashInfer wrapper 的形参要求标量）。

### 7.2 写入与读取

- **写入**：`store_kvcache_fp8` 按 slot scatter，`(key/k_scale).to(fp8)`；`slot == -1` 的 token 直接跳过。
- **CSR 转换**：引擎内部用 FA2 的 `block_tables [bs, mb]` + `cache_seqlens`，FlashInfer 要 `(indptr, indices, last_page_len)`。`block_tables_to_csr` 全 GPU 完成、零 host 同步：`cumsum` 求 indptr，位置散射做紧凑 packing，无效条目散射到末尾 trash 槽（不被读取）。
- **plan 在图外**：FlashInfer 的 `plan` 内部有 D2H，不能进 CUDA Graph。捕获前按每个 bs 建一个 CG wrapper 并 plan；replay 前在图外再 plan 一次当前批；图内只含 `wrapper.run`。
- **prefill**：无前缀缓存的首 chunk 仍走 FA2 varlen（直接读当前批 bf16 k/v，不碰 cache）；有缓存时走 FlashInfer paged prefill（fp8 cache + scale）。

### 7.3 校准的取舍

校准用的是 `ModelRunner` 构造期的一次假 decode 前向（token id=1）。这是上游 vLLM 的同类做法（`calc_kv_scales`），但存在风险：如果真实请求的 k/v 幅度显著更大，会饱和截断。2 倍 headroom 缓解；更稳妥的做法是用真实 prompt 校准，或在首个真实 prefill 前完成校准（仍需在录图前）。

### 7.4 一次请求中的完整通路

**启动期**：`KVPool` 以 `float8_e4m3fn` 分配（同显存块数翻倍）；`Fp8KVManager` 建 wrapper/CSR 缓冲；`_calibrate_fp8_kv` 跑一次假 decode 前向，逐层校准并冻结 scale。

**prefill**：每层 `store_kvcache_fp8`（按固定 scale 量化写入）；
- 无前缀缓存的首 chunk：attention 直接读当前批 bf16 k/v（FA2 varlen），不读回 cache；
- 有前缀缓存：`plan_prefill`（图外）→ FlashInfer paged prefill 读 fp8 cache。

**decode**：图外 `_plan_decode_fp8`（CSR 转换 → `wrapper.plan`，含 D2H）→ 图内 `run_decode_cg`（`wrapper.run`，`k_scale/v_scale` 以 Python float 传入）→ 输出。

**收益/代价**：KV 池 2.25 → 1.12 GiB（同 blocks），8B batch-8 CG 下 decode −6%（FlashInfer paged + 每步 plan）。它是容量手段，不是吞吐手段。

---

## 8. 优化清单

| 优化 | 位置 | 收益 | 代价 / 约束 |
|---|---|---|---|
| 激活量化融进 FP8 GEMM 的 k 循环 | `fp8_gemm.py:157` | 省一次 `[M,K]` 读写与 kernel launch | 内核更复杂；BLOCK_K 必须对齐 scale 组 |
| INT8 per-token 量化单 kernel 两遍过 K | `int8_gemm.py:32` | 165 µs → 20 µs（M=128,K=2048） | 两遍读同一行，依赖 L1/L2 |
| INT8 小 M 专用 tile（BLOCK_M=next_pow2(M)） | `int8_gemm.py:181` | 267 µs → 36 µs（M=1） | 多一套 kernel 与分派阈值 |
| INT4 三维 load + reshape 解包 | `int4_gemm.py:400` | 无重复访存、无 bank conflict | 需要 in-major 布局 |
| INT4 固定 tile、不用 autotune | `int4_gemm.py:426` | 首次编译稳定、可进 CUDA Graph | 放弃按 M 的最优配置 |
| 加载期归一化 AWQ/GPTQ（交织/转置/解包） | `int4_gemm.py:126` | 热路径只剩移位与乘加 | 加载慢一次；布局判定需覆盖已知格式 |
| 在线量化用 CPU staging | `fp8.py:191`、`int8.py:174` | GPU 峰值不翻倍（8B 可在 24 GB 卡上量化） | 需要等量 host RAM；finalize 有 CPU 时间 |
| 逐段量化 MoE 专家 | `experts.py:288` | 省全量 staging | 段边界必须与 128 块对齐 |
| 输出 buffer 池化 | 各 kernel 的 `out=` | 消除每层每步分配，CG 可复用 | 返回值在下一次调用前有效（契约） |
| CUDA Graph 覆盖 decode | `cuda_graph.py` | 摊掉 kernel launch（见下） | autotune / host 同步必须移出图 |
| FP8 KV：固定 scale + 图外 plan | `fp8_kv.py` | KV 显存减半、可进图 | 每步一次 plan 的 D2H；decode −6% |
| 采样 id 异步 D2H | `model_runner.py:_stage_samples` | CPU 不阻塞在采样上 | 需要 pinned buffer |

其中最容易被低估的是 **CUDA Graph 对量化收益的影响**：同一组量化配置，CG 下 decode 是 BF16 的 1.63×，eager 下反而只有 0.78×。原因不是量化本身变了，而是 eager 下量化路径多出的 kernel launch 无法被摊薄（见 [`quantization-bench.md`](quantization-bench.md) §3.3）。

---

## 9. 错误复盘：怎么看出来、为什么这么改

调试方法论先放在前面，它比单个 bug 更可复用：

1. **先建参照系，再判对错。** 数值对不上时，先用 torch 手写"理想实现"（同粒度、同 scale 语义、朴素算子），把误差拆成"内核 vs 理想"和"理想 vs 高精度"两段。前者大 → 内核缺陷；后者大 → 方案固有误差。FP8 dense 的 0.0075 vs 0.0395 一出来，结论就唯一了。
2. **同内核对照组隔离变量。** MoE 的手工模拟连 bf16 基线都对不上（0.21），说明模拟不可信；改用"同一个内核、只换输入（fp8 权重 vs 反量化回 bf16 的权重）"，差值就只剩激活量化一项。
3. **埋点看状态，不看猜测。** 预热失效那个 bug 卡了很久，转折点是打印真实对象状态（client 类型、`waiting`/`running` 队列长度），而不是继续推理"为什么 warmup 不生效"。
4. **警惕"逐位相同"。** 量化结果不可能逐位等于未量化结果；这种异常几乎总指向别名、缓存或比较对象搞错（MoE 池化缓冲就是这么暴露的）。
5. **注意自己的探针有没有把 bug 抄进去。** 两个探针给出相反结论，往往是因为其中一个把有缺陷的写法换成了正确写法。探针与产品代码共用未验证的辅助函数时，会一起错或一起掩盖。

### 9.1 案例总表

| # | 现象 | 怎么定位 | 根因 | 修复 |
|---|---|---|---|---|
| 1 | Triton `NameError: Cannot access global variable` | traceback 指向 kernel 体引用的模块常量 | Triton 3.x 只允许内核引用 constexpr 全局 | 为内核建 `tl.constexpr` 别名，host 常量不动 |
| 2 | `Unsupported lhs dtype fp8e4nv` | 监视 kernel 入参：权重变成了 bf16 | `model.to(dtype=...)` 把 fp8 权重和 fp32 scale 一起转成激活 dtype | `_to_device` 按 `quant_method` 归属保护量化参数 |
| 3 | CUDA illegal memory access | `CUDA_LAUNCH_BLOCKING` + spy 记录每次调用的 M/N/K | per-block kernel 的 load 缺 M/N 掩码，M=160 时读第 255 行 | load 与 store 用同一套掩码 |
| 4 | `Invalid scaling configuration` | 枚举 scale 形状组合 | torch 2.13 拒绝 per-token × 标量的混搭 recipe | 标量 scale 广播成连续 `[1,N]`，落到 rowwise × colwise |
| 5 | `Warmup: 0.00s` 但 TTFT 1.4 s | patch `warmup_engine`，打印 `drain -> 0.0000s` 时 `waiting=1` | `drain_engine(is_finished)` 被传入 `has_unfinished`，谓词取反 | 改签名为 `has_unfinished`，`while has_unfinished(): step()` |
| 6 | MoE fp8 `rel=0.0000`（逐位相同） | `got is ref`、同一 `data_ptr` | `fused_moe_forward` 返回池化缓冲，第二次调用原地覆写参考值 | 测试侧 `.clone()`；API 契约保持零拷贝 |
| 7 | 3D 融合 FP8 MoE 反量化误差 0.95 | 用 stored scale 反量化后与原权重比 | `load_fused_w13_scale` 的 gate/up 变量取反，`cat` 成了恒等 | 交换两段；新增回归单测 |
| 8 | INT8 per-channel 误差偏高 | 对比截断 vs 四舍五入 | `.to(int8)` 向零截断，系统性收缩偏置 | `.round()` 后再 cast |
| 9 | FP8 KV `Expected float but got ffi.Tensor` | 读 flashinfer 源码：`sm_scale *= k_scale` 后按 float 下发 | wrapper 的 scale 形参要求 Python float | 校准期同时存 Tensor 与 float |
| 10 | FP8 KV `size of indices should be ≤ buffer` | 读 `plan()` 的校验 | CG wrapper 的缓冲按 bs 分配，传了 max_bs 全长 | 按 `bs*max_blocks+1` 切片 |

### 9.2 几个值得展开的案例

**#2 量化参数被 `.to(dtype)` 覆盖。** 报错是 Triton 的，但问题在加载：`load_model()` 内部 `finalize_model()` 已经完成量化，随后 `model.to(dtype=self.dtype)` 遍历**所有**浮点参数——fp8 权重、fp32 scale 全被转成 bf16。为什么按"归属"而不是按 dtype 白名单修？因为量化参数不止 fp8：scale 是 fp32，INT4 还有 int32 packed、int8 zeros，未来还有 int4 零点；用 `quant_method.extra_params()` 的归属信息判断是唯一对所有方法都正确的做法。

**#3 越界读为什么"有的形状炸、有的不炸"。** M=160 崩溃而 M=176/240 存活，是典型的"越界读落在 torch 缓存分配器已映射的页内"——普通运行看不出来，只有 `--cg` 的 graph 内存布局才把它变成致命错误。store 有掩码而 load 没有，属于同一函数内的不对称；修的时候把 N 方向也一起补上（N 当时恰好是 128 的倍数所以没暴露，但任何非整数倍 N 都会踩）。

**#5 预热从未生效。** `drain_engine(is_finished, step)` 的语义是"直到 is_finished"，而 run.py 传的是 `engine.has_unfinished`：`while not has_unfinished()` 恒假，一帧都不跑。影响不是"少了预热"，而是预热请求滞留在队列里，第一次真实 step 连它们一起 prefill，FP8 首次触发的 ~1.4 s autotune 全部落进计时区间——TTFT 1442.8 ms 里有 1.35 s 是一次性成本。修复选在函数侧而非调用点：`Engine` 的公开接口就是 `has_unfinished`，让 `drain_engine` 与之对齐比让每个调用方包一层取反更自然，参数名一并改掉防复踩。

**#6 池化缓冲导致的假通过。** fp8 与 bf16 的结果逐位相同，这在量化里是不可能的。直接比较 `got is ref` 与 `data_ptr` 一眼看到是同一个张量：第二次调用原地覆写了第一次的返回值，测试在拿张量和自己比。修复只在测试侧 clone——引擎内所有调用方都是立即消费（残差相加），把 API 改成默认拷贝会让每层每步多一次 kernel 与一份缓冲（实测 3.6–4.9% 的 MoE 时间，graph 内 ≤1%，但图捕获下 48 层要占约 400 MB），为防呆不值得。

**#7 段序错位。** `load_fused_w13` 把 HF 的 `[gate; up]` 转成 cutlass 的 `[up; gate]`，scale 加载器注释也写着要转，但代码里 `up = loaded[:, :gate_go]` 实际取到的是 gate，`cat([up, gate])` 成了恒等操作。用 gate/up 量级差 20 倍的构造一测就现形：up 段 rel=0.95、gate 段 rel=14.9；交换后 0.0357。这类"注释与代码不一致"的 bug 只能靠构造可区分的输入来测，所以补了一个带反向断言（直接用原序 scale 必须显著错位）的回归单测。

**#9/#10 FP8 KV 的两个 API 契约。** flashinfer 0.6.15 的 `run()` 把 `k_scale` 乘进 `sm_scale` 后按 float 下发 kernel，传 0-dim Tensor 会被 TIR 拒绝；而 hot path 不能每步 `.item()`（D2H 同步、且进不了图），所以校准期一次性存两套表示。CG wrapper 的 `plan()` 会校验传入 indices 长度 ≤ 它按 bs 分配的缓冲，管理器给的是 max_bs 全长，必须切片。这两个问题 kernel 级测试都覆盖不到（测试直接调 wrapper），所以补了一个**引擎级**测试：同档位 eager vs CUDA Graph 必须逐 token 一致。

### 9.3 不是 bug 的两件事

- **3% 的 FP8 容差不可达**。用理想模拟拆分后：内核 vs 理想 0.0075，理想 vs bf16 0.0395。e4m3 每元素平均 3–4% 相对误差，误差与信号随 K 同步按 √K 增长，不会被平均掉。结论是把验收界放宽到 0.08 / 0.12，而不是改 kernel。
- **e2e "逐 token 一致率 ≥99%" 不可达**。teacher-forced 实测显示，分歧只发生在 BF16 自身 top1–top2 间隔 ≤0.5 的近似并列处（fp8 在该处 margin=0.0，int8=0.5）；一次翻转会级联污染后续 token，16 token 里翻 3 个就是 81%。验收改为：相同上下文下决定性位置的 top-1 一致率 + logit 偏差，另加 eager/CG 逐 token 完全一致。

---

## 10. 验收与实测结果

三层验收：

1. **kernel 数值**（`tests/gpu/test_quant_*.py`）：FP8 dense `rel < 0.08`（实测 0.033–0.039）、MoE `< 0.12`（0.052–0.060）；INT8 `< 0.02`（0.000–0.004）；INT4 dense `< 0.02`。
2. **端到端**（`test_quant_e2e.py`）：teacher-forced 决定性位置 top-1 一致率 ≥95%（实测 fp8/int8 均 14/14），top-1 logit 偏差 ≤1.5（实测 1.125/0.75）；同档位 eager vs CG 逐 token 完全一致。
3. **性能 / 显存**（[`quantization-bench.md`](quantization-bench.md)）：单卡 Qwen3-8B、batch-8、256-token prompt、CG 口径：

| 档位 | decode tok/s | e2e tok/s | peak GiB | 显存比 |
|---|---:|---:|---:|---:|
| BF16 | 435.0 | 406.3 | 18.48 | 1.00× |
| FP8 | 711.0 | 634.7 | 11.96 | 0.65× |
| INT8 | 724.6 | 655.1 | 12.47 | 0.67× |
| FP8 + FP8 KV | 667.0 | 599.7 | 11.00 | 0.60× |

---

## 11. 已知限制

- **INT8 不覆盖 MoE 专家**：显式 `NotImplementedError`，避免静默混合路径。MoE 模型请用 fp8 / awq / gptq。
- **INT4 只能加载预量化 checkpoint**：AWQ/GPTQ 需要校准统计量，无法从 BF16 在线量化；`desc_act=true` 的 GPTQ（g_idx）显式拒绝。
- **SmoothQuant 的 int8 checkpoint 显式拒绝**：其权重已按校准输入 scale 预缩放，配动态 per-token 激活会错。
- **FP8 KV 的 scale 校准于构造期假数据**：2× headroom 缓解饱和风险；更稳妥的做法是在首个真实 prefill 前校准（仍可在录图前）。
- **FP8 per-tensor 路径每层一次 `[1,N]` 分配**：非默认路径；进入热路径可缓存为层 buffer。
- **INT4 MoE 的 scales 存 fp16**：每次 forward `.float()` 一次转换（AWQ/GPTQ checkpoint 的标准布局）；在 CUDA Graph 中复用固定地址，开销小。
- **多卡量化测试**（`test_quant_parallel.py`）仍使用旧的"逐 token 一致率"门槛，需在 2 卡机器上改为 teacher-forced 口径。
