# vLLM_wym 量化引擎落地技术报告

**日期**：2026-10-12 ~ 2026-10-16
**范围**：Phase 0 调研 → 量化框架 → FP8 W8A8（dense/MoE/KV cache）→ INT4 weight-only → INT8 W8A8
**环境**：开发机 RTX 3060 8G（SM86）+ 租用 RTX 4090 24G（SM89）；PyTorch 2.11/2.13、Triton 3.7、FlashInfer 0.6.15、flash-attn（FA2）
**代码**：7 个提交、`src/quant/` 新模块 + 4 处既有文件最小开洞、149 项自动化测试

---

## 1. 项目背景与整体目标

### 1.1 起点

`vLLM_wym` 是一个从零实现的高性能 LLM 推理引擎，采用 control / data / compute 三面架构，已具备 BF16 推理、TP/PP/DP/EP 并行、PagedAttention KV 池、piecewise CUDA Graph decode、chunked prefill、融合 MoE（Triton/cutlass 双后端）等能力。**量化能力为零**。

### 1.2 目标

在不改变既有架构语义（调度、KV 分页、并行策略、生成行为）的前提下引入量化，达到：

| 目标 | 度量 |
|---|---|
| 降低权重显存 | FP8 减半、INT4 减至 1/4 |
| 提升吞吐 | decode tok/s、TTFT（同 prompt 对比 BF16 基线） |
| 扩大硬件覆盖 | INT8 覆盖无原生 FP8 的 SM86（3060） |
| 支持 30B 级 MoE 单卡可跑 | INT4 weight-only |
| 正确性 | greedy token 一致率 ≥ 99%（vs BF16 eager） |

**约束**：完全复用现有技术栈（不引入 vLLM/torchao）；量化关闭时代码路径与现状完全一致（回归基线）；每个 Phase 独立提交、独立可测。

---

## 2. 执行流程与关键步骤

```
Phase 0 调研（1 天）
   └─ 核验 checkpoint 可用性 × scale 格式 × FA/FlashInfer 能力矩阵 → 决策报告
Phase 1 框架（2 天）
   └─ src/quant/（QuantConfig / QuantMethod 抽象 / 注册表）+ EngineConfig 接入
Phase 2 FP8 W8A8（3 天，含 4090 调试）
   └─ dense（per-tensor + per-block）+ MoE grouped GEMM
Phase 3 FP8 KV cache（2 天）
   └─ FlashInfer paged attention 通路 + per-layer 固定 scale + CG 适配
Phase 4 INT4 weight-only（3 天）
   └─ dense 5 类层 + MoE 专家，加载期归一化到统一布局
Phase 5 INT8 W8A8（2 天，本地 3060 闭环）
   └─ per-token 动态激活 × per-channel 静态权重，双路径内核
Phase 6 测试/bench/文档（进行中）
```

每个 Phase 的产出都遵循同一模式：**实现 → CPU 单测（本地）→ GPU 数值测试（本地或 4090）→ 模型级集成测试 → 提交**。INT4/INT8 因不依赖 FP8 硬件，实现了本地 3060 的完整闭环——这是把「INT8 放最后」反过来变成优势的安排：**前面的 FP8 必须租 4090，而后两个 Phase 的 kernel 级验证零成本**。

---

## 3. 设计思路与决策依据

### 3.1 架构：方法注册表而非层内 if/else

**第一性原理**：推理引擎的维度里，模型结构是**封闭维度**（Qwen3 就那几类层），量化格式是**开放维度**（fp8/int4/int8/未来格式）。封闭维度适合直接写；开放维度若以 if/else 散落在 5 种线性层 × 2 个模型 × MoE 专家里，每加一种格式的改动面是乘法级。

**决策**：借鉴 vLLM 的 `get_quant_method(layer)` 分派（借鉴设计，不复制代码——本项目不引入 vLLM 依赖）：

```
层（compute/layers/linear.py）   持有参数、决定 TP 通信语义
方法（quant/methods/*）          创建量化参数与 scale buffer、weight_loader（TP 切分）、forward
```

新增格式 = 加一个文件 + 注册表一行，既有层零改动。实测：INT4 与 INT8 两个 Phase 中，`linear.py` / `qwen3.py` 的**路由主体零改动**（仅 INT4 的专家正则、INT8 的 SmoothQuant 拒绝各加了几行）。

**关键细节**：量化参数的命名直接取 HF 后缀名（`weight` / `weight_scale` / `qweight` / `scales` / `qzeros`），使既有 `load_weights` 的「按参数名查表」路由天然命中——这是「最小开洞」能成立的前提。

### 3.2 量化粒度：按**离群值所在轴**选择

| 张量 | 粒度 | 依据 |
|---|---|---|
| 激活 | per-token（FP8 按 128-group、INT8 按整行） | 离群值分布在** token 间**，per-tensor 会压垮整行 |
| 权重 | per-channel（INT8）/ per-block 128×128（FP8）/ group 128（INT4） | 离群值集中在**少数输出通道/局部块** |

这个选择有实测支撑：INT8 测试中构造「单通道离群 ×40」，per-channel 在非离群行的误差比 per-tensor 小 5 倍以上；FP8 的 per-block（128×128）则是社区 checkpoint 的主流格式（DeepSeek/Qwen3 系均如此），属于**格式兼容优先于性能偏好**的决策。

### 3.3 加载两态：预量化直写 + 在线量化

- **预量化 checkpoint**：fp8/int8 权重 + scale 张量，按 TP 规则切片直写。
- **在线量化**（`quantize_at_load`）：BF16 权重先写 staging buffer，全部 shard 到齐后 `finalize` 统一量化。动机：用户模型库以 BF16 为主，重新下载/离线量化成本高；FP8/INT8 的在线量化是纯数学变换（amax→scale），**无需校准数据**——这与 INT4 形成对比：INT4 需要校准（AWQ 的搜索、GPTQ 的 Hessian），因此 INT4 **只支持预量化**，并在 `from_name` 显式拒绝在线请求。

`finalize` 通过 `model_loader` 的统一钩子触发，**新增量化方法无需改加载器**。

### 3.4 FP8 per-block 为什么自研 Triton 而非 `torch._scaled_mm`

`_scaled_mm` 只接受两种 recipe：**per-tensor × per-tensor** 或 **rowwise × colwise**。而社区 FP8 checkpoint 的 128×128 block scale 同时沿输出和 K 维变化，`_scaled_mm` 无法表达。**格式兼容 > 性能偏好**，自研 Triton kernel 是唯一选择（per-tensor 模式仍走 `_scaled_mm`，两条路径并存）。

### 3.5 INT4 为什么在加载期归一化布局

AWQ/GPTQ 有**三种互不兼容**的张量布局（详见 §5.1）。若在内核里逐一适配，内核数与测试面乘以 3。决策：**加载期一次性归一化到统一 in-major 布局**（解包→重排→重打包，CPU numpy 级成本），运行时单一内核 + 单一测试面。这是「用一次性成本换热路径简单」的权衡。

### 3.6 INT8 为什么按 M 双路径

`torch._int_mm`（cuBLASLt int8）硬性要求 **M > 16**（tensor core 的物理约束，见 §5.2）。decode 的 M=1..16 必须走别的路径：

- `M > 16`：`_int_mm`（int8 tensor core，int32 累加）
- `M <= 16`：自研 Triton 小 M 内核，**`BLOCK_M` 按 M 精确取**（1/2/4/8/16）

### 3.7 FP8 KV cache 的 scale 语义：per-layer 固定

FP8 KV 的正确性由三个约束同时锁定：

```
A（容量）：KV 以 fp8 存储，显存减半
B（一致性）：同一物理 slot 的写入 scale 与读取 scale 必须相同
         ——增量 KV 跨步累积，per-step 动态 scale 直接违反
C（可录图）：整个 store+read 通路必须能进 CUDA Graph
```

**约束 B 一票否决**了 per-step 动态 scale（step t 写入用 scale_t、step t' 读取用 scale_{t'}，存量 KV 全部错位）和 per-slot scale（cache 布局大改、FA/FlashInfer 的 scale 入参都是 per-tensor/per-head）。唯一正解：**per-layer 固定标量**，首次写入时按该层 k/v 的 amax 校准（2× 头部余量抗离群值）后冻结。它同时满足约束 C——校准发生在录图前预热前向里，captured 区域不再碰 scale。

---

## 4. 主要问题与根因分析

按根因层级归类（比按现象归类更有教育意义）：

### 4.1 编译期 / 语言约束类

| 问题 | 根因 |
|---|---|
| Triton kernel 引用非 constexpr 全局 → `NameError` | Triton 3.x jit 函数只能访问 `tl.constexpr` 全局；host/kernel 双用途常量必须建**别名**而非把常量本身改成 constexpr（`tensor / tl.constexpr(448.0)` 会返回 constexpr 对象，静默改变 host 行为） |
| `tl` 张量无 `.round()` / `tl.math.round` 不存在 | Triton 3.7 的 rounding 要走 `triton.language.extra.libdevice.round` |
| `nn.Parameter` 不能包 int32/uint8/fp8 | Parameter 默认 `requires_grad=True`，torch 拒绝非浮点 dtype 的梯度；必须显式 `requires_grad=False` |
| `Tensor.sum(dim=, out=)` 签名已移除 | torch 2.11 起该重载不存在；改用 `out.copy_(x.sum(dim=))` |

### 4.2 API 语义陷阱类

| 问题 | 根因 |
|---|---|
| `model.to(dtype=bf16)` 把 fp8 权重和 fp32 scale 一起转成 bf16 → `Unsupported lhs dtype fp8e4nv` | `nn.Module.to(dtype=...)` 遍历**所有**浮点参数，不区分「激活参数」与「量化参数」。修复不是 dtype 白名单（scale 是 fp32、INT4 还有 int32/int 元数据，白名单永远追不全），而是按**归属**判断：`quant_method.extra_params()` 内的参数保持自己的 dtype |
| `_scaled_mm` 报 `Invalid scaling configuration` | per-token 激活（`scale_a=[M,1]`）× per-tensor 权重（标量 `scale_b`）不在两种合法 recipe 内。修复：把标量权重 scale 广播成**连续**的 `[1,N]`，落到 rowwise×colwise（注意：`expanded` 的 stride-0 视图同样被拒） |
| `_int_mm` 报 `self.size(0) needs to be greater than 16` | cuBLASLt int8 路径的硬约束（M=1 无法用 int8 tensor core） |

### 4.3 数值 / 内存正确性类

| 问题 | 现象 | 根因 |
|---|---|---|
| per-block kernel load 缺掩码 | `CUDA error: illegal memory access`，「同形状有的崩有的不崩」 | store 有掩码而 load 没有 = **半个内核正确**；越界读落在 torch 分配器已映射页内时不报错，换一次分配布局才踩到未映射页 |
| INT4 scale stride 位置序错位 | 输出 NaN | kernel 形参是 FP8 时代的 `(bse, bsn, bsk)`，修改变量赋值时**漏改传参的三行顺序**——bsk/bsn 仍交换 |
| INT4 `offs_n8` 索引错 | `reshape() cannot change total number of elements` | 写成 `offs_n // 8`（BN 个、每列重复 8 次）而非 `BN/8` 个打包列 |
| INT4 `K` 变量作用域 | 非 int4 路径 `UnboundLocalError` | 把 `_, K = A.shape` 挪进了 int4 分支，**静默破坏 bf16 路径**——本地 MoE 测试因环境问题跳过，所以没抓住 |
| INT4 in-major/out-major 形状歧义 | 输出通道排列错乱（rel≈1.4） | `in == out` 时两种布局形状相同（方形层如 o_proj）。修复：歧义时优先 in-major（vLLM 内核约定的最强证据）并**告警**提示可核查 |
| INT8 scale 在 bf16 上计算 | 0.4% 系统误差 | `amax / 127` 在 bf16 下得到 bf16 scale，存进 fp32 参数已损失精度；改为 `amax.to(float32) / 127` |

### 4.4 测试效度类（最隐蔽、危害最大）

| 问题 | 现象 | 根因 |
|---|---|---|
| MoE 池化缓冲别名 | `moe fp8 rel=0.0000`——量化结果**逐位相同**于未量化参考 | `fused_moe_forward` 返回按形状池化的复用缓冲，第二次（fp8）调用原地覆写第一次（bf16）的参考 → `rel(x, x)` 恒 0。**量化的结果不可能逐位等于未量化结果，这类异常几乎总指向别名/比较对象搞错** |
| 预热谓词取反 | `Warmup: 0.00s`，FP8 TTFT 1442ms（基线 107ms） | `drain_engine(is_finished, step)` 的参数语义是「已完成」，调用方传的却是 `engine.has_unfinished` → `while not has_unfinished()` 恒假，循环体一次都不执行。FP8 首次调用的 ~1.4s Triton autotune 全部落在计时区间内 |
| 参考模型用错 | 反量化结果与原权重 rel≈1.5 | checkpoint 由模型 A 随机初始化生成，参考却另建模型 B（**不同的随机初始化**）——两者权重本来就不同 |
| 转置遗漏（多次） | rel≈1.3~1.5、值幅度相近 | 反量化结果是 `[in, out]`，HF 权重是 `[out, in]`，比较前忘了 `.t()` |
| 探针把 bug 抄进去 | 两个探针给出相反结论 | 探针二把有缺陷的 `drain_engine` 换成了手写正确循环 → 「预热明明是好的」。**探针与产品代码共用未验证的辅助函数时会一起错** |

---

## 5. 优化思路与亮眼技术点

### 5.1 自主工程优化

**① INT4 的三维 load + reshape nibble 解包（免重复访存）**

常见写法按 k8 重复加载 int32 八次（靠 L1 缓存兜底）；本实现直接 load `[BLOCK_K, BLOCK_N/8]` int32 tile，经 `w32[:, :, None] >> shifts` 广播成 `[BK, BN8, 8]` 再 `reshape` 到 `[BK, BN]`——**一次访存完成解包**，DRAM 流量就是 packed 尺寸。dense 与 MoE 两个内核共用同一手法。

**② INT8 融合 per-token 量化内核（165us → 20us）**

朴素实现 `abs → amax → clamp → div → round → cast` 是 6~9 个小 kernel、对 `[M,K]` 多次全量读写。实测 M=128/K=2048 时量化要 **165us**，而 `_int_mm` 本身只要 65us——**量化才是瓶颈**。融合成单内核（两遍过 K，同段命中 L1/L2）后降到 ~20us，直接让 INT8 从「比 bf16 慢」翻转为「CG 口径快 1.4x」。

**③ INT8 小 M 的 `BLOCK_M` 按 M 精确取（267us → 36us，再快 bf16 2x）**

`BLOCK_M=16`+掩码对 M=1 是 16 倍的浪费。改为 `BLOCK_M = next_pow2(M)` 后，M=1 让 Triton **自动降级为 FMA 路径**（GEMV 根本不需要 tensor core，带宽才是瓶颈）——实测 36us，比 `BLOCK_M=16` 快 ~7x，且比 cuBLAS bf16（60us）快 2x。参数扫描（BN/BK/warps 18 组）确认 BN=128/BK=128/warps=4 是甜点。

**④ FP8 KV 的 CSR 紧凑 packing + trash 槽（零 host 同步）**

引擎内部用 FA2 风格 `block_tables [bs, mb]`，FlashInfer 要 `(indptr, indices, last_page_len)`。两个坑：
- **indices 必须紧凑**（seq i 的页连续存放于 `indptr[i]:indptr[i+1]`）。最初按行优先写入留下行间空洞，kernel 按 indptr 跳读即错页——这个 bug 被 **CPU 单测**抓住（GPU 上表现为「读错页的随机乱数」，极难定位）。
- **零同步**：indices 缓冲预分配到 `bs*mb+1`，`indptr[-1]` 之后的条目不被读取，因此不需要 `.item()` 回主机。对 decode 热路径，每步一次 D2H 同步 ≈ 2.5ms 步预算的 1%。无效条目散射到缓冲区末尾的 **trash 槽**，用 scatter 的静态形状换零同步。

**⑤ 布局歧义的显式告警**

INT4 在 `in == out`（方形层）时两种布局形状相同，无法从形状判定。实现不静默选择，而是打印告警说明「假设 in-major，若 checkpoint 由 out-major 打包器产生则值会错乱，请对照参考输出核查」——把不可判定的情况变成**可行动的提示**而非隐藏地雷。

### 5.2 来自库 / 框架的令人印象深刻的处理

**① Triton autotune 与 CUDA Graph 的冲突——设计哲学的体现**

`@triton.autotune` 首次遇到某 `(M,N,K)` 组合时用 `do_bench` 试跑候选配置（含同步）——**绝不能发生在 graph capture 期间**。引擎的 `CUDAGraphRunner.capture` 先 `body(bs)` warmup 再录图，warmup 覆盖全部 captured bs，autotune 在录图外完成。我们自己的 INT4/INT8 kernel **有意不用 autotune**：M 变化频繁会累积编译与基准开销，且固定 tile + 掩码覆盖全 M 区间、行为确定。

**② FlashInfer 的 plan/run 分离——为 CUDA Graph 而设计**

`plan` 内部有 D2H 拷贝（host 侧计算 split-K 划分），源码注释原话：「plan is not part of model logic... should not be included in Cuda Graph」。引擎顺着这个设计把 plan 提到录图外：给 `CUDAGraphRunner.capture` 增加 `on_warmup` 钩子，每个 bs 录图前 plan 一次，captured 区域只含 `wrapper.run`（读 plan 写入的设备缓冲）。**这不是妥协，而是框架层面的正确抽象**。

**③ cuBLASLt int8 的 M>16 约束——tensor core 的物理本质**

`_int_mm` 拒绝 M≤16 不是缺陷：int8 tensor core 的 mma 指令形状（如 8×32）在小 M 下无法填满，硬件选择直接拒绝而非给一个慢路径。理解这一点后，小 M 走 Triton FMA 反而是**更优解**（GEMV 是带宽瓶颈，tensor core 无用武之地）。

**④ Triton 对小 M 的自动降级——`BLOCK_M=1` 比 16 快 7x 的原因**

Triton 发现 `tl.dot` 的 M<16 时自动走 FMA 路径。这意味着「按 M 精确取 tile」不只是省掩码，而是**切换了计算范式**（tensor core → FMA）。这个行为没有写在文档里，是参数扫描试出来的。

**⑤ `moe_align_block_size` 的 sorted-token 重排**

把 `(token, expert)` 对按专家排序展平，让每个 expert 的 token 连续存放——这是 grouped GEMM 能用统一 `BLOCK_M` tile 的前提。FP8/INT4 的 MoE 内核都建立在这个抽象上，量化只需在 K 循环里换「解包+scale 校正」，**排序/路由/silu/sum 全部复用**。

**⑥ CUDA Graph 摊掉 launch 开销——eager 与 CG 口径可以给出相反结论**

INT8 在 eager 口径下比 bf16 慢（3 个 kernel vs cuBLAS 1 个 call），在 CG replay 口径下快 1.4~1.8x。**若只跑 eager bench 会误判「INT8 无用」**。引擎 decode 本来就走 CG，所以真实收益为正。这个教训值得固化：**bench 口径必须匹配部署形态**。

---

## 6. 踩坑实录（现象 → 根因 → 修复）

精选十个最有教育价值的：

**坑 1：`model.to(dtype)` 摧毁量化参数**
- 现象：4090 上报 `Unsupported lhs dtype fp8e4nv`，指向 `tl.dot(x_q, tl.trans(w))`
- 根因：`nn.Module.to(dtype=...)` 不区分激活参数与量化参数，把 fp8 权重和 fp32 scale 一起转成 bf16
- 修复：`_to_device()` 按 `quant_method.extra_params()` 归属保护
- 验证：224 次 fp8 matmul 调用，0 次非 fp8 权重

**坑 2：per-block kernel 的半个掩码**
- 现象：`--cg` 崩 `illegal memory access`，且「M=160 崩、M=176/240 不崩」
- 根因：store 有掩码 load 没有；越界读落在已映射页内时不报错
- 修复：load/store 用同一套掩码；被掩元素写 0，fp32 累加不影响有效输出
- 教训：**「同形状有的崩有的不崩」是越界读的典型信号**；compute-sanitizer 在本机起不来，靠 `CUDA_LAUNCH_BLOCKING=1` + spy 记录每次调用形状定位

**坑 3：预热从未生效**
- 现象：两条 e2e 命令都打印 `Warmup: 0.00s`，FP8 TTFT 1442ms
- 根因：`drain_engine` 的参数语义（「已完成」）与调用方传入的谓词（`has_unfinished`）相反，循环一次都不执行
- 修复：按唯一调用点的语义对齐函数
- 过程：两个探针结论相反，转折点是**停止推理、打印真实对象状态**（client 类型、waiting/running 队列长度）——`drain -> 0.0000s` 旁边跟着 `waiting=1`，矛盾自体指出了 bug

**坑 4：池化缓冲让测试假通过**
- 现象：`moe fp8 rel=0.0000`——量化逐位等于未量化
- 根因：`fused_moe_forward` 返回池化复用缓冲，fp8 调用原地覆写 bf16 参考
- 修复：测试侧 `.clone()` + 注释写清契约（引擎内所有调用方都是立即消费——残差相加——别名是设计使然；为防呆付每层 6.8µs 的拷贝不值，**结论是维持原样、把契约写进注释与测试**）

**坑 5：INT4 的 stride 位置序（NaN）**
- 现象：MoE INT4 输出 NaN
- 根因：kernel 形参是 FP8 时代的 `(bse, bsn, bsk)`；我修改变量赋值时漏改传参三行的顺序，bsk/bsn 仍交换
- 定位：三路判别实验（dense kernel / MoE kernel / 参考）→ dense 对、MoE 错 → 逐行 diff 两内核源码 → host 传参比对
- 教训：**修改变量名时必须同时检查所有使用点**；「重命名后按新序传参」比「按位置传参」更安全（本可避免）

**坑 6：INT4 的 `K` 作用域（静默破坏 bf16 路径）**
- 现象：MoE bf16 路径 `UnboundLocalError: K`
- 根因：把 `_, K = A.shape` 从函数开头挪进了 int4 分支
- 为什么没被现有测试抓住：本地 MoE GPU 测试因 torch 2.11 环境问题跳过
- 教训：**本地跑不起来的路径缺少测试保护**；这也直接催成了 INT4 GPU 测试自带 MoE 段 + 环境探测自动跳过

**坑 7：INT4 布局歧义**
- 现象：out-major 加载后 rel≈1.4（值幅度相近、排列错乱）
- 根因：`in == out` 时 in-major 与 out-major 形状相同
- 修复：歧义时优先 in-major + 告警；测试改非方形维度
- 遗留：真实 AWQ checkpoint 的最终确认需上机（已在文档标注）

**坑 8：INT8 scale 的 bf16 系统误差**
- 现象：在线量化的 `weight_scale` 与参考差 0.23%
- 根因：`amax / 127` 在 bf16 下算出 bf16 scale，存进 fp32 已损失
- 修复：`amax.to(float32) / 127`；单测容差从「猜一个值」改为「按 bf16 amax 的物理精度推导」

**坑 9：`_int_mm` 的 M>16**
- 现象：M=1 报 `self.size(0) needs to be greater than 16`
- 根因：cuBLASLt int8 的 tensor core 约束
- 修复：小 M 走 Triton 内核，且 `BLOCK_M` 按 M 精确取（意外获得 7x）

**坑 10：转置遗漏（四次）
- 现象：rel≈1.3~1.5，两个张量幅度相近但不对齐
- 根因：反量化结果是 `[in, out]`，HF 权重是 `[out, in]`，比较前忘 `.t()`
- 教训：**「值幅度相近但 rel 很大」几乎总是轴序问题**，先查转置再查算法

---

## 7. 最终效果与可量化指标

### 7.1 测试

| 层级 | 数量 | 说明 |
|---|---|---|
| unit（CPU，本地 3060 可跑） | 128 | 配置解析、加载器切分、量化往返、模型级加载、CSR 转换、INT4/INT8 全链路 |
| GPU kernel 数值 | 4 套 | FP8 dense/MoE、FP8 KV、INT4 dense/MoE、INT8 dense——**INT4/INT8 本地实跑**，FP8 在 4090 |
| 合计 | **149 passed** | 唯一失败项为本地 torch 2.11 + 3060 的既有环境问题（已用 `git stash` 实验确认与改动无关） |

### 7.2 数值精度（rel = max\|Δ\| / max\|ref\|）

| 方法 | rel | 阈值 | 依据 |
|---|---|---|---|
| FP8 dense per-block（M=1~512） | 0.033~0.039 | 0.08 | e4m3 方案物理下限 ~4%（3 位尾数、误差不随 K 收敛） |
| FP8 MoE（M=1/16） | 0.052~0.060 | 0.12 | 两级 GEMM + 激活量化叠加 |
| FP8 KV decode/prefill | 通过 | 0.08 | e4m3 KV 同级 |
| INT4 dense（kernel vs 参考反量化） | 0.003~0.005 | 0.02 | 仅 bf16 输出舍入 |
| INT4 MoE（vs bf16 权重路径） | 0.160 | 0.25 | INT4 group=128 每元素 ~6%，经 silu 两级放大 |
| INT8 dense（M=1~2049） | 0.0000~0.0038 | 0.02 | int8 ~0.8% + bf16 舍入 |

阈值的设定方法：**先建「理想实现」参照系**（同粒度、同 scale 语义、朴素算子），把总误差拆成「内核 vs 理想」与「理想 vs 高精度」两段——前者大是内核 bug，后者大是方案固有。FP8 dense 的 0.0075 vs 0.0395 对比一出来，结论就唯一了。

### 7.3 吞吐（本地 3060，单层 [N=4096, K=2048]，CUDA Graph replay 口径）

| 方法 | M=1 | M=8 | M=128 | M=512 |
|---|---|---|---|---|
| INT8 W8A8 vs BF16 | **1.69x** | **1.77x** | **1.42x** | 1.06x |
| FP8 W8A8 vs BF16（4090，Qwen3-0.6B e2e） | TTFT +8% / decode −2%（0.6B 不触带宽瓶颈，符合预期） | | | |

INT8 的权重访存减半（int8 vs bf16）+ int8 tensor core，在带宽瓶颈的 decode 场景收益最直接。

### 7.4 显存

| 方法 | 权重 | 说明 |
|---|---|---|
| FP8 | 1/2 | 同时放大 KV 空间（FP8 KV 再减半 → 块数翻倍） |
| INT4 | 1/4 | Qwen3-32B-AWQ ≈ 18GB，单卡 24GB 可跑 |
| INT8 | 1/2 | + 激活 scale 可忽略 |

### 7.5 代码

```
7 个提交；src/quant/ 新模块（config/methods/kernels 三层）
既有文件开洞：linear.py（quant 工厂）、qwen3.py / qwen3_moe.py（路由）、
              moe/experts.py（3D 量化）、attention.py / kv_pool.py / model_runner.py（FP8 KV）
测试：8 个新测试文件
```

---

## 8. 方法论沉淀

1. **先建参照系，再判对错**。数值对不上时不要直接放宽容差——用 torch 手写「理想实现」把误差拆解成内核缺陷与方案固有误差两段。
2. **能在 CPU 抓的 bug 不留到 GPU**。INT4 的 CSR 行间空洞在 CPU 单测里是一条断言，在 GPU 上是「读错页的随机乱数」。
3. **警惕「逐位相同」与「幅度相近但 rel 很大」**。前者几乎总指向别名/比较对象搞错；后者几乎总是轴序问题。
4. **不支持的组合显式 raise，不静默降级**。SmoothQuant input_scale、MoE+INT8、GPTQ desc_act 都是明确报错而非留一个不可预期的混合路径。
5. **归属保护优于类型白名单**。参数保护按 `quant_method` 归属判断，对后续所有量化方法通用；dtype 白名单永远追不全。
6. **bench 口径必须匹配部署形态**。eager 与 CG replay 可以给出相反结论（INT8 的 0.5x vs 1.7x）。
7. **修改变量名时检查所有使用点**；能按语义传参就不要按位置传参。
8. **探针不要复用未验证的产品代码路径**，否则两者会一起错或一起掩盖。

---

## 9. 遗留项与后续

- **真实 checkpoint 的 AWQ 交织约定**最终确认（已按 vLLM 内核证据实现并在布局歧义时告警）——需在 GPU 机上跑 `Qwen3-0.6B-AWQ` 冒烟（交织错则输出乱码，失败模式显眼）。
- **greedy token 一致率**（≥99%）的端到端 gate——只测了 kernel 级数值，未测同 prompt 逐 token 比对。
- **预量化 FP8 checkpoint 路径**（`--quantization auto` + Qwen3-0.6B-FP8）未验证（在线量化路径已验证）。
- **MoE + FP8 + CUDA Graph** 组合未验证（Triton block-scale grouped GEMM 进图是全新组合）。
- **长 prompt TTFT**（8K prefill 的 FP8 开销）未测。
- FP8 + prefill `torch.compile` 主动禁用（Dynamo × Triton autotune 未验证）。
- DSpark 投机解码扩展点：`src/quant/` 注册表与 `Scheduler.schedule()` 的 waiting/running 队列是 draft/target 双序列的天然挂载处（本轮仅预留说明，未实现）。
