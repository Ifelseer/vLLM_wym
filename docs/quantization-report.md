# vLLM_wym 量化落地工作报告

截至 2026-10-13。覆盖 Phase 0（调研核验）→ Phase 1（量化框架）→ Phase 2（FP8 W8A8）→ Phase 3（FP8 KV cache）四个任务。每个任务按「第一性原理：为什么这么做 → 关键设计 → 优化动机 → 技术细节与坑」组织。

- [1] 总览与验收状态
- [2] Phase 0：调研核验——先证明"存在可行解"再动手
- [3] Phase 1：量化框架——把"开放维度"关在笼子里
- [4] Phase 2：FP8 W8A8——在 e4m3 的物理下限内做正确的事
- [5] Phase 3：FP8 KV cache——被 API 能力边界逼出来的架构选择
- [6] 测试体系：三层金字塔与"容差不是调到能过"
- [7] 遗留项与风险清单
- [8] 路线图与后续

---

## 1. 总览与验收状态

| 任务 | 提交 | 代码量 | 测试 | 状态 |
|------|------|-------|------|------|
| Phase 0 调研 | —（决策报告见对话） | — | — | ✅ 完成 |
| Phase 1 框架 | `479d04e` | +1,190 | 41 unit | ✅ 完成 |
| Phase 2 FP8 W8A8 | `479d04e` + `92724fb`（4090 修复） | +1,700 | 74 unit + kernel GPU | ✅ 4090 验收通过 |
| Phase 3 FP8 KV | `FP8 KV cache（SM89+）` | +946 | 89 unit + KV GPU | ⏳ 待 4090 验证 |
| Phase 4 INT4 | — | — | — | 未开始 |
| Phase 5 INT8 | — | — | — | 未开始 |

最终性能（4090，Qwen3-0.6B，在线 FP8）：TTFT +8%、decode −2%、E2E −5% 相对 BF16。
**这个"持平"是预期结果而非失败**——0.6B 权重仅 1.2 GB，4090 上根本不触碰显存带宽瓶颈，FP8 的收益（权重减半、KV 空间翻倍、30B 级模型单卡可跑）在这个尺寸上无法体现，而 per-block 路径的在线量化开销是净负面。**量化在该尺寸下的正确验收口径是显存与容量，不是吞吐**——这一点在选基准时就应明确，否则会误判方案成败。

---

## 2. Phase 0：调研核验——先证明"存在可行解"再动手

### 第一性原理

量化不是一个"实现问题"，而是一个**约束满足问题**。解空间由三个集合的交集决定：

```
可行方案 = 社区 checkpoint 格式 ∩ 硬件内核能力 ∩ 引擎架构可改造性
```

任何一个集合的偏差都会让后面几百行代码作废。例如：若 Qwen3 没有官方 FP8 checkpoint，就要走在线量化；若在线量化被否，整个方案要改。**Phase 0 的全部意义就是在写第一行代码前，把这个交集算出来。**

### 核验结果（检索日 2026-10-12）

| 模型 | FP8 | INT4 | 格式细节 |
|------|-----|------|---------|
| Qwen3-0.6B（稠密） | ✅ 官方 | ❓ | `quant_method=fp8, fmt=e4m3, weight_block_size=[128,128], activation_scheme=dynamic` |
| Qwen3-32B（稠密） | ✅ 家族 | ✅ AWQ | AWQ：`bits=4, group_size=128, zero_point=true, version=gemm` |
| Qwen3-30B-A3B（MoE） | ✅ 官方 | ❓ | 同上 FP8 block 格式；`modules_to_not_convert` = lm_head + 全部 norm + 全部 router |
| Qwen1.5-MoE-A2.7B | ❌ | ✅ GPTQ-Int4 | `sym=true, desc_act=false, exllama_config v2` |

**结论：无死角，不需要 fallback 到在线量化**——这正是用户要求的"如果不可以，就告诉我"的反面：可以。

### 三个改变技术路线的核验发现

1. **FA2 不支持 FP8 KV cache**。`flash_attn_with_kvcache` / `flash_attn_varlen_func` 的签名里没有 `k_scale/v_scale`，FP8 KV 是 FA3 特性。原计划"4090 上用 FA 路径"的前提被推翻。
2. **FlashInfer 0.6.15 原生支持 FP8 KV**（`BatchDecode/BatchPrefillWithPagedKVCacheWrapper` 的 `kv_data_type=fp8` + `k_scale/v_scale`，FA2 后端，SM89 可用）。这把 Phase 3 的技术路线从"改 FA2"（不可能）逼到"切 FlashInfer wrapper"（可做但工作量大）。
3. **cutlass MoE 的 block-scale FP8 仅 SM90**（vLLM 源码 `_supports_quant_scheme` 实证）。Qwen3-30B-A3B-FP8 的 128×128 block 格式在 4090（SM89）上不能走 cutlass → 必须自研 Triton block-scale grouped GEMM。这直接决定了 Phase 2 的 MoE kernel 选型。

### 技术知识点

- **FP8 e4m3**：1 符号 + 4 指数 + 3 尾数，动态范围 ±448，相对精度 2⁻³≈6%（3 位尾数、最近舍入半 ulp）。选 e4m3 不选 e5m2：KV/权重的动态范围需求低于精度需求。
- **block-wise vs per-tensor scale**：per-tensor 一个 scale 覆盖整个张量，离群值会压低全体的有效精度；128×128 block scale 用 1/16384 的存储开销（fp32 scale vs fp8 数据）换局部自适应，是社区 FP8 checkpoint 的主流（DeepSeek/Qwen3 系均如此）。
- **AWQ vs GPTQ 布局**：同为 INT4 group=128，但 AWQ 的 `qweight` 是 `[out, in/8]` int32 打包、`zero_point=true`；GPTQ 是转置布局 + 可选 `desc_act`（act-order，权重行按激活重要性重排，推理时需要 `g_idx` 还原）。**混用布局是静默错误的主要来源**——形状刚好但语义错。
- **SM 架构与 FP8**：FP8 tensor core 从 SM89（Ada Lovelace，4090/L40S）才有；SM86（3060）只有 BF16/INT8。这决定了"本地 3060 验证 INT8、4090 验证 FP8"的分工，不是偏好而是物理约束。

---

## 3. Phase 1：量化框架——把"开放维度"关在笼子里

### 第一性原理：为什么是"方法注册表分派"而不是"层内 if/else"

推理引擎的维度里，**模型结构是封闭维度**（Qwen3/Qwen3-MoE 就那几类层），**量化格式是开放维度**（fp8/int4/int8/awq/gptq/未来格式）。封闭维度适合直接写；开放维度如果以 if/else 散落在 5 种线性层 × 2 个模型 × MoE 专家里，每加一种格式的改动面是乘法级。

vLLM 的答案是 `QuantizationConfig.get_quant_method(layer)`：层只做薄委托，量化逻辑按格式分文件。本引擎照做（**借鉴设计，不复制代码**——本项目不能引入 vLLM 依赖）：

```
层（linear.py）      持有参数、决定 TP 通信语义
方法（quant/methods/）创建量化参数与 scale、weight_loader（TP 切分）、forward（kernel）
```

新增一种格式 = 加一个文件 + 注册表一行，既有层零改动。这是开闭原则在工程上的直接兑现。

### 关键设计决策与动机

| 决策 | 备选 | 为什么这么选 |
|------|------|------------|
| `QuantConfig` frozen dataclass | 可变配置对象 | 不可变 → 可哈希、可跨层共享、无"配置被某层改掉"的隐患 |
| `from_hf_config` + `from_name` 双入口 | 单一入口 | 预量化 checkpoint（格式由模型决定）与 CLI 强制（格式由用户决定）是两种语义，混在一个入口会产生"谁覆盖谁"的歧义 |
| `modules_to_not_convert` **点分段匹配** | 裸子串匹配 | 见下方"坑 1" |
| 量化参数按**归属保护**（`extra_params`） | dtype 白名单跳过 | 见下方"坑 2" |
| 在线量化（`quantize_at_load`） | 只支持预量化 | 用户模型库里大量 BF16 模型，重新下载/量化成本高；FP8 的在线量化是纯数学变换（amax→scale），无需校准数据，成本近零 |

### 两个值得记住的坑

**坑 1：子串匹配会误伤**。`skip_module` 最初用裸子串：skip 列表里的 `mlp.gate`（MoE router）会命中 dense MLP 的前缀 `mlp.gate_up_proj`——把不该跳过的层跳过。改为**点分段感知**（skip 项的分段序列必须是前缀分段的连续子序列）后，`mlp.gate` 只匹配 `mlp.gate`，`lm_head` 仍能命中 `lm_head.weight`。教训：**名字空间的匹配要么精确要么分段，裸子串是 bug 温床**。

**坑 2：`nn.Module.to(dtype=...)` 遍历所有浮点参数**。worker 的 `model.to(dtype=bf16)` 把 fp8 权重和 fp32 scale 一起转成 bf16，直接摧毁量化参数（4090 上表现为 `Unsupported lhs dtype fp8e4nv` 的离奇报错）。修复不是"跳过 fp8 dtype"——scale 是 fp32、未来 INT4 还有 int32 元数据，白名单永远追不全——而是按**归属**判断：`quant_method.extra_params()` 里的参数保持自己的 dtype，其余才转。**归属信息比类型特征更本质**，这个原则对后续所有量化方法通用。

### 技术知识点

- **`weight_loader` 属性约定**：引擎既有代码把加载器挂在参数对象上（`setattr(param, "weight_loader", ...)`），模型的 `load_weights` 按参数名查表调用。量化框架沿用这个约定，量化参数的加载器由 QuantMethod 用闭包绑定——**新增格式不需要改 `load_weights` 的路由主体**，这是 Phase 1 最重要的可扩展性保证。
- **bound method 绑定**：`functools.partial(self._load_weight, layer=layer)` 挂到参数上，调用时签名 `(param, loaded_weight, shard_id)` 与既有约定一致，模型侧零感知。

---

## 4. Phase 2：FP8 W8A8——在 e4m3 的物理下限内做正确的事

### 第一性原理：为什么 dense FP8 走自研 Triton 而不是 `torch._scaled_mm`

`torch._scaled_mm` 是 SM89 原生 FP8 GEMM，性能最好，但它只接受两种 recipe：**per-tensor × per-tensor** 或 **rowwise × colwise**。而社区 FP8 checkpoint（Qwen3 系）是 **128×128 block scale**——scale 同时沿输出和 K 维变化，`_scaled_mm` 无法表达。Triton 的自由度（自己算 scale 索引、自己写校正逻辑）是唯一选择。**性能让位于格式兼容，格式兼容让位于正确性**——这是量化落地的优先级排序。

### 关键设计

1. **激活量化粒度：per-token × 128-group**。激活按行（token）分组、组内 128 元素共享 scale。为什么不是 per-tensor：注意力/FFN 激活存在显著的逐 token 离群值（少数通道值域大一个数量级），per-tensor 会压垮整行精度。为什么不是 per-channel：激活是动态的，per-channel 需要统计缓存，违背"免校准"前提。
2. **权重侧：静态 block scale**。checkpoint 带的 `weight_scale` 直接用；在线量化（BF16 模型）时按同一 128×128 网格算。
3. **两条加载路径**：预量化（fp8 权重 + 独立 scale 张量，直写）与在线量化（staging bf16 → `finalize` 统一量化）。在线路径的动机：用户模型库以 BF16 为主，FP8 在线量化是纯数学变换，**省掉一次全量下载**。
4. **workspace 池化**：GEMM 输出缓冲按形状缓存复用（同 MoE `_scratch` 模式），保证 decode CUDA Graph 录图/replay 期间无逐帧 cudaMalloc。

### MoE FP8：为什么必须自研 Triton grouped GEMM

Phase 0 已证 cutlass block-scale 仅 SM90。4090 上 Qwen3-30B-A3B-FP8 的唯一路径是改造引擎已有的 Triton MoE kernel：在 K 循环里按 k-block 载入 `w_scale[expert, n_blk, k_blk]`，与激活 scale 一起校正 fp32 累加：

```python
acc += tl.dot(a_q, b_q) * (a_s[:, None] * w_s[None, :])
```

`a_s ⊗ w_s` 的外校正是数学关键：block scale 下每个 (token, k-group, n-group) 三元组都有自己的缩放，不能提出循环。这也带来一个细节：**BLOCK_N=64 时多个 n 列共享一个 128-block 的 scale**（n_blk = offs_n // 128），kernel 里用向量索引天然处理，无需特殊分支。

### 4090 调试报告沉淀的六个缺陷（全部已修，详见 docs/fp8-debug-report.md）

按"根因层级"归类比按现象归类更有教育意义：

| 缺陷 | 根因层级 | 一句话教训 |
|------|---------|-----------|
| Triton kernel 引用非 constexpr 全局 | 语言约束 | jit 函数只能访问 `tl.constexpr`；host/kernel 双用途常量要建别名，不能把常量本身改成 constexpr（会静默改变 host 算术） |
| `.to(dtype)` 摧毁量化参数 | API 语义 | `nn.Module.to` 不区分"激活参数"和"量化参数"；按归属保护，不按 dtype 白名单 |
| per-block kernel load 缺掩码 | 对称性 | store 有掩码 load 没有 = 半个内核正确；"同形状有的崩有的不崩"是越界读落在已映射页的典型信号 |
| `_scaled_mm` recipe 混搭 | 上游 API 约束 | per-token 激活 × per-tensor 权重不在两种合法 recipe 内；广播 scale_b 成连续 `[1,N]` 落到 rowwise×colwise，数学等价 |
| 预热谓词取反 | 契约语义 | `while not has_unfinished()` 恒假；两个探针结论相反是因为**探针二把 bug 绕过去了**——探针与产品代码共用未验证辅助函数时会一起错 |
| MoE 池化缓冲别名 | 测试效度 | 量化结果逐位等于未量化参考 ⇒ 一定是比较对象搞错；`rel(x, x)` 恒为 0 |

### 第一性原理：验收容差为什么从 0.03 放宽到 0.08

这不是"调到能过"，是用**参照系实验**证明的。dense 段三组数据：

```
内核 vs bf16 参考        rel = 0.0357
内核 vs 理想 fp8 模拟    rel = 0.0075   ← 内核离理想只差 0.75%
理想 fp8 模拟 vs bf16   rel = 0.0395   ← 方案固有误差 3.95%
```

误差的 89% 来自 e4m3 方案本身。理论上界可以算：3 位尾数 + 最近舍入 ⇒ 每元素相对误差 ~6% 最大、~3~4% 平均；点积中信噪比按 √K 增长、随机误差也按 √K 增长，**比值不随 K 收敛**——所以 ~4% 是随机数据上的渐近下限，与维度无关。放宽容差的判据是"真 bug（漏 scale 校正、走错分支）会给出 >0.10 的误差，仍能被拦住"，而不是"让灯变绿"。

### 技术知识点

- **Triton `tl.dot` 的 fp8 约束**：两侧必须都是 fp8（`lhs.dtype.is_fp8() and rhs.dtype.is_fp8()`），混 bf16/fp8 直接编译错误。这是"某一侧没量化"的报错长得像 kernel bug 的原因。
- **autotune 与 CUDA Graph 的交互**：`@triton.autotune` 在首次遇到某 (M,N,K) 组合时用 `do_bench` 试跑候选配置（含同步）——**绝不能发生在 graph capture 期间**。引擎的解法是 capture 前 warmup（`CUDAGraphRunner.capture` 先 `body(bs)` 再录图），warmup 覆盖全部 captured bs。
- **`torch.compile` × Triton autotune 未验证**：Dynamo 对用户 Triton kernel 的支持在 3.7 已可用，但与 autotune 的交互没有验证。**保守门控：量化开启时禁用 prefill compile**（eager prefill 回退），正确性优先——这是明确的已知取舍而非遗漏。

---

## 5. Phase 3：FP8 KV cache——被 API 能力边界逼出来的架构选择

### 第一性原理：三个约束的交集

FP8 KV cache 的正确性由三个约束同时锁定，缺一不可：

```
约束 A（容量）：KV 张量以 fp8 存储，显存减半、块数翻倍。
约束 B（一致性）：同一物理 slot 的写入 scale 与读取 scale 必须相同
                  ——增量 KV 是跨步累积的，per-step 动态 scale 直接违反。
约束 C（可录图）：整个 store+read 通路必须能进 CUDA Graph。
```

**约束 B 是全部设计的枢纽**。它一举否掉了两个直觉方案：

1. **per-step 动态 scale**（每步按当前批 amax 量化）：step t 写入的 token 用 scale_t，step t' 读取时用 scale_{t'} ⇒ 存量 KV 全部错位。否。
2. **per-slot 缓存 scale**（每个 slot 旁边存自己的 scale）：cache 布局大改、FA/FlashInfer 的 scale 入参都是 per-tensor/per-head，否。

剩唯一正解：**per-layer 固定 scale**——首次写入时按该层 k/v 的 amax 校准（留 2× 头部余量抗离群值），之后冻结。它同时满足约束 C：校准发生在录图前的预热前向里，captured 区域不再碰 scale，replay 读到的是冻结值。

### 架构选择：为什么切 FlashInfer 而不是别的

FA2 无 scale 入口（Phase 0 已证），"改 FA2"不可能；"读时反量化"（fp8 存储 + 反量化回 bf16 再喂 FA2）保容量不保带宽，且多一个 kernel——**在存在原生支持（FlashInfer wrapper）时，中间层翻译是净亏损**。直接切 wrapper，KV 布局 `[blocks, page, nkv, hd]` 与 FlashInfer 的 4-D NHD paged cache 视图**天然一致，零拷贝**。

### 四个关键技术细节

**1. CSR 转换必须零同步且紧凑 packing。** 引擎内部用 FA2 风格 `block_tables [bs, mb]`，FlashInfer 要 `(indptr, indices, last_page_len)`。两个坑：

- **紧凑 packing**：indices 是 seq i 的页**连续存放**于 `indptr[i]:indptr[i+1]`。最初按行优先写入（每 seq 占 mb 格）会留下行间空洞，kernel 按 indptr 跳读即错页。修复：预计算位置网格 `pos[i,j]=indptr[i]+j`，无效条目散射到缓冲末尾的 trash 槽——**用 scatter 的静态形状换零 host 同步**。
- **零同步**：indices 缓冲预分配到 `bs*mb+1`，`indptr[-1]` 之后的条目不被读取，因此不需要 `.item()` 回主机取长度。这一步是 decode 热路径的刚需（每步一次 D2H 同步 ≈ 2.5 ms 步预算的 1%）。

**2. plan 不能进图。** FlashInfer 的 `plan` 内部有 `indptr.to("cpu")`（host 侧计算 split-K 划分），capture 期间做 D2H = 非法。解法是把 plan 提到录图外：`CUDAGraphRunner.capture` 新增 `on_warmup` 钩子，每个 bs 录图前 plan 一次（warmup 前向本身也跑一遍完整计算，顺带校准 scale），captured 区域只含 `wrapper.run`。**plan/run 分离是 FlashInfer 的显式设计**（源码注释原话："plan is not part of model logic... should not be included in Cuda Graph"），引擎只是顺着它做。

**3. CG 补齐行的 CSR 合法性。** CUDA Graph 按 padded bs  replay，补齐行的 `cache_seqlens=0` 会产生 `last_page_len=256` 但 `n_blocks=0` 的不合法组合。修正：补齐行 KV 长度置 1、页表指向 page 0——这些行的输出本就被 `hidden[:bs]` 丢弃，**用"合法但无意义"代替"非法"**。

**4. 首 chunk 不碰 cache。** 无前缀缓存的 prefill 直接读当前批 bf16 k/v（FA2 varlen），只把 k/v 量化**写入** cache 供后续 decode。原因：此时还没有"跨步累积"问题，走全精度还省一次反量化；只有 chunked prefill 的续 chunk（读存量 cache）才必须走 FlashInfer paged prefill。

### 技术知识点

- **`CUDAGraphBatchDecodeWithPagedKVCacheWrapper` 的 batch size 构造时固定**（由缓冲长度决定），与引擎按 bs 列表录图的架构天然对齐——每 bs 一个 wrapper，懒创建，共享 128 MB workspace（顺序使用无冲突）。
- **k_scale 进 softmax、v_scale 进输出**：`sm_scale *= k_scale` 在前（QK² 缩放里），`out *= v_scale` 在后（PV 之后）——这是 FlashInfer 对 fp8 KV 的 scale 注入点，与 FA3 的 `scale_k/scale_v` 语义一致。传 0-dim GPU 张量即可，`float *= tensor` 产生 GPU 端 kernel，capture/replay 均合法。
- **scale 的安全余量**：amax/448×2 只用了半个动态范围。后验依据：Qwen3 的 k 经过 RMSNorm（k_norm），逐头 std≈1，per-tensor amax≈4~5，5.3 覆盖只有 1.2× 余量；用 2× 换 8.5 覆盖，代价是 fp8 有效位数少 1 bit（相对误差 6%→12%）。**这是精度/鲁棒的显式交换，不是拍脑袋**——若某模型激活更宽，应改为加载期校准或 per-head scale。

---

## 6. 测试体系：三层金字塔与"容差不是调到能过"

```
unit（CPU，本地 3060 可跑，89 例）
  ├── QuantConfig 解析/硬件门控/分段匹配         ← 配置语义
  ├── 加载器 TP/EP 切分/量化往返数值             ← 切分正确性
  ├── 模型级 load_model 全链路（dense/MoE）      ← 路由/绑定/catcher 集成
  └── CSR 转换不变量（紧凑 packing/trash/边界）   ← 纯数学语义
gpu（4090，SM86 自动跳过）
  ├── FP8 kernel 对齐（dense per-block/per-tensor，MoE M=1/16）
  └── FP8 KV 对齐（decode/prefill/roundtrip vs FA2 BF16）
e2e（4090）
  └── run.py 端到端 + 与 BF16 基线对比
```

**分层的原则**：能在 CPU 抓的 bug 不留到 GPU——CSR 的行间空洞 bug 就是被 CPU 单测抓到的（GPU 上它表现为"读错页"的随机乱数，极难定位）；能在 kernel 层抓的不留到 e2e。GPU 测试用隔离子进程（`run_isolated`）+ worker 内 SM 门控，同一份代码在 3060 上自动 skip、4090 上全量跑。

**验收口径的方法论**（来自 4090 调试，可复用）：数值对不上时，先建"理想实现"参照系（同粒度、同 scale 语义、朴素算子），把总误差拆成"内核 vs 理想"与"理想 vs 高精度"两段——前者大是内核 bug，后者大是方案固有。**没有参照系就放宽容差，是把验收体系降级为"让灯变绿"**。

---

## 7. 遗留项与风险清单

| 项 | 性质 | 处置 |
|---|---|---|
| FP8 KV GPU 三路对齐 + e2e（含 CG 交互） | 已验证 | `test_quant_kv.py`：decode / prefill / roundtrip 三路 + 引擎级 eager/CG 一致性；修复了 FlashInfer scale 标量类型与 CG indices 缓冲长度两处问题 |
| e2e 正确性口径 | 已修正 | 自由生成的逐 token 一致率 ≥99% 不可达（近似并列翻转会级联）；改为 teacher-forced 同上下文 top-1 / logit 比较，并单列 eager vs CG 一致性 |
| 预量化 checkpoint 路径（`--quantization auto`） | 未测 | 只验了在线量化；Qwen3-0.6B-FP8 的 tied embed 正好覆盖 catcher 反量化 |
| MoE + FP8 + CG | 未测 | Triton block-scale grouped GEMM 进图是全新组合 |
| 长 prompt TTFT | 未测 | +8% 是短 prompt 口径；8K prefill 的 FP8 开销可能放大 |
| FP8 + prefill torch.compile | 主动禁用 | Dynamo × autotune 未验证；验证后开放 |
| per-tensor 路径每层 `[1,N]` 分配 | 小额开销 | 非默认路径；进入热路径再缓存为层 buffer |
| `fused_moe_forward` 返回池化缓冲 | 契约 | 已注释 + 测试 clone；需要跨调用持有时加 `out=` |
| Qwen3-30B-A3B-FP8 在线量化（amax 校准） | 未验 | 该模型只有预量化格式；在线路径对 MoE 仅在无预量化时用 |
| INT4 + EP 组合 | 未实现 | Phase 4 需显式 raise（计划要求：不支持的组合显式报错） |

---

## 8. 路线图与后续

- **Phase 4 INT4 weight-only**（AWQ/GPTQ + Triton fused dequant-GEMM）：本地 3060 可端到端验证；核心风险是 packed 布局的语义正确性（AWQ `[out, in/8]` vs GPTQ 转置 + g_idx），CPU 单测必须覆盖 unpack 数学。
- **Phase 5 INT8 W8A8**（per-token 激活 × per-channel 权重，`torch._int_mm`）：SM86 原生支持，是本地完整验证的兜底路径。
- **Phase 6 测试矩阵 + bench + 文档 + DSpark 预留点**：`docs/quantization.md` 设计文档、`utils/model_hub.py` 预量化条目、`bench/vs_vllm.py` 量化口径（含"同精度同 prompt 的 TTFT/tok/s/显存"与 greedy 一致率软断言）。

---

