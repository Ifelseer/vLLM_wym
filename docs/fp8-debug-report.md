# FP8 量化落地调试报告

记录 FP8 W8A8（在线量化）落地过程中发现并修复的 6 个缺陷、判定依据与修复取舍，以及对"非缺陷"（验收口径）的判定过程。

- [1] 结论摘要
- [2] 测试与基准结果
- [3] 缺陷明细
  - [3.1] Triton 内核引用非 constexpr 全局
  - [3.2] `model.to(dtype=...)` 摧毁量化参数
  - [3.3] per-block 内核 load 缺 M/N 掩码 → 非法访存
  - [3.4] `_scaled_mm` recipe 不匹配 → per-tensor 路径必崩
  - [3.5] `drain_engine` 谓词取反 → 预热从未生效
  - [3.6] MoE 池化缓冲别名 → 测试假通过（测试侧）
- [4] 非缺陷：验收容差不可达
- [5] 方法论：如何区分"内核缺陷"与"方案固有误差"
- [6] 改动清单
- [7] 遗留项

## 1. 结论摘要

三条验收命令（kernel 数值对齐 / FP8 端到端 / BF16 基线）现已全部通过。过程中修掉 6 个缺陷，其中 4 个是**功能性崩溃或静默错误**，2 个是**测量/测试口径失效**——后者尤其值得记取：预热从未生效、MoE 断言自我比较，两者都会让结果"看起来正常"。

最终 FP8 与 BF16 基线持平（TTFT +8%，decode −2%）。修复前 FP8 的 TTFT 是 1442.8 ms，比基线差 14 倍，但那 1.35 s 不是"慢"，而是**被计进了错误的位置**（见 §3.5）。

## 2. 测试与基准结果

环境：单卡 RTX 4090（sm_89, 24 GB），Qwen3-0.6B，PyTorch 2.13.0+cu132 / Triton 3.7.1 / FlashInfer 0.6.15 / CUDA 13.2。

### 2.1 kernel 数值对齐（`tests/gpu/test_quant_fp8.py`）

修复前 `1 failed`（`AssertionError: block fp8 M=1 rel=0.0357`），修复后：

| 用例 | rel 实测 | 验收界 |
|------|---------:|-------:|
| block fp8 M=1 / 7 / 128 / 512 | 0.0357 / 0.0329 / 0.0386 / 0.0385 | < 0.08 |
| tensor fp8（`_scaled_mm`） | 0.0314 | < 0.08 |
| moe fp8 M=16 | 0.0600 | < 0.12 |
| moe fp8 M=1（decode 形状） | 0.0524 | < 0.12 |

口径 `rel = max|Δ| / max|ref|`。容差界的取值依据见 §4。

### 2.2 端到端（`python run.py --cg`，同 prompt）

| 档位 | Warmup | TTFT | Decode | E2E |
|------|-------:|-----:|-------:|----:|
| BF16（修复前） | 0.00s | 106.7 ms | 429.4 tok/s | 350.8 tok/s |
| BF16（修复后） | 0.09s | 83.7~85.1 ms | 439.9~440.2 tok/s | 372.3~373.1 tok/s |
| FP8（修复前） | 0.00s | **1442.8 ms** | 404.1 tok/s | 103.3 tok/s |
| FP8（修复后） | 1.45s | 90.2~100.1 ms | 427.0~430.8 tok/s | 353.2~362.3 tok/s |

区间为多次运行的波动范围。FP8 相对 BF16 的稳态开销约 +6% TTFT / −2% decode：per-block Triton GEMM 要跑 fp32 激活量化与 scale 校正，相对 cuBLAS bf16 路径略有一点点代价，量级符合预期。

## 3. 缺陷明细

### 3.1 Triton 内核引用非 constexpr 全局

**现象**：`pytest tests/gpu/test_quant_fp8.py` 直接 `NameError: Cannot access global variable SCALE_BLOCK from within @jit'ed function. Triton kernels can only access global variables that are instantiated as constexpr.`

**定位**：traceback 落在 kernel 体内引用的三个模块级常量：`fp8_gemm.py` 的 `SCALE_BLOCK` / `FP8_MAX`，`moe/kernels/triton.py` 的 `_GROUP`。host 侧代码引用同样的常量没问题。

**判定**：Triton 3.x 的语言约束（编译期常量折叠要求），不是数值或逻辑问题——报错发生在 trace 阶段，与数据无关。

**修复**：为内核侧新建 `tl.constexpr` 别名，host 侧常量保持原样：

```python
_K_SCALE_BLOCK = tl.constexpr(SCALE_BLOCK)
_K_FP8_MAX = tl.constexpr(FP8_MAX)
```

**为什么不直接把原常量改成 `tl.constexpr(...)`**：host 侧在用它们做算术。实测 `tensor / tl.constexpr(448.0)` 返回的是 constexpr 对象而不是张量，会静默改变 host 行为；`EVEN_K=(K % SCALE_BLOCK == 0)` 这类判断也依赖普通 int。别名把两种用途隔开，代价只是 4 行。

**验证**：测试从 `NameError` 推进到数值断言阶段。

### 3.2 `model.to(dtype=...)` 摧毁量化参数

**现象**：`python run.py --quantization fp8 --cg` 报 `triton.compiler.errors.CompilationError: Unsupported lhs dtype fp8e4nv`，指向 `tl.dot(x_q, tl.trans(w))`。

**定位**：Triton 的 fp8 `tl.dot` 要求两侧都是 fp8（`lhs.dtype.is_fp8() and rhs.dtype.is_fp8()`），所以是**某一侧没变成 fp8**。写探针监视 `w8a8_block_fp8_matmul` 的入参：

```
权重不是 fp8: shape=(4096, 1024) dtype=torch.bfloat16
weight_scale 也是 bfloat16，形状 (32, 8)
```

调用链（qkv_proj → `linear.py` → `fp8.py:293`）本身是对的，问题在权重加载。

**判定**：`create_weights` 的契约是权重为 `float8_e4m3fn`、scale 为 `float32`。实测两者都成了 bf16 —— 契约被下游某步覆盖。追到 `src/compute/worker.py`：

```python
load_model(model, self.config.model)                      # 内部 finalize_model() 已完成量化
return model.to(dtype=self.dtype, device=self.device)     # ← 把 fp8 权重和 fp32 scale 一起转成激活 dtype
```

`nn.Module.to(dtype=...)` 会遍历**所有**浮点参数，区分不了"激活参数"和"量化参数"。实测确认各层加载器都用 `param.data.copy_()` 写入、构造期已在 `default_dtype_context(self.dtype)` 下，所以整体 `.to(dtype)` 对普通参数本就是多余的。

**修复**：整体转换改为按"归属"判断——`quant_method.extra_params()` 里的参数（fp8 权重、fp32 scale、以及将来可能的 int32 元数据）保持自己的 dtype，其余浮点参数才转激活 dtype：

```python
def _to_device(self, model: torch.nn.Module) -> torch.nn.Module:
    model = model.to(device=self.device)
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        owned = {id(p) for p in method.extra_params(module)} if method is not None else set()
        for param in module.parameters(recurse=False):
            if id(param) not in owned and param.is_floating_point():
                param.data = param.data.to(self.dtype)
    return model
```

**为什么按"归属"而不是按 dtype 白名单**（例如跳过 `float8_e4m3fn`）：量化参数不只有 fp8——scale 是 fp32，将来 int4/awq 还有 int 参数与零点。用 `quant_method` 的归属信息判断，对后续量化方法是通用的。

**验证**：224 次 fp8 matmul 调用，0 次非 fp8 权重。

### 3.3 per-block 内核 load 缺 M/N 掩码 → 非法访存

**现象**：修完 §3.2 后 `--cg` 崩溃 `torch.AcceleratorError: CUDA error: an illegal memory access`，位置在 autotuner 的 `do_bench` → `di.synchronize()`——即在**编译期试跑**阶段就炸，说明某个候选配置本身越界。

**定位**：`CUDA_LAUNCH_BLOCKING=1` 把错误收敛到 `fp8_gemm.py` 的 kernel launch。再用 spy 记录每一次调用的形状：

```
[1348] M=160 K=3072 N=1024 x.stride=(3072,1) contig=True w.dtype=float8_e4m3fn ws=(8,24)
```

崩溃发生在第 1348 次调用，而 **M=176 / 240（同样非 128 整数倍）的调用却存活** —— 这个"同形状有的炸有的不炸"是典型信号：越界读落在 torch 缓存分配器已映射的页内时不报错，换一次分配布局就会踩到未映射页。

**判定**：store 有掩码而 load 没有：

```python
mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)   # store 有
x = tl.load(x_ptrs)                                    # load 没有
```

BLOCK_M=128 时，M=160 的第二个 M 块会读第 128..255 行，越界 95 行。N 方向因为 N 常是 128 的倍数而没暴露——**只修 M 会留下同类隐患**。这是内核缺陷而非配置问题：任何 M 非 BLOCK_M 整数倍的输入都该是合法的。

**修复**：load 与 store 用同一套掩码，`EVEN_K=False` 分支再叠 K 方向掩码：

```python
m_mask = offs_m[:, None] < M
n_mask = offs_n[:, None] < N
if EVEN_K:
    x = tl.load(x_ptrs, mask=m_mask, other=0.0)
    w = tl.load(w_ptrs, mask=n_mask, other=0.0)
else:
    k_mask = offs_k[None, :] + k < K
    x = tl.load(x_ptrs, mask=m_mask & k_mask, other=0.0)
    w = tl.load(w_ptrs, mask=n_mask & k_mask, other=0.0)
```

被掩掉的元素写 0，fp32 累加后不会影响有效输出（对应 `x_amax` 取整行时行内非法列已被清零，量化 scale 也一致）。

**验证**：M=1..100 的最小复现脚本全通过；补充实测崩溃形状 (160, 3072, 1024)、(176, …)、(240, …)、(1, 2048, 1024)、(7, …)、(133, 1024, 6144) 均正确。

**为什么之前没抓到**：`compute-sanitizer` 在本机起不来（NCCL fatbinary 报错），而 torch 分配器内部的越界对普通运行不可见——只有 `--cg` 的 graph 内存布局才把它变成致命错误。

### 3.4 `_scaled_mm` recipe 不匹配 → per-tensor 路径必崩

**现象**：dense per-block 段通过后，per-tensor 段（`torch._scaled_mm`）报 `RuntimeError: Invalid scaling configuration.`

**定位**：枚举 `scale_a` / `scale_b` 形状组合（mat2 用列主序 view，与真实调用一致）：

| scale_a | scale_b | 结果 |
|---------|---------|------|
| `[M,1]` | 标量 / `[1]` | ✗ Invalid scaling configuration |
| `[M,1]` | `[1,N]` | ✓ |
| `[1]` / 0-dim | `[1]` / 0-dim | ✓ |

**判定**：torch 2.13 只接受两种 recipe——**per-tensor × per-tensor**，或 **rowwise × colwise**（两侧都量化到行列粒度）。原实现是 `scale_a=[M,1]`（per-token 激活）+ 标量 `scale_b`（per-tensor 权重），属于两种 recipe 的混搭，被调度器直接拒绝。这是 torch API 的约束变更，不是本项目的逻辑错误。

**修复**：把标量权重 scale 广播成连续的 `[1,N]`，落到 rowwise × colwise 上：

```python
scale_b = w_scale.reshape(1, 1).expand(1, w_fp8.shape[0]).contiguous()
```

**为什么不是改成 per-tensor 激活量化**（那是上游 vLLM 对 per-tensor 权重的做法，只需一次 `_scaled_mm`、零额外分配）：本项目文档里明确写的是 per-token 动态激活量化，per-token 对含离群值的行精度更好。广播方案在数学上与原式 `a_s ⊗ w_scale` 完全等价，只是把标量铺满一行。

**注意点**：必须是**连续**张量——`expanded` 的 stride-0 视图同样被拒（实测）。每层额外分配一个 `[1,N]` fp32（4096 元素 = 16 KB），相对该路径每次都要分配的 `[M,K]` fp8 激活可忽略。

**顺带**：补了 `M == 0` 早退，与 per-block 路径对齐（空 batch 时 `_scaled_mm` 无意义）。

**验证**：M=1/2/7/64/512/2049、池化 `out=`、M=0 全部通过；rel 与手工 `(a_q.float() @ b.float()) * a_s * w_s` 逐位一致（同为 0.0340），确认语义未变。

### 3.5 `drain_engine` 谓词取反 → 预热从未生效

**现象**：两条端到端命令都打印 `Warmup: 0.00s`，但 FP8 的 TTFT 高达 1442.8 ms（基线 106.7 ms）。

**定位**（这条最曲折，值得完整记录）：

1. 先排除"没预热到 fp8 形状"：`_probe_cg_footprint` 在 engine 构造期会跑一次真实前向，但 FP8 特有的 Triton autotune 不该被它覆盖。
2. 写探针复刻 `run.py` 的 `_warmup`。**第一个探针照抄了 `drain_engine(eng.has_unfinished, eng.step)`，得到 `warmup_engine = 0.000s`——正确复现了 bug。**
3. **第二个探针手写了 `while eng.has_unfinished(): eng.step()`，得到 1.590s**——看起来"预热明明是好的"，与探针一矛盾。这个矛盾一开始被我归因于探针噪声，实际是探针二把 bug 绕过去了。
4. 最后用代理方式在**真实 run.py 代码路径**上埋点：patch `run.warmup_engine`，并从 bound method 反取引擎对象打印 client 类型与调度队列长度：

```
[probe] client=InprocClient engine=Engine
  add_request(5) -> unfinished=True waiting=1 running=0
  drain -> 0.0000s waiting=1 running=0     ← 请求在队列里，循环却没跑
```

**判定**：`drain -> 0.0000s` 时 `waiting=1`——请求明明在队列中。对照函数定义：

```python
def drain_engine(is_finished, step):
    while not is_finished():     # 参数语义是"已完成"
        step()
```

调用方 `run.py:115` 传的是 `engine.has_unfinished`（`scheduler.is_finished()` 的取反）。`while not has_unfinished()` 恒为假，**循环体一次都不执行**。语义反了，是明确缺陷。

**影响**：预热从未生效。三组 dummy 请求一直滞留在 waiting 队列里，随后第一次真实 `engine.step()` 把它们和新请求一起 prefill；FP8 首次调用触发的 ~1.4 s Triton autotune（每个 (M,N,K) 4 个候选配置试跑）全部落在**计时区间内**。这也解释了为什么基线看起来"还行"——构造期的 footprint 探测已经把 bf16 路径的编译成本吸收掉了，只剩残留 dummy 请求的干扰。

**修复**：按唯一调用点的语义对齐（全项目只有 `run.py` 一处调用）：

```python
def drain_engine(has_unfinished: Callable[[], bool], step: Callable[[], None]) -> None:
    """反复 step 直到没有未完成请求（谓词为 ``Engine.has_unfinished``）。"""
    while has_unfinished():
        step()
```

**为什么改函数而不是改调用点**：调用点写 `lambda: not engine.has_unfinished()` 需要引入一个 lambda，可读性更差；且 `Engine` 的公开接口就是 `has_unfinished`，让 `drain_engine` 与之对齐比让每个调用方包一层取反更自然。参数名一并改掉，避免下一个人再踩同一个坑。

**验证**：`drain -> 0.0437s`（2 步）、`waiting=0`；`warmup_engine` 正确返回 1.45s（FP8）/ 0.09s（BF16）并被排除在计时外；FP8 TTFT 1442.8 → 90.2 ms，基线 TTFT 106.7 → 85.1 ms。

### 3.6 MoE 池化缓冲别名 → 测试假通过（测试侧）

**现象**：MoE 段打印 `moe fp8 (M=16) rel=0.0000`——fp8 量化**逐位相同**于 bf16 参考。这是不可能的结果。

**定位**：直接比较两个对象：

```
same tensor object : True
same data_ptr      : True
```

**判定**：两次调用返回的是**同一个张量对象**。`fused_moe_forward` 返回按形状池化的复用缓冲（`_pool`，键为 `(设备, M, H, dtype)`，与层无关），第二次（fp8）调用原地覆写了第一次（bf16）的结果，于是 `rel(got, ref)` 变成张量与自身比较，恒为 0。用 clone 快照重测，真实误差是 0.0645 —— 旧界 0.06 是**卡在小数点后第 4 位的失败**，只是被别名掩盖了。

**修复（测试侧，API 维持原样）**：给参考值加 `.clone()` + 注释说明契约。

```python
ref = fused_moe_forward(...).clone()
# 必须 clone：fused_moe_forward 返回按形状池化的复用缓冲（调用方须立即消费），
# 不 clone 的话下面 fp8 那次调用会原地覆写 ref，rel(got, ref) 恒为 0（假通过）。
```

**为什么不改 API 让它返回副本**：额外一次拷贝实测约 6.8 µs（memcpy）/ 9.3 µs（clone）每 MoE 层，占 30B-A3B 形状下单层 MoE forward 的 3.6% / 4.9%（M=1），M≥128 时降到 0.5%。这些数字几乎全是 CPU 侧 launch 开销而非带宽——M=1 时载荷只有 4 KB；对照 M=8192/H=4096 的 128 MB 拷贝是 146 µs = 877 GB/s，基本贴着 4090 上限。decode 走 CUDA Graph replay 后每节点开销会塌到 1~2 µs，实际代价 ≤1%。

真正要留心的是显存：多一份 `[M,H]` 缓冲/层，graph capture 时每层落在不同地址，48 层 × 8 MB（M=2048）≈ 400 MB。综合判断：引擎内所有调用方都是立即消费（残差相加等），别名是设计使然；为"防呆"付出这个代价不值。**结论：维持原样，把契约写进注释与测试。**

后续若确有调用方需要长期持有结果，推荐加一个可选 `out=` 参数（需要稳定缓冲的调用方自己传），而不是默认拷贝——零性能代价地解决同一问题。

## 4. 非缺陷：验收容差不可达

`tests/gpu/test_quant_fp8.py` 原本要求 dense `rel < 0.03`、MoE `rel < 0.06`，实测分别约 0.033~0.039 与 0.052~0.060。**判定这不是内核缺陷，而是验收口径超出了 e4m3 方案的物理下限**，依据是两组"参照系"实验：

**dense（per-block）** —— 用 torch 手写等价量化-反量化作为"理想实现"：

| 对比 | rel |
|------|----:|
| 内核 vs bf16 参考 | 0.0357 |
| 内核 vs 理想 fp8 模拟 | **0.0075**（≈ 输出 bf16 舍入） |
| 理想 fp8 模拟 vs bf16 参考 | **0.0395** |

内核与理想实现只差 0.75%，而理想实现本身离 bf16 就有 3.95% —— 误差来自量化方案，不是内核。20 个随机种子下 M=1 的最小/中位/最大为 0.0290 / 0.0388 / 0.0483，**19/20 超过 3%**。

**MoE** —— 用同一内核做对照，隔离两个误差源：

| 误差源 | max 口径 | mean 口径 |
|--------|---------:|----------:|
| 权重量化（反量化权重 vs bf16 权重） | 0.0444 | 0.0454 |
| 激活量化（fp8 vs 反量化权重） | 0.0405 | 0.0450 |
| 合计 | 0.0645 | 0.0630 |

**为什么误差不会被 K 平均掉**：e4m3 每元素相对误差平均约 3~4%（3 位尾数，最近舍入半 ulp≈6.25% 上限）。点积里信号按 √K 增长，随机符号的误差也按 √K 增长，比值不随 K 收敛——所以 ~4% 是该方案在随机数据上的**下限**，与 K 无关。

**最终取值**：dense/tensor `0.03 → 0.08`（约 2.1x 余量），MoE `0.06 → 0.12`（2.0x）。界不是"调到能过"——漏做 scale 校正、走错量化分支这类真 bug 会给出 >0.10 的误差，仍会被拦住。

## 5. 方法论：如何区分"内核缺陷"与"方案固有误差"

本次能快速定性，靠的是三个可复用的手段：

1. **先建参照系，再判对错。** 数值对不上时，不要直接放宽容差，先用 torch 手写一份"理想实现"（同样的量化粒度、同样的 scale 语义，但用朴素算子算），把总误差拆成"内核 vs 理想"和"理想 vs 高精度"两段。前者大 → 内核缺陷；后者大 → 方案固有。dense 段 0.0075 vs 0.0395 的对比一出来，结论就唯一了。
2. **用同内核对照组隔离变量。** MoE 手工模拟连 bf16 基线都对不上（0.21），说明模拟本身不可信——此时改用"同一个内核、只换输入（fp8 权重 vs 反量化回 bf16 的权重）"的对照，差值就只剩激活量化一项，无需建模整个 MoE 数据流。**参照系不可信时要换参照系，而不是硬解释差异。**
3. **埋点看状态，不看猜测。** §3.5 卡了最久，转折点是停止推理"预热为什么不生效"，改为打印真实对象的状态（client 类型、`waiting`/`running` 队列长度）。`drain -> 0.0000s` 旁边跟着 `waiting=1`，矛盾自体就指出了 bug 所在。
4. **警惕"逐位相同"。** 量化的结果不可能逐位等于未量化结果；这类异常几乎总指向别名、缓存或比较对象搞错。同理，§3.3 里"同形状有的炸有的不炸"指向的是越界读落在已映射页内。
5. **注意自己的探针有没有把 bug 抄进去。** §3.5 的两个探针给出相反结论，正是因为探针二把有缺陷的写法换成了正确写法。探针与产品代码用**同一个未验证的辅助函数**时，二者会一起错——要么两边都复现，要么两边都掩盖。

## 6. 改动清单

`src/` 与 `utils/` 共 4 个文件（+53/−15），测试 1 个文件：

| 文件 | 位置 | 内容 |
|------|------|------|
| `src/quant/kernels/fp8_gemm.py` | 31-33 | 新增 `_K_SCALE_BLOCK` / `_K_FP8_MAX` constexpr 别名 |
| | 149-164 | load 补 M/N 掩码；内核体改用 `_K_*` |
| | 247-262 | per-tensor 路径：`scale_b` 广播为连续 `[1,N]`；补 `M == 0` 早退 |
| `src/moe/kernels/triton.py` | 24-25, 134, 150 | 新增 `_K_GROUP` 别名并替换内核体引用 |
| `src/compute/worker.py` | 145-166 | `model.to(dtype=...)` → `_to_device()`：按 `quant_method` 归属保护量化参数 |
| `utils/run_utils.py` | 82-85 | `drain_engine` 谓词语义修正（`has_unfinished`） |
| `tests/gpu/test_quant_fp8.py` | 64, 84, 106, 135, 147 | 容差 0.03→0.08 / 0.06→0.12；两处参考值 `.clone()`（含 MoE M=1 的 159） |

复现命令：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=.
pytest tests/gpu/test_quant_fp8.py -q --tb=short
python run.py --model ~/huggingface/Qwen3-0.6B --quantization fp8 --cg
python run.py --model ~/huggingface/Qwen3-0.6B --cg
```

## 7. 遗留项

- **`fused_moe_forward` 的缓冲契约**：目前只在文档与测试里注明"返回值仅在下一次 MoE 调用前有效"。若出现需要跨调用持有结果的调用方，建议加 `out=` 参数而非默认拷贝（§3.6）。
- **per-tensor 路径的每层 `[1,N]` 分配**：当前每次调用分配 16 KB。若该路径进入热路径，可在加载期把广播后的 scale 作为 buffer 缓存到层上。
- **验收口径**：本次把 max 口径的界放宽到了 0.08 / 0.12。若希望界更紧，应改用 mean 口径（`mean|Δ|/mean|ref|`）——它对单个最坏元素不敏感，能同时收紧到 ~0.05 又不误报。
- **`pytest` 未覆盖的路径**：`--quant-kv-cache`（FP8 KV cache）本次未验证。
