# ZOS（Zero-Overhead Scheduling）技术报告

> 把"下一拍的提交"提前到"上一拍结果处理"之前，让 CPU 的调度工作藏进 GPU 计算里。
>
> **最终效果**：普通 decode 0.6B **p50 −4.7%**、8B −1.0%（greedy 逐 token 100%）；
> 投机路径**正确性通过**（accept_len / 逐位置曲线 / 128 token 逐位一致），但性能
> **0.948×** —— 负收益，因此默认关。
>
> 代码：`src/control/async_decode.py`、`src/compute/decode_inputs.py`、
> `src/spec/dspark/padded_cache.py`、`src/spec/verify.py`、`src/compute/model_runner.py`
> 测试：`tests/unit/test_zos_{advance,accept,fallback}.py`、`tests/gpu/test_zos_e2e.py`
> 基准：`bench/zos_bench.py`

---

## 1. 问题：decode 的主机空档

稳定 decode 每拍原本是串行的：

```
GPU 前向+采样 N → D2H 回读 token N → CPU 更新状态/准备输入 → GPU 前向 N+1
                ↑ GPU 在这段空等（nsys 上就是两次 forward 之间的 gap）
```

时间可以拆成两段：

```
G = GPU 真正在算的时间
C = CPU 在算、而 GPU 空闲的时间

串行：步时 = G + C
重叠：步时 ≈ max(G, C)
⇒ ZOS 的收益上限 = C / (G + C)
```

**这条公式决定了 ZOS 的适用范围**：它不是"总能加速 20%"的优化，而是一个放大器——
只有本来就存在明显主机空档时才有东西可放大。

---

## 2. 与上游实现的关系

思路来自 SGLang v0.4 的 Zero-Overhead Batch Scheduler，实现对照 **DeepInfer**
（DeepSeek 2026-10 开源的推理引擎，`DEEPINFER_ZERO_OVERHEAD_SCHEDULING` 在
`NORMAL`/`COMPLETION` 引擎类型下 `setdefault=1`，即默认开）。

从 `csrc/engine/zos.cpp` 读到三个关键设计，直接决定了我们的架构：

| 上游做法 | 含义 |
|---|---|
| `prepare_token_inputs` 只填 **fresh 行**，matched 行留 0 由 GPU patch | 跨拍重叠时，输入构造必须下沉到设备 |
| `seqlens_q / seqlens_k / cu_seqlens_q` 上传到 GPU、由 kernel 修正后再异步回传 | 长度是"可事后修正的标量"，不是"必须提前知道的常量" |
| launch 路径有 `forbid_cuda_sync()` 守卫 | 禁的是**阻塞同步**（`.item()`），pinned + event 的异步回传是允许的 |

第三条纠正了我早期的一个错误判断（见 §5.3）。

---

## 3. 实现：四块

### 3.1 `advance_decode`：设备侧的输入推进

普通 decode 的输入由 CPU 填好再 H2D。ZOS 改成在设备缓冲上就地推进：

```python
step  = advances[row]                 # 1（普通）或 1+j（投机）
pos   = positions[row] + step
page  = tables[row, pos // PAGE]
ids[row], positions[row], slots[row], lengths[row] = token, pos, page*PAGE+pos%PAGE, pos+1
```

**设计要点**：`advances` 从第一天就是"**每行一个**"，所以普通 decode（全 1）与投机
（`1+j`）共用同一个 kernel，不必写两个。

### 3.2 `accept_greedy_device`：设备侧接受裁决

投机原本在 verify 末尾有硬同步点：

```python
preds = self.model.compute_logits(hidden).argmax(dim=-1).tolist()   # ← 全流水线排空
```

设备版只做 elementwise + `cumprod` + 一次 gather：

```python
match    = (preds[:, :k] == drafts)
j        = match.cumprod(1).sum(1)        # 最长可接受前缀
next_tok = preds.gather(1, j)             # 下一拍输入
advances = 1 + j
```

一个能省事的观察：`accept_greedy` 返回的 `drafts[:j] + [preds[j]]` **最后一个元素恒为
`preds[j]`**，所以下一拍的 anchor 不需要拼 tokens。

### 3.3 `AsyncDecode` 控制器

```python
def step(self):
    if self.pending is None:
        sched = core.scheduler.schedule()
        if not self._eligible(sched):
            return core._step_sync(sched)      # 复用已排好的批（见 §4 的 bug 1）
        self.pending = self._submit(sched)     # 只发射 + 异步 D2H
        return [], len(sched.decodes)          # 首拍不返回（落后一拍）

    prev = self.pending
    nxt = self._submit_ahead(prev) if self._can_lookahead(prev) else None
    outputs = self._consume(prev, nxt)         # 唯一的同步点
    self.pending = nxt
```

**四个 drain**（任一成立就回普通路径）：

| 条件 | counter | 原因 |
|---|---|---|
| `waiting` 非空 | `waiting_drains` | 新请求要 prefill，批次结构会变 |
| `running` 行序与上一拍不一致 | `batch_drains` | 行序变了，`advance` 会写错行 |
| 有序列即将结束 | `terminal_drains` | 下一拍会超 `max_tokens` |
| KV 页不够 | `memory_drains` | 提前一拍要预留到 `num_cached + 1 + k` |
| 超过 32 拍未重锚 | `ub_drains` | `ub` 只增不减，长生成会一路退回 eager |

**两个 fence**（必须先等在途那拍跑完，再动 CPU 状态）：

| 情况 | counter | 原因 |
|---|---|---|
| 本拍产出 EOS / 到 `max_tokens` | `eos_fences` | `update` 会释放页，在途那拍还在写 |
| radix 跨页（可能 remap） | `prefix_fences` | 页号变了，在途那拍的 slot 已失效 |

### 3.4 投机路径（方案 B）

投机每拍推进 `1+j`，而 `j` 只在设备上。凡是"用 host int 做切片/偏移"的地方都会变成
阻塞同步。方案 B 的核心是**把需要 host 知道的量全部换成上界**：

| 量 | 原来 | 方案 B |
|---|---|---|
| 草稿缓存追加量 | `1+j`（设备） | **`m = 1+k`**（host 恒定） |
| 有效位 | `valid[s : s+seg]` | **`valid[s : s+m] = arange(m) < seg_r`**（host 切片 + 设备掩码） |
| verify 桶 | 精确 `num_cached + m` | **`ub += m`**（host 上界） |
| `crop` | 截到 `accepted+1` | **消失**（无效位靠 mask） |

配套三块：

- `advance_verify`：按设备 `starts` 写 `[bs, m]` 的 positions/slot_mapping，
  `cu_seqlens_k = cumsum(starts + m)`（前缀和必须设备侧做）；
- `propose_batch_ids(..., segs=)`：草稿的位置编码用 `(starts - segs)` broadcast，
  有效位走设备掩码；
- `_verify_zos`：跳过 `build_prefill`，ids 用 `cat([anchor, drafts])` 在设备上拼。

代价（可算）：每拍浪费 `k - j` 个位置，累计约 `1.33 × T`，所以草稿缓冲开到
`1.5 × context_len` 即可，**不需要压缩**。

---

## 4. 遇到的问题与解决（13 个）

### 4.1 三个阻塞性 bug：接线时"没验证接口契约"

第一版接线在 GPU 上直接失败：

| 现象 | 根因 | 修法 |
|---|---|---|
| `len(token_ids)=30 vs num_cached+1=1`（30 是 prompt 长度，说明 prefill 从未执行） | 回落路径**重复 `schedule()`**：控制器先排一次判断能否进流水线，`_step_sync()` 又排一次 → 刚落地的 prefill 被当成 decode 行 | `_step_sync(sched=None)` 接受已排好的批 |
| `len=1 vs num_cached+1=6` | ahead 批里装的是 **`SeqSchedule`**，而 `ScheduledBatch` 装的是 **`Sequence`**；`to_wire()` 按 Sequence 取字段 | `ScheduledBatch(decodes=list(seqs))` |
| `len=33 vs num_cached+1=32` | 上一拍 `update` 末尾把 `num_scheduled_tokens` 清 0，ahead 拍没经过 `schedule()` 重设 | `consume` 里 `update` 前重置为 1 |

**第二个最有教育意义**：控制器里有 `cur is not old` 的**身份比较**。如果 `seqs` 是
`SeqSchedule`，这个比较**永远为真** → 永远 `batch_drain` → ZOS 永不生效，但**输出仍然
正确**。所以 GPU gate 必须同时断言 `lookahead_submitted > 0`。

### 4.2 四个静默隐患：读代码发现

| 隐患 | 后果 | 修法 |
|---|---|---|
| EOS fence 用 `s.last_token` | 那是**本拍输入** token，比 EOS 会漏 fence | 先 `read_samples`，用本拍**产出**判断 |
| 丢弃在途 ticket 不清回读 FIFO | 下次 `read_samples` 弹出过期数据，token 记到别人头上 | 新增 `discard_pending()`，与 `sync_samples()` 成对 |
| KV 预留逐行判定 | 多行同时跨页时逐行都能过、合计超预算 | 按 `sum(needed_append) > available` 判总量 |
| radix 内部 remap 漏刷页表 | `_sync_block_tables` 只看首/末/长度 | radix 后端每拍都刷（保守） |

### 4.3 投机接线的五个 bug（上机暴露）

| # | 现象 | 根因 |
|---|---|---|
| 1 | `AttributeError: _input_views` | `bind_verify` 抄了 v3 的设计，我们是常驻缓冲，没有视图函数 |
| 2 | `RuntimeError: previous round has no device tokens` | `tokens_dev is None` 的检查挡在 spec 分支前，而投机拍产出是 `_SpecSamples` |
| 3 | `_zos_start` 多推一个 `1+j` | 同步拍存 `num_cached + (1+j)`，`_verify_zos` 又加一次 `advances` |
| 4 | 跨页时按旧页表算 slot | `_verify_zos` 绕过 `build_*`，也就绕过了它的页表同步 |
| 5 | `accept_len 5.548 → 5.493`（**不报错**） | 首拍走 `propose()`（`DynamicCache`），ahead 走 `propose_batch_ids()`（`PaddedKVCache`），两个缓存不是同一个对象；不搬 prompt K/V，草稿只看得到最后一个 ctx 段 |

第 5 个的第二个坑：**transformers 4.57 的 `DynamicCache` 用 `layers[i].keys/values`，
不是旧的 `key_cache`**。只兼容旧布局时 seed 会**静默返回 0**——所以第一次修完统计完全
没变，看起来像"这个 bug 不存在"。

### 4.4 两个"一上机就崩"

| 现象 | 根因 |
|---|---|
| `AttributeError: 'ModelRunner' object has no attribute 'max_bs'`（**影响所有模型**） | `_zos_ub` 用 `self.max_bs` 初始化，而它在这行之后才赋值 |
| `TypeError: argument of type 'int' is not iterable` | 真实 `Scheduler.eos` 是 int，`tid in eos` 只对 set/frozenset 成立 |

### 4.5 三次方向返工（最贵的部分）

| 判断 | 实际 |
|---|---|
| "设备侧裁决就够了，投机就能进 ZOS" | 不够：verify 的 `positions/slot_mapping/cu_seqlens` 还是 CPU 按 `num_cached` 建的 |
| "把草稿 attention 换成 varlen 就行" | 不行：FA 的 `cu_seqlens` 要求**连续 packed**，我们是 per-row 固定槽位，只在 `len == cap` 时相容 |
| "读 j 回主机 = 放弃 ZOS" | 错：`forbid_cuda_sync` 禁的是**阻塞同步**，pinned + event 的异步回传允许 |

三次的共同根因：**先按直觉设计、后核上游实现**。`zos.cpp` 里的 `fresh_input_ids` /
`remap` / `forbid_cuda_sync` 三个符号能一次性避免前两次返工。

---

## 5. 最终效果

### 5.1 普通 decode（默认开）

4090D，greedy，n=128，`--cg`：

| 配置 | p50 步时 | 降幅 | decode tok/s | 一致率 | lookahead | peak |
|---|---|---|---|---|---|---|
| 0.6B bf16 bs=1 | 2.06 → **1.97 ms** | **−4.7%** | 481.4 → **504.5** | 100% | 382 | 3.71 GB |
| 8B fp8 bs=1 | 10.47 → **10.36 ms** | −1.0% | 95.4 → **96.5** | 100% | 382 | 11.76 GB |

p99：0.6B 2.12→2.02 ms；8B 10.54→10.38 ms。
`drains` 只有 terminal，`memory`/`waiting`/`batch` 全为 0 —— 流水线确实提前提交了。

**为什么只有 4.7%**：我们的 decode 路径本来就瘦（采样早就是异步 D2H、前向全在
CUDA Graph 里、`build_decode` 只填 5 个字段 + 一次 H2D），所以 `C` 只有一百多微秒。
收益上限就是 `C/(G+C)`。

### 5.2 投机 + ZOS（默认关）

8B FP8 + dspark k=7，transformers 5.10.2，n=128：

| 配置 | decode tok/s | 相对 |
|---|---|---|
| spec（非 ZOS） | 332.6 | 1.000× |
| spec + ZOS，**eager verify** | 148.5 | 0.446× |
| spec + ZOS，**ub + verify 图** | **315.4** | **0.948×** |

- **正确性逐位一致**：`accept_len 6.000`、`accept_rate 71.43%`、
  `proposed/accepted/steps 469/335/67`、逐位置曲线、128 个 token；
- eager → 图版提升 **2.12×**，全部来自"把 verify 图拿回来"。

**为什么是负收益**：`C` 已被设备侧裁决与 CUDA Graph 压到几十微秒，而 `G` 有 18 ms，
可省的几乎为零；ZOS 反而倒贴三笔成本：

| 新增开销 | 落在哪 |
|---|---|
| `advance_verify` + `cat` 拼 ids + `bind_verify` | G |
| 每拍一次页表 H2D | C |
| **草稿 KV 被撑大**（按 m 追加而非 1+j，每拍多 2 个位置，累计 ~30% 注意力浪费） | G |

> 结论：**投机路径不适合 ZOS，除非先有东西增大 `C`**（例如调度逻辑变重），而不是继续
> 削它的开销。这条结论比一个数字更有价值——它指明了优化方向。

---

### 5.3 与公开实现的对照：为什么只有 4.7%

公开报道的 ZOS 收益集中在 **1.1×（非投机）** 与 **10~20%（投机）**：

| 来源 | 场景 | 收益 | 备注 |
|---|---|---|---|
| **SGLang v0.4 官方 blog**（2024-12） | Llama-3.2-3B，500 请求，输入 4096 / 输出 2048，serving 压测 | **1.1× vs 自身（v0.3）**、1.3× vs 其他 SOTA | Triton attention backend；GPU 与 batch 未写明 |
| 同上（收益条件） | — | "**小模型 + 大 TP 时收益最显著**" | 原文：*most significant on small models and large tensor parallelism sizes* |
| **Google Vertex AI（EAGLE-3 生产化）** | **投机解码**的 draft→verify→draft 循环 | **10~20% 全面提升** | 他们消的是"同步调度器"的气泡 |
| SGLang blog 的泛化说法 | 未优化的引擎 | "最多一半时间花在 CPU 开销上" | 泛指，不是 v0.3 的实测值 |
| **DeepInfer**（ZOS 命名来源） | 生产引擎 | **未公布单独收益** | 只确认默认开启 |

把我们的数字代进同一个公式 `收益 = C/(G+C)`，差距就说得通了：

| | SGLang / Vertex | 我们 |
|---|---|---|
| `C` 的构成 | radix 前缀匹配 + 每请求簿记 + Python 调度 + IPC | `build_decode` 填 5 个字段 + 一次 H2D |
| CUDA Graph 覆盖 | decode 有，**verify 通常没有** | decode **和** verify 都有 |
| 采样回读 | 同步（Vertex 明确称之为"气泡"） | 异步 D2H（ZOS 之前就有） |
| 因此收益 | 10~20% | 1~5% |

**关键结论：ZOS 与 CUDA Graph 对同一块 `C` 是替代关系，不是叠加关系。** 我们已经用
异步回读 + 图把 `C` 压到几十微秒，所以 ZOS 没有东西可藏；投机路径更是如此——设备侧
裁决 + verify 图之后 `C ≈ 0`，ZOS 只增加"设备侧元数据构造 + 草稿 KV 撑大"的成本，
于是净收益 −5%（§5.2）。

**一个反向信号值得记下**：SGLang 明确说"大 TP 收益更显著"（TP 越大每 rank 的 `C` 越
重），而这正是我们唯一排除的场景（§7 的 `tp>1`）。换句话说——**在单卡上 ZOS 本来就
只值 4.7%**；若要让它体现价值，位置是 TP>1，而不是继续在单卡上抠。

---

## 6. 方法论沉淀

| 教训 | 具体化 |
|---|---|
| **"能 import + CPU 单测过" ≠ "能跑"** | 13 个问题里有 7 个在"类型契约"与"函数副作用"上（`ScheduledBatch` 装什么、`schedule()` 有副作用），CPU 单测碰不到 |
| **GPU gate 要断言"功能真的生效"** | 只断言"输出对不对"会放过"永不生效"的 bug（§4.1 的第二个） |
| **并发状态机要先问"谁清了我"** | `num_scheduled_tokens` 被上一拍 `update` 清零，这类"上拍清、下拍用"是异步流水线最易漏的 |
| **冗余状态必须成对更新** | `_lens_host` 与 `lens`、`ub` 与 `starts`，漏一个就是"静默算错" |
| **比较要比不变量，不比布局** | 设备路径与 host 路径的 valid 矩阵**本来就不该相等**（一个裁了 block 段、一个没裁） |
| **默认开要有可观测性兜底** | `--no-zos` + 完整 counters：出问题一行关掉，也能分辨"没生效"还是"生效但收益小" |
| **fail loud 优于静默降级** | `zos_spec=True` 但接线未完成时直接抛错，不猜 |

---

### 6.5 长上下文（n=512）暴露的两个问题

n=128 一切正常，拉到 n=512（跨过第一个 256-token KV 块）才暴露：

**① KV 预留少算了"下一拍的起点"——已修**

```
在途那拍起点 ≤ num_cached + (1+k)     ← 由设备侧 advances 决定
再写 (1+k) 个位置
⇒ 需要 num_cached + 2(1+k)
```

原来只按 `1+k` 预留，跨块那一步的 slot 落到未分配块（表项 `-1` → 负 slot），Triton
kernel 直接 `illegal memory access`（修复前 n=256/512 **4/4 崩**）。修法是
`n = 2 * (1 + spec_tokens)`（普通 decode 的 advances 恒为 1，同样需要 2）。这是
`may_append` 的**瞬时预留**，等价于"多看一个 block"，不是永久膨胀。

回归：`tests/unit/test_zos_kv_reserve.py`（k=7 → 16，k=0 → 2）。

**② 投机 + ZOS 跨块后输出分叉——未解决**

| 配置 | 输出 vs dense spec |
|---|---|
| dense spec n=512 | 基准（accept_len 4.571） |
| ZOS spec n=512（verify 图） | **first diff = 262**（accept_len 3.957） |
| ZOS spec n=512（eager） | **first diff = 262**，且与图版 token 彼此一致 |
| plain ZOS n=512 | **与 sync 逐位一致** |

已排除：verify 图/桶（eager 与 graph 相同）、KV 预留（已修，且修前是崩溃、修后是分叉）、
块表未分配（修后不再有 `-1` slot，plain ZOS 同边界正常）。

范围收窄到：**spec-ZOS 的 `advance_verify` 在块边界上的元数据**（positions / slot_mapping /
cu_seqlens_k），与草稿、padded KV 无关（那些只影响草稿质量，不影响 verify 的 lossless 性）。

建议的排查顺序（比直接对 logits 有效，因为页表是离散可逐值对照的）：

1. 在跨边界那一拍 dump 三份：CPU 的 `seq.block_table`、设备 `block_tables[0,:k]`、
   `advance_verify` 实际查出的 page。**只要设备表 ≠ CPU 表，就是刷新问题，直接定案**；
2. 若相同，再比 `slot_mapping` 与同步路径 `slots_for_range` 在同一区间的值；
3. 都相同，最后才比 target logits。

**当前结论：长上下文不要开 `zos_spec`**（崩溃已修，但仍会分叉）。`--zos-spec` 默认关，
配置护栏（`tests/unit/test_zos_spec_config.py`）只保证组合合法，不保证这个场景正确。

---

## 7. 边界与未做

| 项 | 状态 |
|---|---|
| `tp=pp=1` | ✅ 支持 |
| `dp>1` | ✅ 天然支持（每副本独立，零协调；v3/v4 把它一起拒了，是过宽的判断） |
| `tp>1` | ❌ 缺"每 rank 同一张票"的协调层。**但这恰是 ZOS 收益最显著的场景**（每 rank 的 `C` 更重），若要让它体现价值应从这做 |
| `pp>1` | ❌ lockstep 下需 token/j 反向环，且收益上限只是"藏进流水气泡" |
| 温度采样 | ❌ ZOS 只处理 greedy（图内只有 argmax） |
| 草稿 KV 改 paged | ❌ 独立重构项：能彻底去掉 host 形状依赖与 padding 浪费 |
| `--zos-spec` CLI | ✅ `--zos-spec` / `--no-zos` |
