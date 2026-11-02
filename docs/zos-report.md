# ZOS（Zero-Overhead Scheduling）实现报告

> 把"下一拍的提交"提前到"上一拍结果处理"之前，让 CPU 的调度工作藏在 GPU 计算期间。
> 实测：0.6B **p50 步时 -4.7%**，8B **-1.0%**；greedy 逐 token 100% 一致。
> 配套：`src/control/async_decode.py`、`src/compute/decode_inputs.py`、
> `tests/unit/test_zos_{advance,accept,fallback}.py`、`tests/gpu/test_zos_e2e.py`、
> `bench/zos_bench.py`

---

## 1. 动机：decode 的串行链

稳定 decode 的每一拍原本是串行的：

```
GPU 前向+采样 N → D2H 回读 token N → CPU 更新状态/备输入 → GPU 前向 N+1
                ↑ 这段 GPU 在空等（nsys 上就是相邻两次 forward 之间的 gap）
```

但下一拍要的输入**大部分已经确定**：行序不变、位置 +1、KV 长度 +1、跨页时换一页。
唯一"未知"的 token 其实**已经在 GPU 的采样输出里**。所以顺序可以改成：

```
提交 forward N+1（用设备上的 token N）→ 处理 N 的结果 → 提交 sample N+1
```

CPU 于是**落后一拍**：它处理的是上一拍的结果，而 GPU 已经开始下一拍。步时从
`G + C` 变成接近 `max(G, C)`。

**这条思路的来源**：SGLang v0.4 的 Zero-Overhead Batch Scheduler，以及 DeepSeek
开源推理引擎 **DeepInfer**（2026-10 开源，`DEEPINFER_ZERO_OVERHEAD_SCHEDULING`
在 `NORMAL`/`COMPLETION` 引擎类型下 `setdefault=1`，即默认开启）。我们的实现参照
DeepInfer 的顺序与纠正链路，而不是只看结论。

---

## 2. 四层实现

### 2.1 `advance_decode`：设备侧推进元数据（`src/compute/decode_inputs.py`）

普通 decode 的输入由 CPU 填好再 H2D：

```python
pos  = s.num_cached_tokens
ids  = s.token_ids[-1]
slot = block_table[pos // B] * B + pos % B
csl  = pos + 1
```

ZOS 把它搬到设备上，**从上一拍留在缓冲里的值往下推**：

```python
step  = advances[row]                 # 1（普通）或 1+j（投机）
pos   = positions[row] + step
page  = tables[row, pos // PAGE]
ids[row], positions[row], slots[row], lengths[row] = token, pos, page*PAGE+pos%PAGE, pos+1
```

一个 CTA 处理所有行，零 H2D。**`advances` 从第一天就是"每行一个"**，所以普通
decode（全 1）与投机（`1+j`）共用同一个 kernel，不必写第二个。

### 2.2 设备侧接受裁决（`src/spec/verify.py::accept_greedy_device`）

投机路径原本在 verify 末尾有一个硬同步点：

```python
preds = self.model.compute_logits(hidden).argmax(dim=-1).tolist()   # ← 全流水线排空
```

设备版只做 elementwise 比较 + `cumprod` + 一次 gather：

```python
match    = (preds[:, :k] == drafts)
j        = match.cumprod(1).sum(1)        # 最长可接受前缀
next_tok = preds.gather(1, j)             # ← 下一拍输入，一次 gather
advances = 1 + j
```

**一个省略**：`accept_greedy` 返回的 `drafts[:j] + [preds[j]]` 的最后一个元素恒为
`preds[j]`，所以下一拍的 anchor 不需要把 tokens 拼出来。

### 2.3 `AsyncDecode` 控制器（`src/control/async_decode.py`）

```python
def step(self):
    if self.pending is None:
        sched = core.scheduler.schedule()
        if not self._eligible(sched):
            return core._step_sync(sched)   # ← 复用已排好的批
        self.pending = self._submit(sched)  # 只发射 + 异步 D2H
        return [], len(sched.decodes)       # 首拍不返回（落后一拍）

    prev = self.pending
    nxt = self._submit_ahead(prev) if self._can_lookahead(prev) else None
    outputs = self._consume(prev, nxt)      # 唯一的同步点
    self.pending = nxt
```

**四个 drain**（任一成立就回普通路径，本拍不提前提交）：

| 条件 | 计数 | 原因 |
|---|---|---|
| `waiting` 非空 | `waiting_drains` | 新请求要 prefill，批次结构会变 |
| `running` 行序与上一拍不一致 | `batch_drains` | 行序变了，`advance` 会写错行 |
| 有序列即将结束 | `terminal_drains` | 下一拍会超 `max_tokens` |
| KV 页不够 | `memory_drains` | 提前一拍要预留到 `num_cached + 1 + k` |

**两个 fence**（必须**先等**在途那拍跑完，再动 CPU 状态）：

| 情况 | 计数 | 原因 |
|---|---|---|
| 本拍产出 EOS / 到 `max_tokens` | `eos_fences` | `update` 会 `deallocate` 释放页，在途那拍还在写 |
| radix 跨页（可能 remap） | `prefix_fences` | 页号变了，在途那拍的 slot 已失效 |

### 2.4 双缓冲回读（`ModelRunner._stage_samples`）

提交 N+1 与消费 N 会同时存在，单块 pinned 缓冲会被覆写。改成两组缓冲 + 两个 event
轮换，`read_samples` 消费 FIFO 里**最老**的一份 `(slot, n_p, n_d, spec)`。

---

## 3. 三个阻塞性 bug（诚实记录）

第一版接线**在 GPU 上直接失败**，根因都是"没验证接口契约"：

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | `AssertionError: len(token_ids)=30 vs num_cached_tokens+1=1`（30 是 prompt 长度，说明 prefill 从未执行） | 回落路径**重复 `schedule()`**：控制器先排一次判断能否进流水线，`_step_sync()` 又排一次 → 刚被移到 running 的 prefill 序列被当成 decode 行（`num_cached=0`，只调度 1 个 token） | `_step_sync(sched=None)` 接受已排好的批 |
| 2 | `AssertionError: len=1 vs num_cached+1=6`（CPU 复现） | ahead 批里放的是 **`SeqSchedule`**，而 `ScheduledBatch` 装的是 **`Sequence`**；`to_wire()` 按 Sequence 取字段并断言 | `ScheduledBatch(decodes=list(seqs))` |
| 3 | `AssertionError: len=33 vs num_cached+1=32` | 上一拍 `update` 末尾把 `num_scheduled_tokens` 清 0，ahead 拍没经过 `schedule()` 重设 → `num_cached += 0` 但 token 照 append | `consume` 里 `update` 前重置为 1 |

**第 2 个最有教育意义**：控制器里有 `cur is not old` 的**身份比较**。如果 `seqs` 是
`SeqSchedule`，这个比较**永远为真** → 永远 `batch_drain` → ZOS 永不生效。但那样
"输出仍然正确"，测试会**假通过**。所以 GPU gate 里必须同时断言
`lookahead_submitted > 0`。

---

## 4. 四个非阻塞隐患（读代码发现，均已修）

| 隐患 | 后果 | 修法 |
|---|---|---|
| EOS fence 用 `s.last_token` | 那是**本拍输入** token（CPU 落后一拍），拿它比 EOS 会漏 fence，在途那拍写进马上释放的页 | 先 `read_samples`，用本拍**产出** token 判断 |
| 丢弃在途 ticket 不清 FIFO | 下一次 `read_samples` 弹出过期数据，把上一批 token 记到新序列头上 | 新增 `discard_pending()`，与 `sync_samples()` 成对调用 |
| KV 预留逐行判定 | 多行同时跨页时逐行各自能过、合计超预算 | 改按 `sum(needed_append) > available_slots` 判总量 |
| radix 内部 remap 漏刷页表 | `_sync_block_tables` 只看首/末/长度，内部重映射会被判成"没变" | `prefix_backend=="radix"` 时每拍都刷（保守） |

---

## 5. 实测（4090D 24G）

| 配置 | p50 步时 | 降幅 | decode tok/s | greedy 一致 | lookahead |
|---|---|---|---|---|---|
| 0.6B bf16 CG bs=1 n=128 | 2.06 → **1.97 ms** | **-4.7%** | 481.4 → **504.5**（+4.8%） | 100% | 382 |
| 8B fp8 CG bs=1 n=128 | 10.47 → **10.36 ms** | -1.0% | 95.4 → **96.5** | 100% | 382 |

- p99：0.6B 2.12→2.02 ms；8B 10.54→10.38 ms。
- `drains` 只有 terminal（warmup/rep 结束），`memory`/`waiting`/`batch` 全为 0
  —— 流水线确实提前提交了，不是"每拍被 drain"的假一致。

### 为什么只有 4.7%

我们的 decode 路径本来就瘦：**采样早就是异步 D2H**（`_stage_samples` 在 ZOS 之前
就有），模型前向全在 CUDA Graph 里，`build_decode` 只是填 5 个字段 + 一次 H2D。
所以"两拍之间的 host 工作"只有几十到一百微秒，能被藏起来的也就这么多。

**这个结论本身就是有用的**：ZOS 的收益上限 = `min(G,C)/(G+C)`，当 C 已经被别的
优化压到很小时，它不再是瓶颈。

---

## 6. 适用范围与边界

| 维度 | 支持 | 说明 |
|---|---|---|
| 采样 | **仅 greedy** | 温度采样要走 `Sampler` 的分支，拿不到固定地址的样本 |
| 阶段 | 仅稳定 decode | prefill、批次变化、行序变化一律 drain |
| `tp=pp=1` | ✅ | `UniprocExecutor` |
| **`dp>1`** | ✅ | 每个副本独立跑一套 ZOS，**零协调**（v3/v4 把它一起拒了，是过宽的判断） |
| `tp>1` | ❌ 暂不支持 | 状态天然复制、`j` 由集合操作保证一致，缺的只是"每个 rank 同一张票"的协调层 |
| **`pp>1`** | ❌ | token 必须**反向**从末段回传到 stage 0（ZOS 推迟的正是这条链），需要一条额外反向环 |
| 投机 | ❌ 暂不支持 | 见 §7 |

---

## 7. 未完成：投机进 ZOS（Z3）

设备侧裁决（§2.2）已经就绪，但投机仍进不了流水线，原因是一个更深的依赖：

> `verify` 走的是 **`build_prefill`**，而它的 `positions` / `slot_mapping` /
> `cu_seqlens` 全部由 **CPU 按 `num_cached_tokens` 构建**。ZOS 下 CPU 状态落后一拍，
> 用它建出来的 verify 输入**位置就是错的**。

所以要让投机进 ZOS，还需要：

1. verify 的 `positions` 由设备侧按 `1+j` 推进；
2. `slot_mapping` 用设备页表现算（复用 `advance_decode` 的同一套算术）；
3. `cu_seqlens` 固定（每行 m 相同，与 verify 图的分档前提一致）；
4. 对应 DeepInfer 的"**乐观预留 + 设备侧纠正**"：CPU 按最大 draft 预留，
   设备侧用实际接受数撤销。

当前 `_eligible()` 显式排除 `spec_tokens > 0`，等 Z3 完成再放开。

---

## 8. 三条可复用的教训

1. **"能 import + CPU 单测过"不等于"能跑"**。这三个 bug 全在"类型契约"与"函数副作用"
   上（`ScheduledBatch` 装什么、`schedule()` 有副作用），CPU 单测碰不到。接线类改动
   必须有一条 GPU 端到端 gate，而且 gate 要断言"功能真的生效了"
   （`lookahead_submitted > 0`），不能只断言"输出对不对"。
2. **并发状态机要先问"谁清了我"**。`num_scheduled_tokens` 被上一拍 `update` 清零，
   这类"上一拍清、下一拍用"的字段，是异步流水线里最容易漏的。
3. **默认开要有可观测性兜底**。ZOS 默认开（照 DeepInfer），但同时提供
   `--no-zos`、`enable_zos=False` 与完整的 counters——出问题能一行关掉、也能从计数
   看出是"没生效"还是"生效了但收益小"。

---

## 9. 投机 + ZOS 的实测结论：**正确，但当前无收益**

投机进 ZOS 走的是方案 B（草稿按 `m` 追加 + 设备侧有效位 + `ub` 上界选桶），实现分
四块：`advance_verify`（verify 元数据设备侧）、`accept_greedy_device`（整批裁决）、
`propose_batch_ids`（草稿设备侧入参）、`_verify_zos`（接线）。8B FP8 + dspark k=7：

| 配置 | decode tok/s | 相对 |
|---|---|---|
| spec（非 ZOS） | 332.6 | 1.000× |
| spec + ZOS，**eager verify** | 148.5 | 0.446× |
| spec + ZOS，**ub + verify 图** | **315.4** | **0.948×** |

- 正确性：`accept_len 6.000`、`accept_rate 71.43%`、`proposed/accepted/steps 469/335/67`、
  逐位置曲线、128 个 token，与非 ZOS **逐位一致**；
- 从 eager 版到图版提升 **2.12×**，全部来自"把 verify 图拿回来"。

### 结论：投机路径的 host 工作已经不足以撑起 ZOS

```
普通 decode：C 里有 build_decode(5 字段 + H2D) + 回读 + postprocess  → 藏掉 C 得到 4.7%
投机：      C 已被设备侧裁决 + 图压得很小，而 ZOS 还额外增加了
            「设备侧元数据构造 + ub 判断」的开销            → 净收益 -5%
```

所以 **`zos_spec` 默认关**，这条路径记为"已验证正确、当前实现下无收益"。要让它转正，
前提是先有东西**增大 C**（例如调度逻辑变重），而不是继续优化它的开销。

### 三个只影响性能、不影响正确性的遗留

| 项 | 说明 |
|---|---|
| `ub` 周期重锚 | 已实现（每 32 拍 drain 一次）；不做的话长生成会一路退回 eager |
| `--zos-spec` CLI | 已实现（`--no-zos` / `--zos-spec`） |
| 草稿 KV 改 paged（方案 A） | 能彻底去掉 host 形状依赖 + padding 浪费，但与 ZOS 解耦，属独立重构项 |
