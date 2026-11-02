# 投机解码接入 ZOS（方案 B）设计

> 状态：**设计已定，实现未开始**。前置已完成：`advance_decode`（普通 decode 的设备侧推进）、
> `advance_verify`（verify 元数据）、`accept_greedy_device`（设备侧裁决）、
> `propose_batch_ids`（草稿设备侧入参）、`_SpecSamples.dev_*`（样本异步回传）。
> 参考实现：DeepInfer `csrc/engine/zos.cpp`。

---

## 1. 为什么投机不能直接套用普通 decode 的 ZOS

普通 decode 每拍推进量恒为 1，所以 `lens`、位置、slot 都能由 host 侧的确定值推出。
投机每拍推进 `1 + j`，而 **j 只在设备上**（接受裁决的产物）。于是任何"用 host int 做
切片/偏移"的地方都会变成阻塞同步点：

```python
kv_len = int(self.lens[rows].max()) + m   # .item() → 等上一拍算完 → lookahead 失效
kk[rows][:, :, :kv_len]                   # SDPA 输入必须是 host 形状
```

**这是结构性约束，不是实现瑕疵**：目标模型没问题（flash-attn varlen 的 `cu_seqlens`
是设备张量，kernel 内部读），但草稿侧用的是 **dense padded + SDPA**，形状必须先知道。

---

## 2. 方案 B 的核心：让所有"需要 host 知道"的量都变成**上界**

### 2.1 草稿缓存：按 `m` 追加，而不是按 `1+j`

```
每拍追加 m = 1+k 个位置（host 已知，不需要 j）
  ├─ 写入偏移 = lens[r]（host 维护，每拍 += m）
  └─ 有效位   = arange(m) < seg_r      ← seg_r = 1+j_r 是**设备标量**
```

`valid[r, s : s+m] = arange(m) < seg_r` 这种"**host 切片 + 设备掩码**"是合法的：
切片下标是 host int，掩码是 device bool 张量，不产生同步。

于是 `crop` 消失了——不需要"截到 accepted+1"，因为无效位置在 attention 里被 mask 掉。

### 2.2 代价：缓冲要留余量（可算）

每拍浪费 `k - j` 个位置，所以缓冲区容量相对真实上下文要放大：

```
真实占用 ≈ 生成 token 数 × (m / accept_len) = T × (1+k)/(1+avg_j)
k=7、accept_len≈6 时 → 约 1.33 × T
```

取 `max_ctx_len = int(context_len × 1.5) + 64` 即可覆盖。**不需要压缩，也不需要在
drain 时重建**——这是 B 相对"真设备索引"省下来的最大一块复杂度。

### 2.3 verify 的桶选择：用 host 上界

CPU 不知道精确 start，但知道**上界**：

```
ub_{N+1} = ub_N + m        // m = 1+k，纯整型，无同步
初始 ub = num_cached_tokens（首拍是同步提交，值准确）
```

桶选择只需要"装得下"：取最小的 `bound` 满足 `ub + m <= bound`。
**这样 verify 的 CUDA Graph 得以保留**（否则为 ZOS 丢掉 2.36× 的图收益，净收益大概率为负）。

`ub` 会随轮次缓慢漂移（比真实 start 大，最多累计 `轮数 × j`）。每 **N 拍强制 drain 一次**
（自然同步点用 `read_samples` 的真实长度重新对齐），漂移就被截断在 `N × m` 以内。

---

## 3. 数据流（一拍）

```
提交 N+1（此时 j_N 未知）
├─ 草稿：propose_batch_ids(anchors=j 无关的最后一 token, starts=ub 上界)
│         ctx = 上一拍的整块 aux[m]，seg = 1+j_{N-1}（设备标量）
├─ ids   = cat([anchors, drafts], dim=1)              ← 全在设备上
├─ advance_verify(batch, starts_device, m)            ← positions/slots/cu_seqlens_k
├─ 桶选择：host ub → bound（装不下就 drain 回同步）
└─ forward（graph replay）→ accept_greedy_device(整批) → (next_tok, advances=1+j, toks)

消费 N（唯一同步点）
├─ read_samples → d_ids / d_lens（经 pinned 异步回传）
├─ 落账：num_cached_tokens += 1+j，ub = ub + m
└─ 每 N 拍：强制 drain，用真实长度重锚 ub
```

---

## 4. 实现清单

| # | 内容 | 可测性 |
|---|---|---|
| 1 | `PaddedKVCache` 支持"按 m 追加 + 设备 seg 掩码"（`crop` 走空实现） | CPU：host list seg vs device seg 的掩码逐位等价 |
| 2 | `_dspark.update_full(seq_id, aux_block)`：存整块 `[m, F]`，不切片 | CPU |
| 3 | `_run_verify` 的 ZOS 分支：跳过 `build_prefill`，设备 ids + `advance_verify` + 整批裁决 | GPU e2e |
| 4 | `ub` 维护 + 桶选择 + 每 N 拍 drain | CPU：`ub >= 真实 start` 不变量 |
| 5 | `_eligible` 放开 spec，加 `zos_spec` 开关（默认开，可单独关） | — |

第 1 项是唯一的数值风险点（掩码写错 → accept_len 掉但不崩），所以它的 CPU 等价测试
必须先绿，再动第 3 项。

---

## 5. 与"方案 A（草稿 KV 改 paged）"的关系

A 能彻底消除"host 形状"依赖，顺带去掉 padding 浪费、支持更长上下文。但它不是"换个
varlen 就行"：

- FA 的 `cu_seqlens` 要求**连续 packed 布局**，而我们的缓存是 per-row 固定槽位
  （行 r 从 `r*cap` 开始，`cu[r]` 推出来是 `sum(len)`，两者只有在 `len == cap` 时相等）；
- 要么每行单独一次 varlen（需要 `len_r` → 又一次 host 同步），要么真 packed（crop 时
  要左移数据 → 形状变化 → 还是需要 host 长度）；
- DeepInfer 用的是 **paged KV + kernel 内按页寻址**，长度全程走 device 指针，
  并有 `forbid_cuda_sync()` 守卫整条 launch 路径。

所以 A 是"把草稿 KV 从 dense padded 升级到 paged"的独立重构项，建议单独立项，
不与 ZOS 绑在一起做。

---

## 6. 已知边界

- 只支持 `tp=pp=1`（`dp>1` 天然可用，每副本独立）；
- `pp>1` 需要 token/j 的**反向环**（末段 → 各 stage），且我们的 PP 是 lockstep，
  收益上限只是"把 host 工作藏进流水气泡"；
- 周期 drain 的 N 需要按实测调（N 太小 → 同步多、收益少；N 太大 → 桶浪费 + `ub` 漂移）。
