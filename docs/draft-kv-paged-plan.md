# 草稿 KV 页式化（PagedDraftKV）

> 把 DSpark 草稿的 KV cache 从「每行预留满 `max_len`」改成「页池 + 页表」，
> 让容量从 `O(行数 × 上限)` 降到 `O(实际 token)`，并为 offload / 多卡预留页 id 空间。
>
> **状态**：P0~P2 已完成并上机前自检（`285 passed`）；P3（FA paged 快路径）代码就绪、
> **默认关且未在 GPU 上验证**；P4 GPU 验收脚本已就绪。

---

## 1. 为什么改

稠密布局的容量与用量无关：

| 场景 | 稠密预留 | 真实需要 |
|---|---|---|
| b=1，n=128（8 行 × 4096） | 671 MB | ~2.5 MB |
| b=4，n=128（16 行 × 4096） | 1.34 GB | ~17 MB |
| 1M 上下文 | 不可行（20 GB/序列起） | 20 GB（仍需多卡） |
| 1024M 上下文 | 20 TB | 20 TB（需分层/offload） |

单 token KV：草稿 `2 × 8 头 × 128 × 2B × 5 层 = 20 KiB`；目标（36 层）`144 KiB`。

> **对照对象必须说清**（GPU 验收时踩到的口径问题）：
> - 上面的 671 MB / 1.34 GB 是 **`PaddedKVCache`（`spec_draft_batch=True` / ZOS 路径）**；
> - **单序列**（`max_num_seqs=1`，未开 batch 草稿）的 dense 走 `DynamicCache`，
>   128 token 只有约 **1.6 MB**，比页池（默认 128 页 × 16 = 2048 token ≈42 MB）还小；
> - 所以"页式更省"只在**长上下文 / 多序列 / batch 草稿**下成立，
>   单序列短上下文用页式反而更占显存（池是上界，不是实际用量）。

## 2. 两条读路径（收益来自存储，不来自读路径）

| 档 | 做法 | 代价 | 状态 |
|---|---|---|---|
| **一（默认）** | `read_dense_layer()` 按页表 gather 成 `[B, heads, kv_len, dim]`，再走原 SDPA | 每步每层一次拷贝（128 token ≈ 12 MB） | ✅ 已实现、已 CPU 验证；**vendored attention 零改动** |
| 二（未来） | `flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens, block_table)` | 改 `modeling.py`（约 25 行）；**页大小必须 256 的倍数** | ⚠️ 代码就绪、默认关、**未在 GPU 验证** |

**关于「去掉掩码」**：只有路径二能真正去掉。路径一为了让 SDPA 拿到矩形张量仍需
`mask()`；但 pad 占用的页在 `crop` 时退还，所以浪费是**瞬时且不累积**的（稠密是永久预留）。

### 页大小约束（已核实）

- 本仓 `src/config.py` 对目标模型有硬校验：`kvcache_block_size % 256 != 0` → `raise`
  （注释写明「FA2 paged 要求 256 的倍数」）；
- `flash_attn_with_kvcache` 的参数名是 **`block_table=`**（不是 `page_table`）；
- 所以：**路径一可用 16；路径二必须 256**。`fa_args()` 里对此 fail loud。

## 3. 页号编码（为扩展留空间，现在零成本）

```
page_id (int64)
  bit  0..35  页内编号（单机）
  bit 36..43  tier   0=GPU 1=CPU 2=SSD
  bit 44..51  device 0..255（TP / CP）
  负数        未分配
```

页表类型因此不必随扩展改动——将来 offload / 跨卡只是「填不同的值」。
`encode_page()` / `decode_page()` 提供编解码，`_release_page()` 只把 tier==GPU 的页
放回空闲链（换出的页不能直接复用）。

## 4. 三条不变量（写错都不报错，只让接受率无声下滑）

1. **页表覆盖的页数必须够放下逻辑长度** —— 不够就读到空页；
2. **`crop` 必须退还尾部整页** —— 只做「逻辑缩短」会让页数按 `m` 增长，1.33 倍浪费照旧；
3. **未参与本步的行绝不能被推进或改写** —— 行池长期持有，每步参与者不同。

三条各有一条单测（`tests/unit/test_dspark_paged_cache.py`，共 15 项）。

### 实现中撞到并修正的一处

`kv_len` 是**批内共享标量**（`max(lens) + seg_max + block`），所以分配必须按
`seg_max` 而非 `seg`，否则短行读不到共享 `kv_len` 覆盖的位置。pad 页在 `crop` 时退还，
所以这个"按上界分配"不会累积成浪费。

## 5. 容量与降级

- 池容量 = `num_pages`（`spec_draft_paged_pages`，0 表示 `max(64, max_rows × 16)`）；
- 页不足时**不抛错**：`begin_step` 把该行移出本步（`active_rows` 变短），
  调用方退化为「本步不起草」——只损收益，不影响正确性；
- `propose_batch` 一旦发现 `active_rows` 少行就**整批退回逐序列**，避免行号错位串数据。

显存估算（页大小 16，5 层，8 头，128 维，bf16）：
`num_pages × 327 KB`。默认 `max_rows×16 = 128` 页 ≈ **42 MB**（对比稠密 671 MB~1.34 GB）。

## 6. 验收口径（必须改一条）

| 比较 | 口径 | 原因 |
|---|---|---|
| 页式 vs 稠密（**都是 SDPA**） | **逐位相同** | 数学等价、K/V 同源；已用 CPU 单测 + `tests/gpu/test_dspark_paged_gpu.py` 覆盖 |
| 路径二 vs 路径一（FA vs SDPA） | **`accept_len` 差 ≤0.1**、逐位置曲线形状一致 | 归约顺序不同，逐位要求不成立 |
| 页式内部：批量 vs 单跑 | **逐位相同** | 验的是页表/偏移，与数值内核无关 |

## 7. 影响范围

| 文件 | 改动 |
|---|---|
| `src/spec/dspark/paged_cache.py` | **新增** ~430 行：页池、页表、编码、crop 退页、gather 读、FA 参数 |
| `src/spec/dspark/drafter.py` | `_ensure_paged/_ensure_cache`、`_propose_paged`、`propose_batch` 走 `_ensure_cache`、失效兜底 |
| `src/spec/dspark/modeling.py` | **仅路径二**：paged 分支约 25 行（这是"上游零改动"优势的代价） |
| `src/config.py` | `spec_draft_paged` / `spec_draft_page_size` / `spec_draft_paged_pages` / `spec_draft_paged_fa` |
| `src/compute/model_runner.py` | 构造实参透传四个开关 |
| `bench/spec_bench.py` | `--spec-draft-paged` 等三个 flag |
| `tests/unit/test_dspark_paged_cache.py` | **新增** 15 项不变量 |
| `tests/unit/test_dspark_drafter_batch.py` | **新增** `TestPagedEquivalence`（3 项，页式 vs 稠密逐位） |
| `tests/gpu/test_dspark_paged_gpu.py` | **新增** GPU 验收（token 一致、接受数一致、显存下降、页池耗尽兜底） |

## 8. 已知风险与后续

| 项 | 说明 |
|---|---|
| **路径二未验证** | `modeling.py` 的 FA 分支只在本地通过语法/导入检查；需 GPU 跑 `--spec-draft-paged-fa --spec-draft-page-size 256` 并用探针对拍 |
| **vendored 分叉** | 路径二启用后草稿 attention 与上游 DeepSpec 不再逐行一致，上游升级需重新对齐 |
| **短上下文可能更慢** | 路径一每层多一次 gather（128 token ≈ 12 MB），长上下文才划算；需实测 |
| **页池大小是静态的** | 目前不能在运行中扩容；长上下文场景需要按 `context_len` 调大 `spec_draft_paged_pages` |
| **与 ZOS 投机的交互** | 页式明确拒绝 `device_segs`（ZOS+投机需路径二）；`zos_spec` 与 `spec_draft_paged` 同时开会在 first propose 时报错，属 fail loud |

## 9. 下一步（按价值排序）

1. **GPU 验收路径一**：`pytest tests/gpu/test_dspark_paged_gpu.py`（token 逐位 + 显存）；
2. 若显存收益确认，把 `max_ctx_len` 从 4096 改为跟随 `context_len`（纯参数，不改结构）；
3. 路径二在长上下文机型上验证后，才考虑把 `spec_draft_paged` 设为默认；
4. 将来接 offload / CP 时，只需扩 `tier`/`device` 两段的填值逻辑与一个"页缺失换入"的钩子。

---

## 10. GPU 验收记录（2026-11-03，4090 24G，5.10.2，8B fp8 + dspark k=7）

### 通过项

| 用例 | 结果 |
|---|---|
| 路径一（gather → SDPA）vs 稠密 | **48/48 token 逐位相同**、accepted 32=32、steps/proposed 一致 |
| 页池耗尽（pages=4，容量 64 token） | 允许中途不起草，输出与稠密前 32 位逐位一致 |
| 路径二（FA paged）vs 路径一（n=128） | 两者都是 30 steps / 210 proposed / 98 accepted / **accept_len 4.2667**，差 0.0；128 token 逐位相同 |
| FA kernel vs SDPA（隔离） | 单行 37 token、双行跨页（600/300，页号非连续）`max|diff| ≈ 0.002–0.004` |

### 验收中暴露并已修的 7 处

| # | 问题 | 症状 | 修法 |
|---|---|---|---|
| 1 | FA 分支自己 reshape + return，**漏 `o_proj`** | `accept_len 1.114 (13/798)`；输出 token 仍正确（verify 不看草稿质量）→ **静默** | 抽 `_project_out()` 作唯一出口，两条路径共用 |
| 2 | `begin_step` 短行按自身 `lens+seg_max` 预页，gather 用批内 `max(lens)` | 多行不等长时 `_slots` 断言「页表不足」 | 先算批内 `kv_len`，所有行按它预页 |
| 3 | `fa_args` 交整张 `[num_rows, ...]` 页表 | 首次运行 `RuntimeError: block_table must have shape (batch_size, max_num_blocks_per_seq)` | `block_table(subset=True)` 只取参与行 |
| 4 | 页表宽度按"人均份额 ×4" | 单行占满页池时 `IndexError`（而非优雅降级） | 宽度取 `num_pages`（最坏情况），代价 KB 级 |
| 5 | 池空时本轮已分配的页未回滚 | 长跑逐轮泄漏 | 失败分支 `_release_row_pages(keep=before)` |
| 6 | GPU 测试 `sum(stats["proposed"])` | 对 int 调 `sum` → `TypeError` | 改 `stats["accepted"] <= stats["proposed"]` |
| 7 | "峰值显存不升"断言口径错 | 单序列 dense 走 `DynamicCache`(1.6 MB) vs 页池 168 MB，靠 +0.2 GB 容差蒙过 | 该用例只做宽松回归；新增 **batch 模式** dense(`PaddedKVCache`) vs paged 的对照 |

第 1、2、5 条都是**不报错但算错/泄漏**的类型，只能靠"同一输入两条路径逐位比对"和"页池计数不变量"发现——这也是为什么每条都补了 CPU 可复现的回归测试。

### 未覆盖（诚实标注）

- 引擎级 `n=128` 总长 158 < 页大小 256，**只覆盖单页**；跨页/多行只到 kernel 级；
- **多行引擎路径（`max_num_seqs>1` + 页式）未跑**；
- `test_batch_mode_paged_uses_less_memory`（batch 模式下 dense `PaddedKVCache` vs
  paged 的显存对照）**尚未实测**。注意该用例必须并发 ≥2 条请求（单序列的 dense 仍走
  `DynamicCache`，对照对象不成立），且每次 `_run` 都要 `reset_peak_memory_stats()`；
  该用例只含 `torch.cuda` 分配，不含 `torch.cuda.empty_cache()`，分进程跑更稳；
- 路径二的 `spec_draft_paged_fa` 仍需在长上下文机型上验证才可考虑设默认。
