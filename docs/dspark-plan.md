# DSpark 投机解码实施计划

> 上游事实（架构、权重名、推理协议）见 [dspark-spec.md](dspark-spec.md)；本文只记**决策、范围、排期与验收**。
> 上游来源：`github.com/deepseek-ai/DeepSpec`（MIT）、论文 arXiv 2607.05147、checkpoint
> `deepseek-ai/dspark_qwen3_8b_block7`。检索日：**2026-10-25**。

---

## 1. 已确认的三项决策

| # | 决策 | 说明 |
|---|---|---|
| 1 | **草稿用 HF 加载** | 复用 DeepSpec 的 `Qwen3DSparkModel`（约 20 KB 单文件，MIT），以 `from_pretrained` 加载权重；不自研草稿结构。自研留作 Phase 6 可选 |
| 2 | **树形验证从 `topk=2, depth=4` 起** | 先线性打通正确性（Phase 1），再上树（Phase 4）；树的最多节点数 2⁴=16，便于手算与对拍 |
| 3 | **v1 只支持 `max_num_seqs=1`** | 与上游 `build_dspark_proposal` 的 `assert batch_size == 1` 一致；多序列（每序列一棵树）留到 v2 |

---

## 2. 范围

**做**：dense 模型（Qwen3-8B）+ DSpark 草稿 + 树形验证 + 标准拒绝采样（greedy 与温度）+ 与量化的组合验证。

**不做（v1）**：MoE、batch>1、PP、draft 与 target 异构词表（`use_heterogeneous_vocab`）、suffix/ngram（仅作为 Phase 1 的测试替身，不进产品路径）。

---

## 3. 依赖的上游事实（影响设计的四条）

1. 草稿消费**目标模型 5 个中间层的 hidden states**（`target_layer_ids=[1,9,17,25,33]`）→ 目标模型 forward 必须能返回 aux hidden。
2. 草稿块内**非因果**（`is_causal=False`），块与块之间互不可见 → 一块 7 个候选一次前向产出（`parallel_drafting`）。
3. 草稿有**自己的 KV cache**，但每步结束后 `crop(start)` 把块内 KV 丢掉，只留"已接受 token 的 context"。
4. 验证输入是 `[anchor, d1..dn]`（上游为**线性**）→ 树形是本项目对上游的扩展。

---

## 4. 分阶段计划与验收

### Phase 0 规范核查 ✅（本次完成）

产出：[dspark-spec.md](dspark-spec.md)（架构/权重名/协议/配置）。

验收：能回答"草稿要什么输入、输出什么、KV 怎么滚"。当前状态：**已满足**；剩余一项待上机确认——
`deepseek-ai/dspark_qwen3_8b_block7` 是否随仓库提供 `modeling_*.py`（决定 `trust_remote_code` 还是 vendor 源文件）。

### Phase 1 框架 + 线性验证（ngram 草稿，greedy）

目的：把 `Sequence / KV 账本 / 调度 / verify 通路` 的改动与 DSpark 算法解耦验证。

改动点（已定位）：

| 位置 | 现状 | 改法 |
|---|---|---|
| `src/control/scheduler.py:77` | `seq.num_scheduled_tokens = 1` | 改 `k+1`；token budget 不足时**退化普通 decode** |
| `src/data/batch.py:45` | `assert len(token_ids) == num_cached_tokens + 1` | 放宽为 `>=`；`_decode_schedule` 发 `token_ids[num_cached_tokens:]` |
| `src/control/scheduler.py:184-185` | 先 `+=` 再 `sync_prefix` | **先裁决 j → `truncate_to(j)` → 再 `sync_prefix`**（避免把被拒 draft 的 KV publish 进 prefix cache） |
| `KVManager.may_append` | 只按 1 token 追加块 | 批量 `may_append(seq, n)` |
| `Sequence` | 无回滚 | 新增 `truncate_to(n)` |

新增 `src/spec/`：`base.py`（`Drafter` 抽象）、`ngram.py`、`verify.py`（线性验证 + greedy 接受 + 回滚）。

**验收**：greedy 与 baseline **逐 token 完全一致**；ngram 在重复文本上 decode ≥1.2×；`quantization=none` 且不开 spec 时性能零变化。

### Phase 2 温度采样 + 拒绝采样

改动：`Sampler.logprobs()`（`log_softmax`）、verify 时禁用 `argmax` 捷径、标准拒绝采样（接受 `min(1, p_t/p_d)`、拒绝时从 `max(0, p_t−p_d)` 归一化后采样、全接受则采 bonus）。

**验收**：固定 seed 与 baseline 一致；10k 样本分布检验（每位置 top-1 频率差 <1%）。

### Phase 3 DSpark 草稿接入（4090）

改动：目标模型输出 aux hidden（`target_layer_ids` 的 5 层，`hidden_states[layer_id+1]` 口径）；vendor/加载 `Qwen3DSparkModel`；实现 `DSparkDrafter.propose()`（7 个 mask token + anchor → 块前向 → markov → confidence 截断）。

**验收**：acceptance length ≥ vLLM/DeepSpec 基线的 90%；Qwen3-8B/4090/batch=1/256-prompt decode ≥1.4×；greedy 逐位一致。

### Phase 4 树形验证

改动：候选树构建（topk=2×depth=4）、tree attention kernel（节点只 attend 祖先 + prefix）、树上拒绝采样、KV 临时槽与回滚。

**验收**：同 workload 下 acceptance length 高于线性；greedy 逐位一致；温度下分布检验通过；每接受 token 的成本低于线性。

### Phase 5 性能与组合

CUDA Graph 分档（节点数 8/16/32/64）、draft 侧 CG、量化组合矩阵（bf16/FP8/INT8 × dense）、`bench/spec_bench.py`、文档。

**验收**：CG 下 verify 的 launch 开销被摊掉（对比 eager 有可测收益）；INT8 的 `M=16` 双路径边界有用例覆盖。

---

## 5. 风险清单

| 风险 | 影响 | 缓解 |
|---|---|---|
| 目标模型要返回 5 层 aux hidden | 改 forward 返回值 + CG 输出缓冲 + PP 汇总 | Phase 3 先 eager 打通，CG 后补 |
| 树形 attention 无现成实现 | 需自写 Triton mask kernel | 先线性（Phase 1）再树（Phase 4）；节点数 ≤64 可控 |
| draft_vocab 与 target_vocab 不等时的概率对齐 | 拒绝采样会算错 | Phase 2 就实现该映射并用单测验；Phase 3 核对 checkpoint 的 `vocab_size` |
| `sync_prefix` 时序 | 被拒 draft 污染 prefix cache | Phase 1 的"先裁决再 publish"用例 |
| verify 与 CG 的变长 M | decode CG 收益被稀释 | Phase 5 按节点数分档录图 |
| 与量化叠加 | INT8 的 M 分派、FP8 KV 的 scale 共用 | 量化文档 §8 已列出，Phase 5 覆盖 |

---

## 6. 里程碑

| 里程碑 | 判据 |
|---|---|
| M1 骨架可用 | Phase 1 验收通过（greedy 逐位一致 + ngram 收益） |
| M2 采样正确 | Phase 2 验收通过（分布等价） |
| M3 DSpark 生效 | Phase 3 验收通过（≥1.4×，acceptance ≥基线 90%） |
| M4 树形超越线性 | Phase 4 验收通过 |
| M5 可交付 | Phase 5 验收通过 + 文档/bench 齐全 |
