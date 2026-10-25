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

#### Phase 1 落地记录（2026-10-25）

已完成的代码（`spec_tokens=0` 时全部走原路径，零行为变化）：

| 文件 | 内容 |
|---|---|
| `src/spec/{__init__,base,ngram,verify}.py` | `Drafter` 接口、`NgramDrafter`（后缀匹配，窗口 256）、`accept_greedy`（纯函数裁决） |
| `src/data/sequence.py` | `draft_token_ids` / `num_accepted` / `append_tokens` |
| `src/data/batch.py` | `SeqSchedule.draft_ids`：草稿随 wire 下发 |
| `src/data/{block_manager,kv}.py` | `needed_append/can_append_n/may_append_n`：一步写 1+k 个位置，且对"块表已多于逻辑长度"幂等 |
| `src/control/scheduler.py` | 草稿预算（context / token budget 双夹紧）、KV 不足时退化为普通 decode、`_update_spec_decode` 落账、`spec_stats` 观测 |
| `src/compute/model_runner.py` | `_run_verify`：复用 prefill 因果通路一次算 `[anchor]+drafts` 的 logits；`read_samples` 增 `d_lens` |
| `src/control/{executor,core}.py` | 采样结果三元组（含每序列 token 数），IPC 编码兼容（三个 `list[int]`） |
| `run.py` / `src/config.py` | `--spec-tokens` / `--spec-draft`；`spec_tokens>0` 强制 `max_num_seqs==1` |

三条关键约定（后续 Phase 不应打破）：

1. **草稿不进 `token_ids`**：被拒草稿不污染长度/块账本/`num_completion_tokens`，只把验证通过的 token append。
2. **先裁决再 publish**：`num_cached_tokens += 1+j` 后，用 `num_scheduled_tokens = 1+j` 临时驱动 `sync_prefix`，保证只有被有效 KV 覆盖的整块进前缀索引。
3. **`len(token_ids) == num_cached_tokens + 1` 必须始终成立**：单测逐个场景断言（含零接受、EOS 截断、max_tokens 截断）。

**已验证（CPU，24 项）**：草稿器边界（k/预算/最近匹配）、`accept_greedy` 的 off-by-one 全枚举、块账本幂等、落账不变量、`spec_tokens=0` 回归。
**未验证**：GPU 端到端逐位一致（需 `Qwen3-0.6B` 权重，本机未下载）。测试已写好：`tests/gpu/test_spec_ngram_e2e.py`，同时断言"草稿被提出且被接受"以防空跑。

**已知限制**：单序列；`_run_verify` 里 `argmax().tolist()` 是同步读回（Phase 5 再考虑搬到 GPU）；
**verify 步走 eager**（复用 prefill 通路，不进 decode CUDA Graph），所以开投机后每步的
launch 开销不再被图摊薄——Phase 5 需要按 verify 的节点数分档录图，否则小模型上可能
"接受了草稿但总吞吐反而下降"。树形验证尚未接入（Phase 4）。

### Phase 2 温度采样 + 拒绝采样

改动：`Sampler.logprobs()`（`log_softmax`）、verify 时禁用 `argmax` 捷径、标准拒绝采样（接受 `min(1, p_t/p_d)`、拒绝时从 `max(0, p_t−p_d)` 归一化后采样、全接受则采 bonus）。

**验收**：固定 seed 与 baseline 一致；10k 样本分布检验（每位置 top-1 频率差 <1%）。

#### Phase 2 落地记录（2026-10-25）

实际实现与最初设想有一处修正：需要的是**归一化后的概率**而不是 log-probs，
且必须是"采样器实际使用的分布"（含 top-k / top-p / min-p 截断后重新归一化）。
否则输出分布会与 baseline 不一致——top-k 时尤其明显。

| 文件 | 内容 |
|---|---|
| `src/compute/sampler.py` | 新增 `final_probs()`：与 `forward` 的截断规则逐条对齐（top-k 按阈值、top-p 按 `cum_before ≤ p`、min-p 按 `p ≥ min_p·max_p`），greedy 行返回 one-hot |
| `src/spec/verify.py` | `accept_rejection()`：逐位 `u < min(1, p/q)`；拒绝后从 `normalize(max(0, p−q))` 恢复；全接受取 bonus。`draft_probs=None` 表示确定性草稿（one-hot，`q=1`） |
| `src/compute/model_runner.py` | `_run_verify` 按 `all_greedy` 分流：greedy → `accept_greedy`；否则 `final_probs` → `accept_rejection` |

**一个必须记住的前提**：拒绝采样的正确性要求草稿 **`x_i ~ q_i`**。ngram/greedy
草稿是 one-hot，等价于 `q_i(x_i)=1`；Phase 3 接入 DSpark 时必须传**真实的**
草稿分布，不能把"模型最可能的值"当作确定草稿却不更新 q——那样输出分布会跑偏。
（写测试时我第一版把草稿固定成某个 token，用固定 q 去验，得到 [0.50,0.50] vs
期望 [0.25,0.75]，正是这个前提的体现。）

**已验证（CPU，14 项）**：
- `final_probs` 归一化 / 温度缩放 / top-k=1 与 top-k=2 / top-p / min-p / all_greedy；
  并用 2 万次 `forward` 采样频次反证它就是"实际采样分布"；
- `accept_rejection` 的分布等价性（2 万次试验、容差 2%≈5.6σ）：one-hot 草稿、
  多步 k=2 的首位边际、以及**非 one-hot 草稿（x 从 q 采样）**三种情形；
- 长度不变量 `len(tokens) == accepted + 1`。

另加 `tests/unit/test_spec_scheduler_loop.py`（5 项）：用"假 target + 确定性草稿"
驱动**真实**的 `Scheduler` / `KVManager` 跑完整生成——全接受时输出必须逐 token
复现标准答案；周期性拒绝只校验不变量；KV 紧张与接近 context 上限时不得越界。
它覆盖多步串接与满页边界，这类错误原本只能靠 GPU e2e 暴露。

> 写这个测试时踩到两个"测试自身的错"：① `deallocate` 会把 `block_table` 清空
> 且 `num_cached_tokens` 归零，所以序列结束后不能再断言常规不变量；② 块表只需
> 覆盖「已写 KV 的 0..cached-1」，断言写 `ceil((cached+1)/B)` 会在 cached 恰好
> 落在块边界时误报——真正写满那一块的是**上一步**，pending token 要到下一步才写。

**未验证**：GPU 温度采样 smoke（缺模型文件）。GPU 侧复用同一批函数，没有
GPU 专属分支；`tests/gpu/test_spec_ngram_e2e.py` 已加 smoke 用例（断言能跑通、
token 数正确、spec 路径被走到）。

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
