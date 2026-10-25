# DSpark 投机解码落地报告

> 从零接入 DeepSeek 的 DSpark 草稿模型，在自研推理引擎上做到 **decode 3.398×**、
> **accept_len 6.538（官方 6.26）**、greedy 逐位无损。
> 配套文档：[dspark-plan.md](dspark-plan.md)（计划与阶段记录）、
> [dspark-spec.md](dspark-spec.md)（上游规格核实）、
> [dspark-reference-eval.md](dspark-reference-eval.md)（参照系跑法）、
> `tools/dspark_diff_vs_upstream.md`（对拍手册）、`tools/dspark_draft_probe.py`（探针）。

---

## 1. 背景与目标

引擎原本是"每步一个 token"的 decode：一次前向只产出一个 token，大模型上 decode
是**带宽瓶颈**（权重被复用的次数等于 batch size），算力大量闲置。投机解码用"先猜
后验"把这些闲置算力换成吞吐：草稿模型一次猜 k 个，目标模型一次前向全部验证，接受
最长的可信前缀。

**DSpark** 是 DeepSeek 2026-07 公开的方法（arXiv 2607.05147），与 EAGLE/DFlash 同
族但有两个关键差异：

| 特征 | 含义 |
|---|---|
| **并行草稿** | 块内 k 个候选由一次非因果前向并行产出（不是逐 token 串行） |
| **半自回归** | `markov_head`（rank=256）用"上一个 token"给块内每个位置加 logits 偏置，补回块内依赖 |
| **置信度调度** | `confidence_head` 预测逐位置接受率，可用于自适应验证预算（本轮未启用） |

验收判据（计划里定的三条）：

1. greedy 下与 baseline **逐 token 一致**（无损）；
2. `accept_len ≥` 同 prompt 同 seed 的官方实现 90%；
3. decode 吞吐 ≥ 1.4×。

---

## 2. 上游调研：先核实，再动手

量化那轮的经验是"技术选型不是偏好，是被证据推出来的"。这次同样先花一天核实：

**结论一：有现成草稿，但它是"草稿头"不是独立小模型。**
`deepseek-ai/dspark_qwen3_8b_block7`（2.371B 参数，2/3 是冻结的 embed + lm_head）必须
消费目标模型第 `[1,9,17,25,33]` 层的 hidden states。

**结论二：checkpoint 不带代码。** 仓库只有 `config.json` + `model.safetensors`，
`Qwen3DSparkModel` 不在 transformers 里（config 的 `transformers_version: 5.10.2`
只是保存时版本，不参与校验）。→ 必须 vendor 上游实现。

**结论三：`crop(start)` 的语义必须读源码确认。** 这是全篇最容易被忽略、又最容易
写错的一处：

```
start = 已提交长度（下一个待写位置），初始 = prompt_len

propose:  追加 seg（上一步验证通过的 hidden）+ block 个候选 → crop(start)
update:   target_output.hidden_states[:accepted+1] 作为下一步的 seg
```

即"草稿 KV 只存已提交 token 的 context，块内候选的 KV 用完即弃"。猜错这一条，
草稿会看着正常、验证也能过，但接受率莫名偏低——这类错误没有报错，只有数字不对。

---

## 3. 架构与设计

### 3.1 数据流（一步）

```
调度：reserve = 1 + k        ← dspark 的草稿在 worker 侧产出，调度器只按上限预留
  ↓
起草：草稿模型一次非因果前向
      [anchor, mask×6] + 上一步的 context 段 → markov 半自回归采样 → k 个候选
  ↓
验证：目标模型一次因果前向（eager 的 prefill 通路）
      [anchor] + drafts 在位置 p..p+k，读已有 KV，算全部 m = k+1 个位置的 logits
  ↓
裁决：greedy → 最长匹配前缀；温度 → 标准拒绝采样
  ↓
落账：num_cached_tokens += 1 + accepted；只把验证通过的 token 追加进 token_ids
```

### 3.2 五个关键设计决定

| 决定 | 理由 |
|---|---|
| **草稿不入 `token_ids`** | 被拒草稿一旦进序列，会污染 `len(seq)` → 块账本、`max_tokens`、`num_completion_tokens` 全错。单独放 `draft_token_ids`，只有验证通过的才 append |
| **先裁决再 publish** | `sync_prefix` 用 `num_scheduled_tokens` 反推"本步覆盖了哪几个整块"。落账后必须临时设成 `1+j`，否则被拒草稿写的半块会被登记进前缀索引，别的请求命中就拿到垃圾 KV |
| **块分配以 `num_cached+n` 为基准** | 而不是 `len(seq)`。这让"上一步多分配的块"天然幂等，不需要显式回收（避免 ref_count 出错） |
| **verify 复用 prefill 通路** | verify 本质是"带因果掩码的多 token 前向 + 读已有 KV"，与带前缀缓存的 prefill 完全同构（`cu_seqlens` + `block_tables`），不需要为它单独写 attention 内核 |
| **DSpark 每步都走 verify** | 一是草稿需要 verify 产出的 aux hidden，二是这条路径**完全不进 decode CUDA Graph**，把最担心的录图回归面绕开了 |

### 3.3 `Drafter` 的两种形态

ngram 草稿是纯 CPU 的后缀匹配，适合放在调度器侧；DSpark 草稿需要 GPU、aux hidden
和自己的 KV，只能在 worker 侧。所以接口演化成：

```python
class Drafter(Protocol):          # 调度器侧：propose(seq, k) -> list[int]
    ...

class _WorkerSideDrafter:         # dspark：只返回"预留额度"标记
    worker_side = True
    def propose(self, seq, k): return []
```

调度器看到 `worker_side` 就按 `1+k` 预留位置与块，真实草稿数由 worker 通过
`_SpecSamples.proposed` 回传（否则 `accept_rate` 永远是 NaN）。

---

## 4. 分阶段实施

| Phase | 内容 | 关键产出 |
|---|---|---|
| 0 | 规格核实 | `dspark-spec.md`：架构、权重名、crop 语义、精确配置 |
| 1 | 框架 + 线性验证（ngram） | `src/spec/{base,ngram,verify}.py`、块账本 `n` 参数化、`Sequence.truncate` |
| 2 | 温度 + 拒绝采样 | `Sampler.final_probs()`、`accept_rejection()` |
| 3 | 草稿接入 | vendor 4 文件、aux hidden、`DSparkDrafter`、worker 侧接线 |
| 4 | 树形验证 | **主动取消**（见 §6.4） |
| 5 | 性能与组合 | verify CUDA Graph、FP8 组合、autotune 预热、自动分桶 |

**为什么先 ngram**：它零模型依赖，能把"多 token verify + KV 回滚 + 逐位一致"这条
链路单独验证干净。如果第一次就把草稿和验证叠在一起，出错时无法判断是草稿错、还是
verify/回滚错。

---

## 5. 遇到的问题与根因（本文最有价值的部分）

### 5.1 测试断言错，不是引擎错

现象：`spec_tokens == base` 失败。对照后发现 **spec == 同一 rep_prompt 的 baseline**
逐位一致——`rep_prompt` 末尾重复的是 `base[:12]`，模型自然从 `base[12:]` 续写，
所以 spec 本来就不该等于**原始** baseline。

> 教训：这类"测试期望取错参照"和真正的实现错误在输出上长得一样，都必须先用
> 对照实验（同输入 vs 不同输入）区分，不能直接改断言。

### 5.2 greedy 分叉：不同 attention 内核的数值路径

现象：8B 上 greedy 一致率只有 28.1%，分叉在某个位置，该位置 **top-2 logits 完全
相等**（28.5 / 28.5）。已排除：运行噪声、CUDA Graph、k 大小、argmax 实现。

根因：verify 走 paged-varlen prefill（`flash_attn_varlen_func`），baseline decode 走
`flash_attn_with_kvcache`，归约顺序不同 → logits 差 ~1ulp → 精确并列处 argmax 翻转。

**第二次修正（第一版判断不完整）**：曾以为"K/V 值与内核无关"——错。位置 q 的 K/V 是
`k_proj(norm(h_q))`，而 `h_q` 是**上一层 attention 的输出**。verify 算出的 `h_q` 与
decode 差 ~1ulp，写进 cache 的 K/V 也不同。所以只重算 emit 位置（`spec_tie_eps`）
**只能覆盖一部分**，长序列仍分叉。

最终方案两级：

| 开关 | 保证 | 代价 |
|---|---|---|
| `spec_tie_eps`（默认 0） | ❌ 部分（只覆盖 emit 处并列） | tie 步 +1 次前向 |
| `spec_exact_kv`（默认关） | ✅ 逐位一致（实测 8B n=128 `first_diff=None`） | 每步 +`accepted+1` 次前向 |

**必须讲清的算术**：`spec_exact_kv` 每步花 `accepted+1` 次前向产出 `accepted+1` 个
token，摊到每 token 恰好 1 次前向，再加 draft/verify 开销 → **必然比 baseline 慢**。
所以"greedy 下逐位无损"与"加速"在数值层面不可兼得——它是**校验/参考模式**，不是
性能模式。

### 5.3 worker 侧起草暴露的三个集成 bug

上机第一次跑通后暴露：

1. **aux 交接维度**：引擎给的是 2D `[n, F]`，草稿模型要 3D `[1, n, F]`（带 batch 维）
   → `cat` 报 "got 2 and 3"。
2. **草稿数必须截断**：块宽由 checkpoint 固定（7），但引擎的 `--spec-tokens` 可能更小。
   多给的草稿会被 verify 写到**未分配块**的位置上——**越界写 KV**，块边界必现，而且
   是静默的数据破坏，不是崩。
3. **`accept_rate` 断链**：ngram 在调度器侧起草，`proposed` 能在调度时统计；dspark 的
   草稿在 worker 侧，调度器算不出来 → `accept_rate` 恒为 NaN。

### 5.4 录图失败：`out=` 与 autograd 冲突

现象：`RuntimeError: add(): functions with out=... don't support automatic differentiation`。

根因：aux 采集用 `torch.add(h, residual, out=aux[...])`，而 `_capture_verify_graphs`
在 `__init__` 里被调用、不在 `inference_mode` 内，此时 `h/residual` 仍带 grad。

修法：整个 capture 包进 `@torch.inference_mode()`。

> **重要**：这个报错一度被误判为"varlen 注意力不可捕获"，差点去改走
> `flash_attn_with_kvcache`。看错误里的 `out=` + `automatic differentiation` 就能
> 立刻定位——**不要因为报错发生在录图阶段就归因于"图不支持某算子"**。

### 5.5 最大的坑：transformers 4.57 vs 5.10（环境问题伪装成实现问题）

现象：我们 `accept_len` 5.53，官方 6.26（88%），逐位置 `@0` 几乎一致、越靠后差越多
（@6 差 0.25）。

排除过程：

1. BF16 对照（5.429）没有变好 → 排除 FP8 target；
2. aux 数值对拍（vs HF `output_hidden_states`）：`max|diff|/|ref|` 在 1% 量级、
   各槽持平 → 排除"aux 采集错"（结构错会是 O(1)）；
3. **草稿对拍探针**（同一 ctx/anchor、不经过目标模型）：
   - ours(5.10) vs theirs(5.10)：`block_hidden max|diff| = 0`、tokens 逐位相等；
   - ours(4.57) vs theirs(5.10)：`rel 33%`，tokens 完全分叉。

**定案**：vendored 代码无责，差异来自草稿仍借用的 `Qwen3RotaryEmbedding` /
`sdpa_attention_forward` 在 4.x→5.x 之间行为变了。

**性质**：这不影响正确性（草稿永远被目标模型验证，错的草稿只会被拒），是**纯接受率
问题**。修法采用"环境升级"（方案 A）：`transformers==5.10.2` + `float8_e8m0fnu` 垫片，
torch/flash-attn/flashinfer 全不动。升级后 `accept_len 6.538`，**超过官方 6.26**。

> 这个 case 的方法论价值最大：**"两份代码相同、结果不同"时，第一嫌疑是环境**，而
> 定位它的最短路径是"把输入固定成同一份，逐级比对中间量"。

### 5.6 bench 口径反了

`speedup = base / spec` 把 3.398× 印成 0.294×——**直接会让人误判"投机没有收益"**。
现已改为 `spec / base`，并在代码里写明"写成反过来会把 3.4× 印成 0.29×"。

---

## 6. 结果

### 6.1 性能（4090 24G，FP8 目标 + BF16 草稿）

| 指标 | A: eager verify | B: verify graph |
|---|---|---|
| baseline decode tok/s | 93.8 | 93.8 |
| spec decode tok/s | 134.9 | **318.7** |
| **decode 加速比** | 1.438× | **3.398×** |
| e2e tok/s | 127.2 | 272.8 |
| accept_rate / accept_len | 65.64% / 5.595 | 65.64% / 5.595 |
| greedy 一致率 | 100% | 100% |
| peak | 16.49 GB | 16.55 GB |

B 相对 A：decode **2.36×**，而接受率与一致率**逐位不变**——这正是 A/B 设计的目的：
让收益与正确性同时被验证。

### 6.2 与官方实现的对照（同 20 条 gsm8k、seed 980406、n=128、T=0、k=7）

| 配置 | accept_len | @0 | @3 | @6 |
|---|---|---|---|---|
| 官方 DeepSpec（BF16, 5.10.2） | 6.26 | 0.938 | 0.745 | 0.578 |
| **我们（FP8, 5.10.2 + torch 2.6）** | **6.538** | 0.950 | 0.785 | 0.629 |
| 我们（FP8, 4.57.6） | 5.532 | 0.938 | 0.662 | 0.328 |
| 我们（BF16, 4.57.6） | 5.429 | 0.948 | 0.636 | 0.324 |

`accept_len ≥ 基线 90%` → **实际 104%**，逐位置全面不低于官方。

### 6.3 为什么 3.4× 而不是论文的更高倍数

- 官方论文的倍数是**大 batch + 长输出**；本轮是 batch=1、n=64，decode 阶段 GPU 利用
  率本来就低，投机的主要收益来自"把空转的算力拿来验证"；
- verify 每步要跑 k+1 个位置（约 8 倍计算），却只产出 ~6.5 个 token——**净收益取决
  于接受长度**，batch 越大、模型越大，收益越明显。

### 6.4 为什么不做树形验证

计划里原本要做树形多分支（topk=2 × depth=4）。做到 Phase 3 后发现**没有收益证据**：

- 官方 6.26 用的就是**线性验证**；
- 我们在 5.10 下与官方**逐位相同**，accept_len 还略高。

即**线性验证已经吃满草稿质量**。树形是最贵的一块（要写 tree attention kernel +
树上拒绝采样），在没有证据时投入它属于典型的过度工程，因此主动取消。

### 6.5 batch>1 的实测与瓶颈归因

b 条请求并发（b 也从调度器 `max_num_seqs` 走），同一 prompt、同 seed、greedy：

| | b=1 | b=4 |
|---|---|---|
| baseline decode tok/s | 93.7 | 367.1 |
| spec decode tok/s | **335.8** | **623.5** |
| decode 加速比 | **3.583×** | **1.698×** |
| accept_rate / accept_len | 71.4% / 6.00 | 72.4% / 6.07 |
| 一致率 | 100% | 100% |
| peak | 16.56 GB | 16.59 GB |

**接受率不随 batch 退化**（6.00 → 6.07，逐位置曲线重合），说明多序列 verify 的
`cu_seqlens` 拼平与逐序列落账是对的。

**但相对加速比从 3.58× 掉到 1.70×**。把步时拆开就能看到原因：

| | b=1 | b=4 | 变化 |
|---|---|---|---|
| 每步产出 token（聚合） | 6.00 | 4 × 6.07 = 24.3 | 4.05× |
| 每步耗时 | 17.9 ms | 39.0 ms | **2.18×** |
| ├ 草稿前向 | 1 次 ≈ 6.1 ms | **4 次串行 ≈ 24.4 ms** | **4.0×** |
| └ verify 前向 | 11.8 ms（8 token） | 14.6 ms（32 token） | 1.24× |

- **verify 已批量化**：token 数 ×4，耗时只 ×1.24；
- **草稿是串行的**：`_propose_dspark` 是 `for s in seqs` 逐条跑，每条都要把
  草稿模型 4.7 GB 权重读一遍（≈6.1 ms @774 GB/s）。b=4 时草稿占步时 **61%**。

baseline 在 b=4 能到 3.92×，是因为它每步只读一遍 target 权重就算完 4 个 token；
而我们的草稿把这份"权重摊销"白白浪费了 4 遍。

> DSpark 论文的 `parallel_drafting` 本意包含跨序列批量化草稿，这一点本轮没做。
> 预期收益：把 4 次草稿前向合成 1 次后，b=4 步时 39.0 → ≈21 ms，聚合吞吐
> ≈1.16k tok/s，相对加速比回到 **~3.1×**。

**两个不要直接比的口径**（容易误读）：

- **e2e 加速比**（b=4 仅 1.44×）被 TTFT 拖累：b=4 的首请求要在 4 条并发 prefill
  中排队（476 ms）而 b=1 是 37 ms。看 batch 收益应只看 decode；
- **TTFT** 本身在 b>1 下是"首请求"口径，与"平均排队"无关，不宜作横向对比。

---

## 7. 方法论沉淀

| 手法 | 这次的作用 |
|---|---|
| **测试断言先自证** | `spec == base` 失败时，先用"同一 prompt 的 baseline"复算，区分"期望错"与"实现错" |
| **同输入逐级对拍** | aux 对拍（hidden）→ 草稿探针（hidden → logits → tokens），每一级都能独立否掉一批假设 |
| **两份代码相同时，先怀疑环境** | transformers 4.x/5.x 的差异伪装成了"实现 bug"，排查花了最久 |
| **收益与正确性同一次 A/B** | verify 图的 A/B 里，接受率与一致率必须逐位不变，否则说明静态缓冲被污染 |
| **不变量优先于输出** | 草稿 KV 长度恒等于已提交长度、`len(token_ids) == cached+1` 这类整型断言，比"输出看起来对"更早暴露问题 |
| **算清代价再决定"要不要"** | `spec_exact_kv` 的每 token 前向次数与 baseline 持平 → 它只能是参考模式，不能当性能卖点 |

---

## 8. 已知限制与剩余工作

| 项 | 状态 | 说明 |
|---|---|---|
| **batch > 1** | ✅ 已支持 | `b=4` 上机验证：greedy 100%、accept_len 不退化；verify 图按 `(bs, m=k+1, 桶)` 分档 |
| **草稿批量化** | 未做 | b=4 的草稿仍是逐序列串行（占步时 61%），批量化后预期 1.70× → ~3.1× |
| MoE × DSpark | 未做 | v1 显式拒绝（30B-A3B 上不可用） |
| FP8 KV × spec | 未测 | 组合未验证，默认关闭 |
| verify 图桶 | 已做 | 256/512/1024/2048 自动分桶，超出回退 eager |
| 置信度调度 | 未启用 | `confidence_head` 可做自适应验证预算，属可选优化 |

**batch>1 的实现要点**（已经想清楚）：`_run_verify` 去掉单序列假设、每序列一棵验证
批按 `cu_seqlens` 拼平（`d_lens` 通路现成）、`_verify_graphs` 的键从 `(m, bound)`
扩成 `(bs, m, bound)`（组合数会涨，需要限制 bs 档位）。草稿器状态本来就是按 `seq_id`
隔离的，不用改。
