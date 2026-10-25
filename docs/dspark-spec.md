# DSpark 上游规格核实（Phase 0 产出）

> 来源：`github.com/deepseek-ai/DeepSpec`（MIT），读取日 **2026-10-25**（main 分支）。
> 论文：arXiv 2607.05147《DSpark: Confidence-Scheduled Speculative Decoding with
> Semi-Autoregressive Generation》。
> 本文只记录**从源码读到的事实**，推断性内容标注「推断」。

---

## 1. 发布的草稿 checkpoint

README 的 Table 1 给出三算法 × 四目标模型的矩阵，DSpark 一行：

| 目标模型 | 草稿 repo | 训练配置 |
|---|---|---|
| `Qwen/Qwen3-4B` | `deepseek-ai/dspark_qwen3_4b_block7` | `config/dspark/dspark_qwen3_4b.py` |
| **`Qwen/Qwen3-8B`** | **`deepseek-ai/dspark_qwen3_8b_block7`** | `config/dspark/dspark_qwen3_8b.py` |
| `Qwen/Qwen3-14B` | `deepseek-ai/dspark_qwen3_14b_block7` | `config/dspark/dspark_qwen3_14b.py` |
| `google/gemma-4-12B-it` | `deepseek-ai/dspark_gemma4_12b_block7` | `config/dspark/dspark_gemma4_12b.py` |

同一批还发布了 **Eagle3**（`eagle3_qwen3_8b_ttt7`）与 **DFlash**（`dflash_qwen3_8b_block7`）
作为对照，可用于消融与 baseline。

---

## 2. Qwen3-8B 的精确配置

`config/dspark/dspark_qwen3_8b.py`：

```python
block_size = 7                      # 每块 7 个候选（含 anchor 位）
num_draft_layers = 5                # 草稿 5 层
target_layer_ids = [1, 9, 17, 25, 33]   # 取目标模型这 5 层的 hidden
mask_token_id = 151669              # 块内占位 token（Qwen3 词表内）
num_anchors = 512                   # 训练时每样本采样的 anchor 数（推理无关）
markov_rank = 256                   # 马尔可夫头低秩维度
markov_head_type = 'vanilla'
confidence_head_alpha = 1.0         # >0 即启用置信度头
confidence_head_with_markov = True  # 置信度头输入 = hidden ⊕ markov embedding
loss_decay_gamma = 4.0              # 块内位置衰减（训练）
ce_loss_alpha = 0.1; l1_loss_alpha = 0.9
```

对应 Qwen3-8B 的 `hidden_size=4096`、`num_hidden_layers=36`、`vocab_size=151936`。
所以草稿的 `fc` 输入维是 `5 × 4096 = 20480`。

> 注：`target_layer_ids` 是**0 基层号**，取 hidden 时用 `hidden_states[layer_id + 1]`
> （`hidden_states[0]` 是 embedding 输出），见 `common.py::extract_context_feature`。

---

## 3. 草稿模型结构（`deepspec/modeling/dspark/qwen3/modeling.py`）

```python
class Qwen3DSparkModel(Qwen3PreTrainedModel):
    embed_tokens = nn.Embedding(vocab_size, hidden_size)        # 从目标模型拷贝后冻结
    layers       = [Qwen3DSparkDecoderLayer] * num_draft_layers # = 5，full_attention
    norm         = Qwen3RMSNorm(hidden_size)
    rotary_emb   = Qwen3RotaryEmbedding(config)
    fc           = nn.Linear(len(target_layer_ids)*hidden, hidden, bias=False)   # 20480→4096
    hidden_norm  = Qwen3RMSNorm(hidden_size)
    lm_head      = nn.Linear(hidden_size, vocab_size, bias=False)  # 从目标模型拷贝后冻结
    markov_head  = build_markov_head(config)      # rank=256, vanilla
    confidence_head = AcceptRatePredictor(hidden + markov_rank)   # → 1 维 logit
```

要点：

- **词表与目标一致**：`build_draft_config` 里 `tie_word_embeddings=False`，但 `vocab_size`
  沿用目标模型；`initialize_embeddings_and_head` 把 target 的 embed/lm_head 权重拷进来后
  冻结。→ **不存在 draft_vocab → target_vocab 的映射问题**（与 vLLM 里 `draft_vocab_size`
  的兼容分支无关，后者面向其他形态）。
- **草稿注意力是双流的**（`Qwen3DSparkAttention.forward`）：
  ```python
  k = cat([k_proj(target_hidden_states), k_proj(hidden_states)])   # 上下文流 + 块流
  v = cat([v_proj(target_hidden_states), v_proj(hidden_states)])
  ```
  即每层都对「目标 hidden 特征」算一份 K/V，再拼上块内 K/V。`target_hidden_states` 在
  `_forward_backbone` 里先过 `fc + hidden_norm`。
- **块内非因果**：`self.is_causal = False`，训练时由 `flex_attention` 的 mask 保证
  「块内互看、跨块不可见」；推理时 `attention_mask=None, is_causal=False`，因为跨块隔离
  由「块 KV 用完即丢」+ cache 结构保证。
- **markov head**：块内候选不是独立采样的，`sample_draft_tokens` 走
  `markov_head.sample_block_tokens(base_logits, first_prev_token_ids, hidden_states, T)`，
  用 rank=256 的低秩转移注入前后依赖（半自回归）。
- **confidence head**：`predict_confidence_step(hidden, prev_token_ids)` → 每位置一个 logit，
  推理时取 `sigmoid < threshold` 的首个位置截断 proposal。

---

## 4. 推理协议（`deepspec/eval/dspark/draft_ops.py` + `evaluator.py`）

### 4.1 初始化（prefill 后）

```python
context.target_hidden_states = extract_context_feature(
    initial_output.hidden_states, draft_model.target_layer_ids)   # [1, prompt_len, 20480]
context.past_key_values_draft = DynamicCache()
```

### 4.2 每一步 propose

```python
draft_input_ids = full((1, 7), mask_token_id)     # 7 个占位
draft_input_ids[:, 0] = output_ids[:, start]      # 第 0 位放 anchor（最后接受的 token）

block_hidden = model._forward_backbone(
    target_hidden_states=context.target_hidden_states,   # 上一步验证出的新 context 片段
    noise_embedding=embed_tokens(draft_input_ids),
    position_ids=position_ids[:, pkv.len : start+7],     # 绝对位置 start..start+6
    attention_mask=None,                                 # 非因果
    past_key_values=past_key_values_draft, use_cache=True, is_causal=False)

past_key_values_draft.crop(start)                  # ← 丢掉块内 KV，只保留 context
```

产物：`block_hidden [1, 7, 4096]` → `base_logits = lm_head(...)` →
`sample_draft_tokens(..., temperature)` → `sampled_tokens [1, 7]` →
confidence 截断得 `n = proposal_draft_tokens` →
`verify_input_ids = [anchor, d1..dn]`、`draft_probs = softmax(draft_logits[:n] / T)`。

### 4.3 验证与更新

```python
verification = 目标模型对 [anchor, d1..dn] 做一次因果前向（n+1 个 token）
   → 逐位接受/拒绝（标准拒绝采样，见 base_evaluator）
context.target_hidden_states = extract_context_feature(verification.hidden_states)[:, :accepted+1]
```

### 4.4 与 vLLM 的对照

vLLM 的 `SpeculativeConfig` 对 `method="dspark"` 做了四件事，与上面一致：

```python
uses_aux_hidden_states = True              # 要目标模型的中层 hidden
parallel_drafting = True                   # 块内并行草稿
num_speculative_tokens = block_size        # 默认取 checkpoint 的 block_size
enable_adaptive_verification               # confidence 调度（仅 dspark）
```

---

## 5. 与本引擎的落差（改造清单输入）

| 需要 | 我们的现状 | 缺口 |
|---|---|---|
| 目标模型输出 5 层 aux hidden | `Qwen3ForCausalLM.forward` 只返回最终 hidden | 增加 `output_aux_hidden_states` + 静态输出缓冲（CG）、PP 汇总 |
| 草稿模型 | 无 | vendor `modeling.py` + `markov_head.py`（MIT，约 30 KB）或 `pip install deepspec` |
| 草稿 KV cache | 无（引擎 KV 池是给 target 的） | 草稿用独立的 `DynamicCache`/自建小池；每步 `crop` |
| 一步生成多个 token | `num_scheduled_tokens = 1` 硬编码 4 处 | Phase 1 的改动点 |
| 多 token 验证 | decode 路径假设 `M == bs` | 复用 prefill/因果路径（线性）；树形需自写 mask kernel |
| 采样概率 | `Sampler` 只返回 id | 新增 `logprobs()` 支路 |
| token 回滚 | 只能 append | `Sequence.truncate_to(n)` + 先裁决后 publish |

---

## 6. 上机核实结果（2026-10-25）

### 6.1 已确认

1. **checkpoint 不自带 remote code**：仓库只有 `config.json` + `model.safetensors`（4.5 GB），
   没有 `modeling_*.py`。→ **已 vendor**（2026-10-25），落在 `src/spec/dspark/`：

   | 上游 | 本地 |
   |---|---|
   | `deepspec/modeling/dspark/qwen3/modeling.py` | `src/spec/dspark/modeling.py` |
   | `deepspec/modeling/dspark/qwen3/config.py` | `src/spec/dspark/config.py` |
   | `deepspec/modeling/dspark/common.py` | `src/spec/dspark/common.py` |
   | `deepspec/modeling/dspark/markov_head.py` | `src/spec/dspark/markov_head.py` |
   | `deepspec/utils/sampling.py`（仅 `sample_tokens`）+ `metrics.py`（`add_metric`） | `src/spec/dspark/_compat.py` |

   均为 MIT，每个文件带 provenance 头。**本地只改三处**：① `deepspec.*` → 相对导入；
   ② `create_block_mask` 改函数内惰性导入（训练/教师强制才用，推理不碰，避免因一个
   训练依赖让推理 import 失败）；③ 加头注释。

   **CPU 冒烟已过**（`tests/unit/test_dspark_vendor.py`，7 项）：vendored 模块可独立
   import（`deepspec.*` 依赖已切断）、8 个符号齐全、极小配置下双流注意力前向形状正确、
   `sample_draft_tokens` 的 markov 偏置确实生效、`predict_confidence_step` 输出逐位置
   logit、`extract_context_feature` 的 0 基/`-1` 语义正确。
2. **实际 config 与训练配置一致**：

   ```
   architectures: ["Qwen3DSparkModel"]     block_size: 7
   num_hidden_layers: 5                    num_target_layers: 36
   target_layer_ids: [1, 9, 17, 25, 33]    mask_token_id: 151669
   markov_rank: 256                        markov_head_type: vanilla
   confidence_head_with_markov: true       vocab_size: 151936
   layer_types: 5 × full_attention         max_window_layers: 36
   ```

   与 `config/dspark/dspark_qwen3_8b.py` 的训练设置逐项对齐（`num_anchors=512` 是训练期
   采样参数，推理不涉及，故 config 里没有）。

### 6.2 依赖结论修正

config 里的 `transformers_version: 5.10.2` **不是硬约束**——`Qwen3DSparkModel` 不在
transformers 里，而是 DeepSpec 仓库的实现；该字段只是"保存时用的版本"。真正的依赖是
那份 `modeling.py` 引用的 8 个符号：

```
Qwen3MLP / Qwen3RMSNorm / Qwen3RotaryEmbedding / eager_attention_forward /
rotate_half / ALL_ATTENTION_FUNCTIONS / GradientCheckpointingLayer / FlashAttentionKwargs
```

本机（transformers 5.19.0）实测**全部存在**；`transformers.models.qwen3` 自 4.51 起加入，
所以租机上的 4.57.6 大概率也齐（Phase 3 第一步就是跑一遍符号检查，缺哪个补哪个，不必
为一个 config 字段去升 torch）。

**已上机实测（2026-10-25，租机 transformers 4.57.6）**：`Qwen3DSparkModel.from_pretrained`
加载官方 checkpoint 一次通过——

```
missing   : []
unexpected: []
params B  : 2.371
block/target_layers/markov: 7 [1, 9, 17, 25, 33] 256
```

即**权重名与 vendored 类完全对齐**（不需要 `_keys_to_ignore` 或重命名），参数量落在
预期区间（embed + lm_head 各 ~0.62B 冻结 + 5 层草稿），且 `loader.py` 的
`dtype=` / `torch_dtype=` 双参数名兼容在 4.57 上生效（`dtype=` 是 4.56+ 才有的写法）。

### 6.3 `crop(start)` 语义（已核实，2026-10-25）

读 `deepspec/eval/base_evaluator.py::generate_decoding_sample`（26 KB）后的结论：

**`start` = 已提交长度**（`output_ids` 中已确定的 token 数），即"下一个待写位置"，
初始值为 `prompt_len`。不是"草稿 KV 的已有长度"。

每步流程（源码逐行）：

```python
verify_input_ids = [anchor, d1..dk]                  # 长度 k+1
target_output = target_model(input_ids=verify_input_ids,
                             position_ids=position_ids[:, start:start+k+1],
                             past_key_values=pkv_target, use_cache=True)
# → 写位置 start..start+k，cache 长度变为 start+k+1
output_ids[:, start : start+j+1] = verify_input_ids[:, :j+1]
output_ids[:, start+j+1] = next_token                # 恢复/bonus
start += j + 1
pkv_target.crop(start)                               # 丢弃被拒草稿的 KV
```

所以 `crop` 的参数是**更新后的** `start`（`+j+1` 之后），语义 = 保留 anchor 与
accepted 草稿的 KV、丢弃被拒部分。**草稿侧另有一次 crop，参数是更新前的 `start`**：

```python
# forward_dspark_draft_block
draft_position_ids = position_ids[:, pkv_draft.get_seq_length() : start + block]
block_hidden = model._forward_backbone(..., past_key_values=pkv_draft, is_causal=False)
pkv_draft.crop(start)        # ← 保留 0..start-1 的 context，丢弃块内 block 个位置
```

即草稿 KV 只保存**已提交 token 的 context 特征**，块内那 k 个候选的 KV 用完即弃。
`_update` 再把 `target_output.hidden_states[:accepted+1]` 作为下一步的 context 输入。

**与本引擎的映射（语义等价，只是"物理删除" vs "逻辑截断"）**：

| DeepSpec | 本引擎 |
|---|---|
| `start`（下一个待写位置） | `seq.num_cached_tokens` |
| `pkv_target.crop(start_new)` | `num_cached_tokens += 1+j` + **不 publish** 被拒块 |
| `pkv_draft.crop(start)` | 草稿 KV 只保留已提交 token 的 context（Phase 3） |
| `verify_input_ids` | `[last_token] + draft_ids` |
| `committed_tokens` | `tokens = accepted drafts + next_token` |

差别只在"被拒草稿的 KV 是物理删除还是留在池里"：我们由 `cache_seqlens =
num_cached_tokens` 保证**永不读到**越界位置，下次写同一 slot 时被覆盖，因此等价。

⚠️ Phase 3 实现草稿 KV 时注意：草稿 context 长度是 **`start`（已提交 token 数）**，
不是 `num_cached_tokens + 1`——多算一个会把 anchor 重复计入（上游已把 anchor 作为
块的第 0 个位置喂进去）。

### 6.4 环境依赖：transformers 5.x 对接受率是硬要求（2026-10-25 定位）

用 `tools/dspark_draft_probe.py`（同一 ctx/anchor，不带目标模型）对拍：

| 组合 | block_hidden | base_logits argmax | tokens |
|---|---|---|---|
| ours(5.10) vs theirs(5.10) | **0** | 完全一致 | **完全一致** |
| ours(4.57) vs theirs(5.10) | rel **33%** | 从第 2 位起不同 | 完全分叉 |

结论：**vendored 代码与上游功能一致**（同 venv 下逐位相等），差异来自 4.57 与 5.10
的 Qwen3 组件行为——草稿仍借用 transformers 的 `Qwen3RotaryEmbedding` 与
`sdpa_attention_forward`。

影响面：**不影响正确性**（草稿永远被目标模型验证，错的草稿只会被拒），但接受率
明显偏低（`accept_len` 5.53/5.43 vs 官方 6.26，即 88%/87%）。

两条出路：

1. **要求 transformers ≥ 5.0**（最简单）；租机因 `torch 2.6.0a0` 被判版本过低而
   无法升到 5.x，需要换 torch 或打补丁；
2. **自研 RoPE + 直接调 `torch.nn.functional.scaled_dot_product_attention`**，
   彻底不依赖 transformers 的内部组件——顺带把 vendor 时的 8 个符号依赖砍到
   3 个（`Qwen3MLP`/`Qwen3RMSNorm`/`Qwen3PreTrainedModel`），是把"版本脆弱性"
   一次性解决的做法。

`loader.py` 已加启动告警：transformers < 5 时打印上面这段结论，避免静默降级。

### 6.5 尚未做

**vLLM 同机基线**（acceptance length / decode tok/s）——租机时补一次，作为 Phase 3
验收的对照口径；目前只有本引擎 baseline（Qwen3-8B greedy 59.9 tok/s）。
