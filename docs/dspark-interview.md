# DSpark 投机解码 · 面试提纲

> 配套工程报告：[dspark-report.md](dspark-report.md)。
> 用法：先讲 §1（30 秒），对方有兴趣再展开 §2（3 分钟），§3/§4 是被追问时的弹药库。

---

## 1. 三十秒电梯版

> "我在自研推理引擎里接了 DeepSeek 的 DSpark 投机解码。难点不是写解码循环，而是
> **三件必须同时成立的事**：草稿模型要吃到目标模型的中间层 hidden、验证要和 baseline
> 数值一致、还要把 verify 也塞进 CUDA Graph 才有收益。
>
> 最终 decode **3.4×**（93.8 → 318.7 tok/s），接受长度 **6.54**，同 prompt 同 seed 下
> 比官方实现 6.26 还略高，greedy 逐位无损，峰值显存 16.5G。"

---

## 2. 三分钟主线（前因后果）

### 2.1 为什么做：decode 是带宽瓶颈

引擎原本每步一个 token。decode 阶段权重只被复用 `batch` 次，算力大量闲置、带宽吃满。
投机解码用"先猜后验"把这些闲置算力换吞吐：草稿一次猜 k 个，目标一次前向全验，接受
最长可信前缀。

**关键判断**：DSpark 不是"独立小模型"，它是**草稿头**——必须消费目标模型第
`[1,9,17,25,33]` 层的 hidden states。这一条决定了整个改造范围（要动目标模型的
forward 返回值），是调研阶段最重要的产出。

### 2.2 怎么做：一次前向的四段

```
起草（草稿模型，非因果）  [anchor, mask×6] + 上一步的 context 段
                          → markov 半自回归采样 → 7 个候选
验证（目标模型，因果）     [anchor] + drafts 一次前向，算 8 个位置的 logits
裁决                     greedy：最长匹配前缀；温度：标准拒绝采样
落账                     cached += 1 + accepted，只追加验证通过的 token
```

三个设计决定值得讲：

- **草稿不进 `token_ids`**：被拒草稿一旦进序列，会污染 `len(seq)` → 块账本、
  `max_tokens` 全错。单独放 `draft_token_ids`。
- **先裁决再 publish**：`sync_prefix` 靠 `num_scheduled_tokens` 反推"覆盖了哪几个
  整块"。落账后要临时设成 `1+j`，否则被拒草稿写的半块会进前缀索引，**别的请求命中
  就拿到垃圾 KV**。
- **verify 复用 prefill 通路**：verify 就是"带因果掩码的多 token 前向 + 读已有 KV"，
  与带前缀缓存的 prefill 同构（`cu_seqlens` + `block_tables`），不需要新 attention 内核。

### 2.3 卡在哪：三个真问题

1. **greedy 分叉**：verify 走 varlen prefill、baseline 走 `with_kvcache`，两者在
   **精确并列**处 argmax 会翻。更要命的是我一开始判断错了——以为 K/V 与内核无关，
   实际上 K/V 的输入 hidden 就是 attention 算出来的，所以 verify 写进 cache 的 KV
   也带 1ulp 差，长序列仍会分叉。
2. **环境伪装成 bug**：accept_len 只有官方的 88%，查到最后是 transformers 4.57 与
   5.10 的 Qwen3 组件行为不同（块前向差 33%）。
3. **收益上不去**：eager verify 只有 1.44×，因为每步几十次 kernel launch 没被摊掉。

---

## 3. 最能讲的三个硬问题

### 3.1 「为什么 verify 和 decode 会算出不同的 logits」

**答**：内核不同。verify 是 paged-varlen prefill（`flash_attn_varlen_func`），decode 是
`flash_attn_with_kvcache`，attention 的归约顺序不一样，logits 差 ~1ulp。平时无所谓，
但目标分布**精确并列**时（实测某位置 top1 = top2 = 28.5）argmax 就翻。

更深一层：只修"emit 位置"不够。位置 q 的 K/V 是 `k_proj(norm(h_q))`，而 `h_q` 是
**上一层 attention 的输出**——verify 算出的 `h_q` 与 decode 差 1ulp，写进 cache 的
K/V 也就不同。后续普通 decode 读到这份 KV，依然会在并列处翻转。

所以做了两级：`spec_tie_eps`（只重算 emit 位置，便宜但不保证）和 `spec_exact_kv`
（每步把 anchor + accepted 逐个用 decode 内核重写 KV，保证逐位一致）。

**必须主动说清的算术**：`spec_exact_kv` 每步花 `accepted+1` 次前向产出 `accepted+1`
个 token，摊到每 token 恰好 1 次前向，加上 draft/verify 开销 → **必然比 baseline 慢**。
所以它是**校验模式**，不是性能模式。greedy 下"逐位无损"与"加速"在数值层面不可兼得。

### 3.2 「你怎么确定不是自己写错了」

这条我讲**排查链路**，因为它比结论更有说服力：

1. **先怀疑测试**：`spec == base` 失败时，我没有直接改断言，而是用"同一个 rep_prompt
   的 baseline"复算——发现 spec 与它逐位一致，是参照取错了。
2. **排除量化**：BF16 对照（5.43 vs FP8 5.53）→ 不是量化。
3. **aux 对拍**：与 HF `output_hidden_states` 比，相对误差 ~1%、各槽持平 → 不是采集错
   （结构错会是 O(1) 量级）。
4. **草稿探针**：把 ctx/anchor 固定成同一份、不经过目标模型，比 hidden → logits →
   tokens 三级。**同 venv 下逐位相等，换回 4.57 立刻 33% 分叉** → 一句话定案。

> 一句话总结这套方法：**"两份代码相同、结果不同"时，第一嫌疑是环境；定位它的最短
> 路径是把输入固定成同一份，逐级比对中间量。**

### 3.3 「越界写 KV 那个 bug 怎么来的」

草稿的块宽由 checkpoint 固定（7），但引擎的 `--spec-tokens` 可以是 4。调度器按
`1+4` 预留块，worker 却给了 7 个草稿 → verify 写到**没分配块**的位置上。

它不崩、不报错，只是**静默破坏相邻序列的 KV**，而且只在块边界出现。修法是在 worker
侧把草稿截到 `k`——前向仍按 block_size 跑（模型结构决定），采样后截断。

**教训**：调度器与 worker 之间"预留量"和"实际量"必须显式对齐，不能靠"通常一样"。

---

## 4. 亮眼优化点

**① verify 也进 CUDA Graph —— 1.44× → 3.40×**

eager verify 每步几十次 kernel launch，全落在关键路径上。按 `m = 2..k+1` 各录一张图
后 decode 提升 **2.36×**，而接受率与一致率逐位不变（这正是 A/B 的设计目的：让收益与
正确性同时被验证）。

**② 两个"进图"的硬约束**

- `max_seqlen_k` 是 kernel 的**标量参数**，进图后不可变 → 用上界桶
  （256/512/1024/2048），运行期选**最紧**的桶，浪费的只是空 tile（早退），不影响正确性；
- 所有 verify 图**共用一个 graph pool**，否则 `桶数 × (k+1)` 张图各自持有私有池会
  线性吃显存。

**③ autotune 必须挡在图外，也要挡在请求外**

FP8 per-block GEMM 带 `@triton.autotune(key=["M","N","K"])`，而 verify 的
`M = 1+k` 是 decode/prefill 预热都没见过的形状 → 首次调用现场 `do_bench`（清 L2 +
同步）＝秒级尖刺。现在启动期对 `m=2..k+1` 各跑一次假 verify 预热。

**④ 用"不变量"而不是"输出对不对"来测**

- 草稿 KV 长度恒等于已提交长度；
- `len(token_ids) == num_cached_tokens + 1`；
- 块表只覆盖"已写 KV 的 0..cached-1"（写成 `ceil((cached+1)/B)` 会在满页时误报）。

这些整型断言在 GPU 上跑不了的时候也能测（纯 CPU），比"输出看着对"更早暴露问题。

---

## 5. 结果（面试报这些）

| 项 | 值 |
|---|---|
| decode 加速比 | **3.398×**（93.8 → 318.7 tok/s，FP8 target + verify 图） |
| 其中 verify 图贡献 | 1.438× → 3.398×，即 **2.36×** |
| accept_rate / accept_len | 79.1% / **6.538**（k=7，上限 8） |
| 官方同 prompt 同 seed 对照 | 6.26 → 我们 **104%**，逐位置全面不低于 |
| greedy 一致率 | **100%**（`spec_exact_kv` 提供逐位严格口径） |
| 峰值显存 | 16.5 GB（24G 卡有余量） |
| 单测 | 234 项 |

---

## 6. 主动暴露的不足（加分项）

> "有三个地方我想说明：
>
> 一是**只支持单序列**（`max_num_seqs=1`）——投机模式下还不能服务并发请求，这是我
> 下一个要做的，实现路径已经清楚（每序列一棵验证批 + 拼平 cu_seqlens + 图分档加 bs 维）。
>
> 二是**树形验证我主动砍了**。计划里原本要做 topk=2 × depth=4，但拿到数据后发现官方
> 6.26 就是线性验证，而我们逐位相同还略高——线性已经吃满草稿质量，再写一个 tree
> attention kernel 是没有证据支撑的投入。
>
> 三是**置信度调度（confidence head）没启用**——DSpark 论文里它是可选优化，用来决定
> '这一轮验几个'，我把它留作后续。"

这段话的作用：展示你知道边界在哪，而且**砍功能是有依据的**（不是没做完）。

---

## 7. 预判追问

| 问题 | 回答要点 |
|---|---|
| 为什么不直接用 vLLM？ | 栈冲突（Torch 2.6 + FlashInfer 0.6.15 + 自研三面架构），且 vLLM 的 `method:"dspark"` 需要专门分支；只借鉴设计不复制代码 |
| 为什么 verify 不复用 decode 的图？ | 形状不同：decode 每序列 1 个 token，verify 是 1+k 且要读 varlen；分档维度也不一样 |
| 3.4× 为什么比论文低？ | batch=1、n=64，decode 阶段 GPU 本来就闲；verify 每步算 8 倍位置只换 6.5 个 token，净收益取决于接受长度与 batch。论文是大 batch + 长输出 |
| 温度采样怎么保证分布不变？ | 拒绝采样：`u < min(1, p/q)` 接受，拒绝后从 `normalize(max(0, p−q))` 恢复。关键是 `q` 必须用**采样器实际使用的分布**（含 top-k/p/min-p 截断），否则 top-k 下分布会偏 |
| 为什么先做 ngram 草稿？ | 零模型依赖，先把"多 token verify + KV 回滚 + 逐位一致"验干净；否则草稿错和回滚错分不开 |
| 草稿 KV 怎么滚？ | 每步先追加"上一步的 context 段 + block 个候选"，再 `crop(start)` 只留 context。猜错这条没有报错，只有接受率偏低 |
| 如果让你继续做，优先级？ | batch>1（唯一的功能缺口）→ MoE 支持 → 置信度调度。树形不做，除非有新的证据说明线性没吃满 |
