# 与官方 DeepSpec 对齐 accept_rate@i 的两步排查法

## 背景：已知的差异形状

同 20 条 gsm8k（seed 980406，n=128，T=0，k=7）：

```
        @0      @1      @2      @3      @4      @5      @6     accept_len
官方   0.938   0.874   0.800   0.745   0.683   0.640   0.578    6.26
我们   0.938   0.853   0.749   0.662   0.562   0.440   0.328    5.53
Δ      +0.000  -0.020  -0.051  -0.083  -0.121  -0.199  -0.250
```

**@0 精确一致、越靠后越差**。这个形状的含义：

- `markov_head.py` 是从上游原样 vendor 的，**同一份代码不可能产生系统性差异**——
  它不是嫌疑，是被喂了错的 `base_logits`；
- `@0` 精确说明 anchor 位置、mask 填充、position ids、`fc` 投影、第一条 markov
  偏置（来自 anchor）全都正确；
- 剩下的可疑面只有"位置 ≥1 才能看到的东西"：**逐步追加的 context 特征**、
  或**草稿 KV 的 append/crop 序列**。

BF16 对照（5.429）没有变好，已排除 FP8 target。

---

## 第一步：aux 数值对拍（先排除"context 特征本身错了"）

### ① 用 HF 拿参考值（deepspec venv，transformers 5.x）

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "/root/huggingface/Qwen3-8B"
m = AutoModelForCausalLM.from_pretrained(
    path, dtype=torch.bfloat16, attn_implementation="sdpa"
).cuda().eval()
tok = AutoTokenizer.from_pretrained(path)

prompt = "..."  # 与 gsm8k 同一条（用 encode_chat_messages 编码）
ids = tok(prompt, return_tensors="pt").input_ids.cuda()
with torch.inference_mode():
    out = m(input_ids=ids, output_hidden_states=True, use_cache=False)

# DSpark 口径：hidden_states[layer_id + 1]，concatenate
ref = torch.cat([out.hidden_states[i + 1] for i in [1, 9, 17, 25, 33]], dim=-1)
torch.save(ref.float().cpu(), "/data/tmp/ref_aux.pt")   # [1, L, 5*4096]
```

### ② 在我们引擎里 dump 同一段 prompt 的 aux

在 `ModelRunner._run_prefill` 里 `_dspark_pending[...] = ...` 那一行后面加临时打印：

```python
import torch as _t
_t.save(self._dspark_pending[seqs[0].seq_id].float().cpu(), "/data/tmp/our_aux.pt")
```

跑一条与上一步**完全相同**的 prompt（`max_tokens=1` 即可，prefill 后就会写文件）。

### ③ 比对

```python
import torch
ref = torch.load("/data/tmp/ref_aux.pt").squeeze(0)   # [L, 5H]
ours = torch.load("/data/tmp/our_aux.pt").squeeze(0)  # [L, 5H]
print("shape", ref.shape, ours.shape)
d = (ref - ours).abs()
print("max|diff|", d.max().item(), "| ref|max|", ref.abs().max().item())
print("per-slot max:", [d[:, i*4096:(i+1)*4096].max().item() for i in range(5)])
print("per-pos max (前 8 个位置):", d.max(dim=1).values[:8].tolist())
```

判据：

| 结果 | 结论 | 下一步 |
|---|---|---|
| `max|diff| / |ref|max| < 1e-2` | aux 正确（bf16 量级差异） | 走第二步 |
| **O(1)**，或某些层/某些位置单独大 | aux 采集错 | 看 `per-slot` 定位是哪一层；看 `per-pos` 定位是否整段偏移 |
| 前几个位置对、之后错 | **切片起点错**（`[:accepted+1]` vs `[1:accepted+2]`） | 直接查 append 的切片 |

---

## 第二步：打印两边的 propose 输入，找第一个分歧点

### 我们的 `src/spec/dspark/drafter.py::propose`（放在 `_forward_backbone` 之前）

```python
print("OURS", dict(
    start=start, cached=cached, seg=seg, anchor=int(anchor_token_id),
    rows=int(ctx.shape[1]),
    ctx_mean=float(ctx.float().abs().mean()),
    ctx_head=ctx.flatten()[:4].tolist(),
), flush=True)
```

### 上游 `deepspec/eval/dspark/evaluator.py::_propose`（放在 `forward_dspark_draft_block` 之前）

```python
print("THEIR", dict(
    start=start,
    cached=context.past_key_values_draft.get_seq_length(),
    seg=int(context.target_hidden_states.shape[1]),
    anchor=int(draft_input_ids[0, 0]),
    rows=int(context.target_hidden_states.shape[1]),
    ctx_mean=float(context.target_hidden_states.float().abs().mean()),
    ctx_head=context.target_hidden_states.flatten()[:4].tolist(),
), flush=True)
```

两边各跑 3 步（第一条样本足够），按顺序 diff：

| 分歧字段 | 结论 |
|---|---|
| `seg` / `rows` 不同 | **我们给的 `accepted+1` 错**（最可能，且直接解释越后越差） |
| `cached` 不同 | 草稿 KV 的 `crop` 没生效或 crop 目标错 |
| `start` / `anchor` 不同 | 账本推进错位 |
| 全同但 `ctx_head` 不同 | aux 数值/顺序问题（回第一步） |
| **全同** | 只剩 verify 写回的 target KV（~1ulp）——那就用 `spec_exact_kv` 复测 |

---

## 为什么值得跑这两步

`accept_len` 是**结果**，`accept_rate@i` 是**形状**。形状已经告诉我们"前位对、后位散"，
而 `@0` 的精确一致把一半的嫌疑（模型、权重、量化、position、markov 代码）都排除了。
剩下的两种可能（context 段错位 / KV append 错位）**都能被第一步的对拍直接分开**，
不需要在完整引擎里盲试。
