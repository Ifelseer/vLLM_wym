# DSpark 的参照系：用官方 DeepSpec 跑同一条 draft

## 为什么需要它

`accept_len ≥ vLLM 基线 90%` 这条验收判据一直悬空，原因有两个：

1. **vLLM 跑不了 dspark**：`method="dspark"` 需要专门的支持分支，我们租机上的 vLLM 没有；
2. **论文 Table 1 的数字不可直接比**：prompt、输出长度、采样温度都不同。

所以参照系换成**官方仓库自己的 eval 入口**：同一份草稿、同一个目标模型、同一个
prompt、同一套解码循环，只是把"我们的引擎"换成"上游的 HF 实现"。两者算出来的
`acceptance_length` 才有可比性——差异只剩 attention/KV 实现。

## 为什么必须单独 venv

`requirements.txt` 钉死：

```
torch==2.9.1  transformers==5.10.2  triton==3.5.1  numpy==2.4.4
PyYAML  tqdm  tensorboard  matplotlib  typing_extensions
sentencepiece  safetensors  prettytable
datasets==4.8.5  openai==2.6.1        # 标注为"数据准备"，但 eval 会被
                                      # deepspec.utils.__init__ 间接带动
```

而本引擎的环境是 `torch 2.6.0a0 + transformers 4.57.6 + flash-attn 2.8.3`。
把 torch 升到 2.9.1 会让 flash-attn 的预编译 wheel 失效。**两者只能并存于不同 venv**。

```bash
# 独立环境，不与 vLLM_wym 共用
uv venv ~/venvs/deepspec
uv pip install -r requirements.txt --python ~/venvs/deepspec/bin/python
```

## 跑之前必须改一处

`eval.py` 里的任务是**写死的**，而且量很大：

```python
TASKS = [("gsm8k", 500), ("math500", 500), ("aime25", 30), ("humaneval", 164),
         ("mbpp", 256), ("livecodebench", 500), ("mt-bench", 80), ("alpaca", 500),
         ("arena-hard-v2", 500)]
```

合计约 3000 条 × `max_new_tokens=2048`，单卡要跑数小时。只为拿一个参照数字，
把它改成一个小任务：

```python
TASKS = [("gsm8k", 20)]
```

同时把 `--max-new-tokens` 降到 128（显存与时间都可控）。

## 命令

```bash
git clone https://github.com/deepseek-ai/DeepSpec.git ~/DeepSpec
cd ~/DeepSpec

# 注意：eval.py 会按 torch.cuda.device_count() spawn 进程，单卡只会起 1 个
~/venvs/deepspec/bin/python eval.py \
    --target_name_or_path ~/huggingface/Qwen3-8B \
    --draft_name_or_path  ~/huggingface/dspark_qwen3_8b_block7 \
    --max-new-tokens 128 \
    --temperature 0 \
    --seed 980406
```

输出表里关注三列：

```
#propose | accept_len | verify_rate | accept_rate@0..6
```

`accept_len` 就是我们要的参照；`accept_rate@i` 是逐位置接受率，能看出草稿在块内
第几位开始失效——如果我们的引擎在 `@3` 之后明显低于参照，说明 **markov 头的块内
依赖**没生效（我们的 `sample_draft_tokens` 走了 markov 路径，但值得核实）。

## 口径对齐（否则还是不可比）

| 项 | 我们的 bench | 官方 eval | 对齐方式 |
|---|---|---|---|
| 温度 | `--temperature 0`（greedy） | 默认 1.0 | 两边都用 **0** |
| 输出长度 | `--n 64/256` | `max_new_tokens` | 两边都设 **128** |
| prompt | bench 里那句素数列表 | gsm8k 题面 | 可接受差异——比的是 **accept_len 的量级**，不是绝对值 |
| KV | paged + FP8 权重 | HF DynamicCache + bf16 | 差异已知，只影响速度不影响 accept_len |

**能得出的结论只有一条**：我们的 `accept_len` 与上游实现相差多少（±10% 内即认为
草稿被正确接入）。不能得出速度对比——上游是 HF eager，我们是 paged+FP8+图。
