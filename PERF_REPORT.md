# vLLM_wym 单机性能测试报告（ZOS / 投机解码 / 页式草稿 KV）

- 日期：2026-11-03
- 代码：`/data/tmp/vLLM_wym`，HEAD `5038138`（工作树干净）
- 硬件：NVIDIA GeForce RTX 4090 D 24 GiB（驱动 570.124.06），宿主内存 20 GiB
- 软件：torch 2.9.0+cu128，transformers 5.10.2
- 模型：Qwen3-8B（fp8，在线量化）、Qwen3-0.6B（bf16）；草稿 `dspark_qwen3_8b_block7`（block=7，k=7）
- 方法：每个引擎单独进程（20 GiB 宿主装不下两个 8B，同进程会 exit 137）；`warmup=2`、`reps=3`
- 口径：`decode_tok_s = (n-1) × bs / (e2e - ttft)`；`accept_len = (accepted + steps) / steps`；
  峰值取进程内 `torch.cuda.max_memory_allocated()`（8B fp8 加载阶段宿主峰值 ~19 GiB，偶发 exit 137，失败项已重试）

> **结论先行**
> 1. ZOS 在普通 decode 上收益与步时占比一致：0.6B p50 −4.5~4.7%、吞吐 +4.6%；8B p50 −1.0%、吞吐 +1.0%。
> 2. dspark k=7 在 8B 上给出 **3.45×**（dense DynamicCache，n=128）；batch bs=2 聚合 **1.83×**；paged 路径一/二在短上下文比 dense 慢 11–14%，单序列不省显存。
> 3. 长上下文（n=512）暴露 ZOS 投机的两个问题：
>    - **已修复（本报告 patch）**：KV 预留少算一拍的 `advances` → 跨 256-token 块时 CUDA illegal memory access（修复前 4/4 崩，修复后不再崩，plain ZOS token 与 sync 逐位一致）。
>    - **未解决（重要）**：ZOS 投机在跨过第一个 KV block 边界后输出分叉（vs dense first diff = 262）；eager 与 verify 图输出彼此一致，说明不是图/桶问题，而是 spec-ZOS 设备侧 verify 元数据在块边界上的错误。**在修复前，长上下文不要开 `zos_spec`。**

---

## 1. 普通 decode：ZOS 开/关（n=128）

| 模型 | bs | sync tok/s | ZOS tok/s | Δ tok/s | p50（sync → zos） | p50 Δ | 峰值 |
|---|---|---|---|---|---|---|---|
| 0.6B | 1 | 483.6 | 506.0 | **+4.6%** | 2.057 → 1.961 ms | −4.7% | 3.71 GB |
| 0.6B | 4 | 1617.4 | 1685.7* | **+4.2%** | 2.444 → 2.334 ms | −4.5% | 3.72 GB |
| 8B fp8 | 1 | 95.9 | 96.8 | **+1.0%** | 10.429 → 10.322 ms | −1.0% | 11.76 GB |
| 8B fp8 | 4 | 374.4 | 378.6 | **+1.1%** | 10.655 → 10.535 ms | −1.1% | 11.76 GB |

- \*0.6B bs=4 的 ZOS 跑了 3 次：1513.7 / 1687.9 / 1685.7 tok/s，首跑是离群点；p50 三次均约 2.33 ms。
- 所有 sync/ZOS 输出 **逐 token 完全一致**（`tokens_all` 相等）。
- ZOS 计数器：lookahead_submitted=371、drains=16（含 5 次终局 fence + 11 次 ub 重锚）。
- 解读：ZOS 省的是每拍 host 开销，收益上限由 p50 中 GPU 步时的占比决定；0.6B（~2 ms/step）收益 4.5%，8B（~10.4 ms/step）只有 1%。

## 2. 投机解码（8B fp8，k=7，n=128，cg + verify graph）

| 配置 | bs | tok/s | 相对 baseline | 聚合/单序列 | accept_len | accept_rate | steps | proposed/accepted | 峰值 |
|---|---|---|---|---|---|---|---|---|---|
| baseline spec=0 | 1 | 96.8 | 1.00× | — | — | — | 0 | 0/0 | 11.75 GB |
| dense DynamicCache | 1 | **333.7** | **3.45×** | 333.7 | 6.000 | 71.4% | 67 | 469/335 | 16.56 GB |
| paged 路径一（16B 页） | 1 | 288.3 | 2.98× | 288.3 | 6.000 | 71.4% | 67 | 469/335 | 16.71 GB |
| paged 路径二 FA（256B 页） | 1 | 296.6 | 3.06× | 296.6 | 6.000 | 71.4% | 67 | 469/335 | 16.71 GB |
| batch bs2 dense（PaddedKVCache） | 2 | **609.7** | 6.30× | 304.9/序列 | 6.046 | 72.1% | 130 | 910/656 | 17.21 GB |
| batch bs2 paged | 2 | 480.1 | 4.96× | 240.1/序列 | 6.046 | 72.1% | 130 | 910/656 | 16.72 GB |
| spec + ZOS（`zos_spec`） | 1 | 287.7 | 2.97× | 0.86× vs dense spec | 5.662* | 68.5% | 71 | 483/331 | 17.19 GB |

- **正确性**：上表所有变体输出 token 与 baseline **逐位一致**（first diff = None）；batch dense vs paged 的 `tokens_all` 也逐位一致。
- paged 路径一/二在短上下文（158 token）比 dense 慢 11–14%；单序列 dense 是 `DynamicCache`（约 1.6 MB），而 512 页 ×16 token 的页池约 168 MB——**单序列用页式反而更占显存**。页式的对照对象是 batch/ZOS 的 `PaddedKVCache`（8 行 ×4096）。
- batch bs2：dense PaddedKVCache 聚合 609.7 tok/s（单序列 304.9，验证图仅 bs=1 生效）；paged 省 0.49 GB，但聚合吞吐 −21%。
- \*ZOS 的 accept 统计口径与同步路径不同（多 4 个 step、多 14 个 proposed、少 4 个 accepted；总产出 token 相同），吞吐是可靠指标；`accept_len` 不宜与同步路径直接比。

## 3. n=512 长上下文

| 配置 | tok/s | accept_len | 峰值 | 输出 vs dense | 说明 |
|---|---|---|---|---|---|
| dense spec n=512 | 157.1 | 4.571 | 16.60 GB | 基准 | 340 steps / 2380 proposed / 1214 accepted |
| ZOS spec n=512（图，修复后） | 113.2 | 3.957 | 17.20 GB | **first diff=262** | 393 steps / 2674 proposed / 1162 accepted |
| ZOS spec n=512（eager，修复后） | 97.3 | 3.957 | 17.14 GB | **first diff=262** | 与图版 token 完全一致 |
| plain ZOS n=512 | 96.7 | — | 11.76 GB | **与 sync 逐位一致** | sync n=512 = 95.6 tok/s |

- dense 在 n=512 的 accept_len 从 6.0 掉到 4.57（内容相关，prompt 是枚举素数），吞吐 157 tok/s。
- ZOS spec 的 n=512 吞吐数字仅作“能跑”的记录；由于输出分叉，**不能作为有效性能结论**。

### 3.1 已修复：KV 预留不足导致 CUDA illegal memory access

**触发**：ZOS（普通或投机）序列跨越第一个 256-token KV block。实测 ZOS spec 修复前在 n=256 与 n=512（开/关 verify 图）**4/4 崩溃**：

```
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
  at model_runner.py:786 (self.model in _verify_zos eager) / read_samples event.sync
```

**根因**：`AsyncDecode._can_lookahead` 按 `1+k` 预留 KV，但在途那拍的起点由设备侧 `advances`（1..1+k）决定：
最晚从 `num_cached + (1+k)` 开始，再写 `(1+k)` 个位置 → 需要 **`2(1+k)`**。
只预留 `1+k` 时，跨块那一步的 slot 会落到未分配的块（表项 -1 → 负 slot），Triton kernel 直接非法访存。
普通 decode 的 advances 恒为 1，同样需要 `n=2`。

**修复**：`n = 2 * (1 + spec_tokens)`（`src/control/async_decode.py`），并新增 CPU 回归测试
`tests/unit/test_zos_kv_reserve.py`（k=7 → 16，k=0 → 2）。

**验证**：
- n=512 ZOS spec 不再崩溃（修复前 4/4 崩）；
- n=512 plain ZOS 与 sync **逐位一致**（96.7 vs 95.6 tok/s）；
- 34 条 ZOS/spec 单测通过；
- patch `patches/zos_kv_reserve_fix.patch`（sha256 `f2e3f2c3…`）。

### 3.2 未解决：ZOS 投机跨 KV block 后输出分叉

**现象**：
- ZOS spec n=512 vs dense spec n=512，**first diff = 262**（绝对位置约 273，恰在 256 token 边界之后）；
- **eager（关 verify 图）与 graph 的输出彼此完全一致**（first diff = None），但两者都与 dense 分叉；
- accept_len 3.957 vs dense 4.571，说明不是“草稿质量下降”——投机 verify 必须是 lossless，输出本应与草稿质量无关；
- plain ZOS n=512 与 sync 逐位一致 → 普通 decode 的 `advance_decode`/页表路径正常，问题只在 spec-ZOS 的 verify 路径。

**已排除**：
- 不是 verify 图/桶选择（eager 与 graph token 相同）；
- 不是 KV 预留（patch 已修，且修前是崩溃、修后是分叉，两者独立）；
- 不是块表未分配（patch 后不再出现 -1 slot，plain ZOS 同边界正常）。

**当前怀疑**（待下一步验证）：spec-ZOS 的 `advance_verify` 在块边界构造的 `positions`/`slot_mapping`/`cu_seqlens_k` 与目标的因果对齐存在细节错误，导致跨块那一步的目标 logits 出错；与 draft/padded KV 无关（只影响草稿）。建议下一步：
1. 在跨边界前后 dump `starts`/`positions`/`slot_mapping`/`cu_seqlens_k`，与同步路径 `build_prefill` 逐值对照；
2. 对 first-diff 那一拍，同时跑 ZOS 与 sync 的 target logits，定位第一个不一致的 token 位置；
3. 检查该步设备块表是否已包含 block 1（时间戳/内容），以及 `_sync_block_tables` 的刷新时机；
4. 重点审查 `advance_verify` 的 `tables[r, pos // block_size]` 与 FA2 varlen + block_table 的 causal 对齐。

---

## 4. 修复清单（打包）

### 4.1 本报告新增（patch，未上游）

| 文件 | 内容 |
|---|---|
| `src/control/async_decode.py` | `_can_lookahead` KV 预留 `1+k → 2(1+k)`，修复跨 block 非法访存 |
| `tests/unit/test_zos_kv_reserve.py` | 新增回归：k=7 预留 16、k=0 预留 2 |

### 4.2 已在上游 `664ade4`（上一轮页式验收的 7 项）

1. FA 分支漏 `o_proj`（accept_len 1.114 vs 4.2667，静默错误）；
2. `begin_step` 多行不等长按批内 `kv_len` 预页；
3. `fa_args` 的 `block_table` 只取本步参与行；
4. 页表宽度改为 `num_pages`（避免长行 IndexError）；
5. 池空时回滚本轮已分配页；
6. GPU 测试 `sum(int)` TypeError；
7. 单序列显存对照口径（DynamicCache vs 页池），新增 batch 模式对照。

### 4.3 已在上游 `5038138`

- batch 显存对照测试：并发 ≥2 条、每次 `_run` 重置峰值（否则 `paged_peak < dense_peak` 必失败）。

> `patches/history/paged_fixes_upstream_664ade4.patch` 是上一轮页式修复的历史 patch，仅为追溯保留；当前 HEAD 已包含，勿重复应用。

---

## 5. 结论与建议

1. **短上下文（≤256 token）**：dspark k=7 收益 3.45×（dense）；ZOS 普通 decode 在 0.6B 上 +4.5%、8B 上 +1%；spec+ZOS 0.86× dense spec（n=128，输出正确）。
2. **页式草稿 KV**：收益在 batch/长上下文；短上下文单序列用页式会慢 11–14% 且更占显存。batch bs2 页式省 0.49 GB、吞吐 −21%。
3. **长上下文**：先不要开 `zos_spec`。崩溃已修，但跨 256-token 边界仍有输出分叉（first diff=262）。
4. **普通 ZOS**：n=512 已与 sync 逐位一致，可继续使用；p50 收益仍约 1%（8B）。
5. 8B fp8 单进程加载宿主峰值 ~19 GiB，20 GiB 容器偶发 exit 137，重试即可；正式跑建议 ≥24 GiB 宿主。

## 6. 复现

```bash
# 环境
cd /data/tmp/vLLM_wym            # HEAD 5038138
export PYTHONPATH=$PWD
PY=/data/tmp/venv-t510/bin/python   # transformers 5.10.2
B=/data/tmp/vllm_zos_perf_20261103/scripts/bench_one.py

# 普通 decode：sync / zos（8B fp8，n=128，bs=1）
$PY $B --kind zos --model ~/huggingface/Qwen3-8B --quantization fp8 --cg \
  --n 128 --max-num-seqs 1 --out /tmp/8b_sync.json
$PY $B --kind zos --model ~/huggingface/Qwen3-8B --quantization fp8 --cg \
  --n 128 --max-num-seqs 1 --zos --out /tmp/8b_zos.json

# 投机 dense / paged / FA / batch
D=~/huggingface/dspark_qwen3_8b_block7
SPEC="--kind spec --model ~/huggingface/Qwen3-8B --quantization fp8 --cg --draft-model $D"
$PY $B $SPEC --spec-tokens 7 --verify-graph --n 128 --out /tmp/spec_dense.json
$PY $B $SPEC --spec-tokens 7 --verify-graph --n 128 --draft-paged --page-size 16 --paged-pages 512 --out /tmp/spec_paged.json
$PY $B $SPEC --spec-tokens 7 --verify-graph --n 128 --draft-paged --page-size 256 --paged-pages 32 --draft-paged-fa --out /tmp/spec_fa.json
$PY $B $SPEC --spec-tokens 7 --verify-graph --n 128 --draft-batch --max-num-seqs 2 --out /tmp/spec_batch2.json

# spec+ZOS（当前仅建议 ≤256 token）
$PY $B $SPEC --spec-tokens 7 --verify-graph --n 128 --zos-spec --out /tmp/spec_zos.json

# 修复 patch
git apply /data/tmp/vllm_zos_perf_20261103/patches/zos_kv_reserve_fix.patch
$PY -m pytest tests/unit/test_zos_kv_reserve.py -q
```

## 7. 附件

```
PERF_REPORT.md
patches/zos_kv_reserve_fix.patch                  # 本次崩溃修复（+单测）
patches/history/paged_fixes_upstream_664ade4.patch# 上一轮页式修复（已在上游）
results/summary.json                              # 26 个 run 的机器可读汇总
results/perf/*.json                               # 每次 run 的原始结果（含 tokens）
results/logs/*.log, *.out                         # 运行日志（含 4 次崩溃现场）
scripts/bench_one.py                              # 单引擎 runner（所有场景）
scripts/agg_perf.py                               # 汇总表生成
scripts/perf_*.sh                                 # 矩阵 / 复核 / 隔离脚本
```

校验：
- `zos_kv_reserve_fix.patch` sha256 = `f2e3f2c3741b8ead022e435dce106559eb6418c4534e06fc0b61b3afcff5a7c3`
- patch 已用 `git apply --check` 验证可干净应用到 `5038138`；应用后工作树恢复干净。
