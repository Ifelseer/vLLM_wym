# 量化参考手册

面向实现与运维的**参考文档**，与 [quantization-bench.md](quantization-bench.md)
（实测数据）、[fp8-debug-report.md](fp8-debug-report.md)（FP8 落地调试）配套阅读。
检索基准日：**2026-10-12**（checkpoint 格式与 FlashInfer/flash-attn 能力矩阵以
该日为准，升级依赖后需复核）。

---

## 1. 快速上手

```bash
# 在线量化（BF16 模型，免重新下载）——FP8 需 SM89+
python run.py --model ~/huggingface/Qwen3-0.6B --quantization fp8 --cg
python run.py --model ~/huggingface/Qwen3-0.6B --quantization int8 --cg   # SM86 可用

# 预量化 checkpoint（自动识别 quantization_config）
python run.py --model ~/huggingface/Qwen3-0.6B-FP8 --quantization auto --cg
python run.py --model ~/huggingface/Qwen3-32B-AWQ --quantization auto      # 24G 单卡

# FP8 KV cache（SM89+；SM86 自动回退 BF16 并告警）
python run.py --model ~/huggingface/Qwen3-0.6B --quantization fp8 --quant-kv-cache --cg
```

`--quantization` 取值：`none`（默认，零改动基线）/ `auto`（按 checkpoint 的
`quantization_config` 决定）/ `fp8` / `int8` / `awq` / `gptq`。
`config.yaml` 同名字段可作默认值，CLI 优先。

---

## 2. Scale 模式

| 方法 | 权重粒度 | 激活粒度 | zeros | 前向 dtype |
|---|---|---|---|---|
| `fp8` per-tensor | 整层标量 | per-token 动态（`[M,1]`） | 无 | fp8 × fp8 → bf16 |
| `fp8` per-block | 128×128 block（`[⌈out/128⌉, ⌈in/128⌉]`） | 128-group 动态（`[M, ⌈in/128⌉]`） | 无 | 同上（Triton 内核） |
| `awq` | group=128（`[in/G, out]`） | 无（weight-only） | 有（asym） | bf16 激活 × int4 权重 |
| `gptq` sym | group=128 | 无 | 无（uint4b8 偏置 8） | 同上 |
| `gptq` asym | group=128 | 无 | 有 | 同上 |
| `int8` | per-channel（`[out]`） | per-token 动态（`[M,1]`） | 无（对称） | int8 × int8 → int32 |

**粒度选择的依据**：离群值在哪根轴上。激活的离群值在 token 间 → per-token；
权重的离群值集中在少数输出通道 → per-channel（INT8）或局部块（FP8 block）。

**FP8 KV cache**：per-layer **固定标量**，首次写入时按该层 k/v 的 `amax × 2`
校准后冻结。不能用 per-step 动态 scale（增量 KV 的写/读 scale 必须一致），
也不能 per-slot（cache 布局与 FA/FlashInfer 的 per-tensor scale 入参都不支持）。

---

## 3. Checkpoint 格式矩阵

内部统一布局（INT4）为 **in-major**：`qweight [in, out/8]` int32、
`zeros uint8 [in/G, out]`、`scales fp16 [in/G, out]`；加载期把各格式归一化到它。

| 来源 | qweight | qzeros | scales | 备注 |
|---|---|---|---|---|
| AWQ（vLLM 约定，in-major） | `[in, out/8]`，输出通道按 `[0,2,4,6,1,3,5,7]` 交织 | `[in/G, out/8]`，同一交织 | `[in/G, out]` | 交织在加载期还原；qzeros 与 qweight 共用同一 nibble 顺序 |
| AutoAWQ（out-major） | `[out, in/8]`，末维按输入打包 | `[out, in/G/8]`，按**组**打包 | `[out, in/G]` | 打包轴不同 → 解包→转置→重打包 |
| GPTQ | `[in/8, out]`，沿输入 K 打包 | `[in/G, out/8]` | `[in/G, out]` | `sym=true` 无 zeros，`zero_bias=8` |
| FP8（Qwen 官方） | `weight` fp8 `[out, in]` | — | `weight_scale`（block 128×128 或 `[out,1]`） | `modules_to_not_convert` 内的层由 catcher 反量化回 bf16 |
| INT8（compressed-tensors） | `weight` int8 `[out, in]` | — | `weight_scale` fp32 `[out]` | 带 `input_scale` 的 SmoothQuant 变体显式拒绝 |

**布局判定**：由张量形状 + 已知 `(in_full, out_full)` 唯一确定。`in == out`
（方形层）时 in-major 与 out-major 形状相同、无法判定 → 优先 in-major 并打印
`[WARN] INT4 qweight layout ambiguous`，此时应拿参考输出核对。

**不支持并显式报错的组合**：

| 组合 | 原因 |
|---|---|
| GPTQ `desc_act=true`（g_idx） | 激活按 `g_idx` 重排，与"组 = 连续 K 段"的内核假设冲突；需先实现 gather |
| SmoothQuant `input_scale` | 权重已按校准统计量预缩放，与动态 per-token 激活量化不兼容 |
| MoE 专家 + `int8` | 未实现；静默退回 bf16 会造成不可预期的半量化路径 |
| AWQ/GPTQ + `quantize_at_load` | 需要校准数据（AWQ 搜索 / GPTQ Hessian），不做在线量化 |

---

## 4. 硬件支持矩阵

| 能力 | SM86（3060） | SM89（4090） | SM90（H100） |
|---|---|---|---|
| FP8 权重 GEMM（Triton block / `_scaled_mm`） | ✗ 不支持（e4m3 无硬件 MMA） | ✓ | ✓ |
| FP8 KV cache（FlashInfer paged） | ✗ 自动回退 BF16 KV | ✓ | ✓ |
| INT4 weight-only（Triton dequant-GEMM） | ✓ | ✓ | ✓ |
| INT8 W8A8（`_int_mm` / Triton 小 M） | ✓ | ✓ | ✓ |
| MoE FP8 block-scale grouped GEMM | ✗ | ✓（Triton） | ✓（cutlass，SM90 专属） |
| MoE INT4 grouped GEMM（Triton） | ✓ | ✓ | ✓ |

启动日志会打印实际生效的量化方法、scale 模式与回退原因；不支持的组合在
`EngineConfig` 解析期或层构造期 raise，不静默降级。

---

## 5. 测试矩阵

```
tests/
├── unit/                          # 无 GPU，本地必跑
│   ├── test_quant_config.py       # 解析/门控/报错路径
│   ├── test_quant_loader.py       # 加载器：5 类层 TP 切分 + MoE EP 过滤
│   ├── test_quant_model_load.py   # 模型级完整 load_model 链路（FP8 在线量化，dense+MoE）
│   ├── test_quant_int4.py         # 打包/解包往返、交织还原、方向识别、TP 切分
│   ├── test_quant_int4_moe.py     # 专家 INT4：[up|gate] 段序、TP、EP、out-major
│   ├── test_quant_int8.py         # 量化往返、per-channel vs per-tensor、在线 finalize
│   ├── test_quant_int8_model_load.py  # INT8 模型级 + SmoothQuant 拒绝
│   └── test_fp8_kv_csr.py         # block_tables→CSR（紧凑 packing/trash 槽）
├── gpu/                           # 需 CUDA；按 SM 自动跳过
│   ├── test_quant_fp8.py          # dense per-block/per-tensor + MoE 数值（SM89+）
│   ├── test_quant_kv.py           # FP8 KV decode/prefill/roundtrip vs FA2（SM89+）
│   ├── test_quant_int4.py         # dense + MoE 数值（任意 SM；本地 3060 实跑）
│   ├── test_quant_int8.py         # W8A8 数值 + 吞吐微基准（任意 SM61+）
│   └── test_quant_e2e.py          # teacher-forced top-1/logit gate + eager/CG 一致性
└── distributed/
    └── test_quant_parallel.py     # tp1/tp2/tp2+ep 贪心一致 + 跨格式一致率
```

**验收口径**

| 层级 | 判据 |
|---|---|
| kernel 数值 | rel = max\|Δ\|/max\|ref\|；FP8 dense <0.08、MoE <0.12；INT4 dense <0.02、MoE <0.25；INT8 <0.02 |
| 端到端 | **teacher-forced**：相同上下文下 top-1 一致率 ≥95%（仅计 BF16 margin ≥0.5 的决定性位置），top-1 logit 最大偏差 ≤1.5；同档位 eager vs CG 逐 token 完全一致。自由生成的"逐 token 一致率 ≥99%"不可达——近似并列处量化噪声会翻转 argmax 并级联（详见 `test_quant_e2e.py` 文档） |
| 吞吐 | 同 prompt 的 TTFT / decode tok/s / e2e；**CG replay 口径**（匹配部署形态，见 §7） |

阈值不是调出来的：先建「理想实现」参照系（同粒度、同 scale 语义、朴素算子），
把总误差拆成「内核 vs 理想」与「理想 vs 高精度」两段——FP8 dense 实测
0.0075 vs 0.0395，说明误差 89% 来自 e4m3 本身而非实现。

---

## 6. 关键实现约定（改代码前必读）

1. **量化参数保护按归属判断**，不要按 dtype 白名单：`worker._to_device()`
   跳过 `quant_method.extra_params()`，否则 `model.to(dtype)` 会把 fp8 权重和
   fp32 scale 一起转成 bf16。
2. **CUDA Graph 内的 kernel 不得有同步/动态分配**：`Triton autotune` 的
   `do_bench`、FlashInfer `plan` 的 D2H、`torch.cuda.empty_cache()` 都要放在
   录图外（`CUDAGraphRunner.capture(on_warmup=...)` 是为此加的钩子）。
3. **MoE 池化缓冲契约**：`fused_moe_forward` 返回按形状复用的缓冲，调用方必须
   立即消费；跨调用持有需 `.clone()`（测试里踩过"量化结果逐位等于未量化"的坑）。
4. **CSR indices 必须紧凑 packing**：`block_tables→FlashInfer` 转换用位置散射
   （`pos = indptr[i] + j`），无效条目散射到末尾 trash 槽，避免任何 host 同步。
5. **INT4 打包轴**：in-major 的 nibble 表示**输出通道**（`8*n8+j`），
   GPTQ 的 nibble 表示**输入通道**；两者不能靠转置互转，必须解包重打包。

---

## 7. bench 回归

```bash
# 跨量化档位对比（在线量化，同 prompt）
python bench/quant_bench.py --model ~/huggingface/Qwen3-0.6B \
    --quantizations none,fp8,int8 --cg --n 64 --reps 3 --json out.json

# 预量化 checkpoint vs BF16 兄弟模型
python bench/quant_bench.py --model ~/huggingface/Qwen3-0.6B \
    --quantized-model ~/huggingface/Qwen3-0.6B-FP8 --cg

# 与官方 vLLM 对比（需同样的量化 checkpoint）
python bench/vs_vllm.py --model ~/huggingface/Qwen3-0.6B --quantization fp8
```

输出列：TTFT / decode tok/s / e2e tok/s / peak GB / 显存相对基线 / token 一致率。

**口径纪律**：int8 在 eager 口径下比 bf16 慢（3 个 kernel 的 launch 开销），
在 CUDA Graph replay 口径下快 1.4~1.8×。报性能必须注明口径，且优先用
`--cg`（引擎 decode 的真实形态）。0.6B 这类小模型不触带宽瓶颈，FP8/INT8 的
吞吐会接近持平甚至略负——要看的是大模型的显存收益与长上下文 KV 容量。

---

## 8. DSpark（投机解码）扩展点

本轮**不实现** DSpark，只保留以下挂载点，避免后续改动扩散：

| 挂载点 | 位置 | 说明 |
|---|---|---|
| 方法注册表 | `src/quant/methods/__init__.py::_MODULE_MAP` | draft 模型可用不同量化档位（如 draft=int4、target=fp8）：注册表已是 `method → 构造器` 的映射，按层前缀分派即可；`get_quant_method(prefix=...)` 已支持按前缀过滤 |
| 量化感知的层工厂 | `src/quant/__init__.py::get_quant_method` | `QuantMethod.forward` 与 TP 通信解耦，draft 的 `forward` 可直接复用同一方法 |
| 调度双序列 | `src/control/scheduler.py::Scheduler.schedule` | `waiting` / `running` 队列是 draft/target 双序列的天然挂载处；`SeqSchedule` 已含 `num_cached_tokens`，投机接受/拒绝的 token 回滚需要在此扩展 |
| KV 复用 | `src/compute/fp8_kv.py::Fp8KVManager` | per-layer 固定 scale 对 draft/target 共享 KV 池是兼容的（同一层同一 scale）；若 draft 用独立 KV 池，按 layer 偏移扩展 `_kv_slice` |
| 形状无关的 kernel | `src/quant/kernels/*` | 三个 kernel 都接受任意 `M`（含 M=1 与 CUDA Graph 分档 tile），draft 的多 token 并行 verify 只需传更大的 M，无需 reconfigure |

**未验证的交互**（实现 DSpark 时必须先测）：draft 与 target 使用不同量化档位时，
KV cache 的 scale 是否共享；投机 verify 阶段的 `M = k+1` 形状是否触发 kernel 的
tile 边界（尤其 INT8 的 `M<=16` Triton 路径与 `M>16` `_int_mm` 路径的切换点）。

---

## 9. 已知限制

- **AWQ 交织约定**按 vLLM `awq_gemm` 内核证据实现（qweight 与 qzeros 共用同一
  nibble 顺序），尚未用真实 checkpoint 逐位比对；方形层布局歧义见 §3 的告警。
- **FP8 + prefill `torch.compile`** 主动禁用（Dynamo × Triton autotune 未验证）。
- **MoE + FP8 + CUDA Graph** 组合未在实机验证。
- **INT8 MoE 专家**未实现（显式 raise）。
- FP8 per-tensor 路径每次调用分配 `[1,N]` 广播 scale（非默认热路径，未池化）。
- 长 prompt（8K）TTFT 的 FP8 开销未实测。
