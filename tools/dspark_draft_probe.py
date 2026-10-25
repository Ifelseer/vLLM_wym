"""把「我们的草稿」与「官方草稿」放在**同一个输入**上对拍。

为什么需要它：aux 数值对拍的结论是"略差但不是结构错"（相对误差 ~1%，各槽持平），
所以 accept_rate@i 的衰减不能归因于 context 特征。剩下的可能是草稿内部的两层：

1. **块前向**：位置 / 掩码 / ctx 的 fc 投影 / KV 追加 → 体现在 `block_hidden` 与
   `base_logits`；
2. **markov 链**：块内逐位置采样的顺序与偏置 → 体现在最终 tokens（base_logits 相同时）。

把 ctx 与 anchor 固定成同一份（脚本用确定性公式生成，不依赖 RNG 版本），两边各跑
一次，就能按 [hidden → logits → tokens] 三级定位，不必在完整引擎里盲试。

用法（两次分别在不同 venv 里跑，第三次随便）：

    # vLLM_wym venv（torch 2.6 + transformers 4.57，用我们 vendored 的实现）
    python tools/dspark_draft_probe.py --mode ours \\
        --model ~/huggingface/dspark_qwen3_8b_block7 --start 68 --anchor 1234 \\
        --out /data/tmp/probe_ours.pt

    # deepspec venv（torch 2.9 + transformers 5.x，用上游 deepspec.modeling）
    python tools/dspark_draft_probe.py --mode theirs \\
        --model ~/huggingface/dspark_qwen3_8b_block7 --start 68 --anchor 1234 \\
        --out /data/tmp/probe_theirs.pt

    # 任意一边
    python tools/dspark_draft_probe.py --mode compare \\
        --a /data/tmp/probe_ours.pt --b /data/tmp/probe_theirs.pt
"""

from __future__ import annotations

import argparse

import torch

# 与 checkpoint 无关的确定性 ctx：不依赖 RNG 版本，两个 venv 生成的一致
SCALE = 3.0


def make_ctx(seg: int, feat: int) -> torch.Tensor:
    n = torch.arange(seg * feat, dtype=torch.float32)
    v = torch.sin(n * 0.001) * SCALE
    return v.reshape(1, seg, feat).to(torch.bfloat16)


def _block_setup(model, anchor: int, start: int):
    """块输入：第 0 位放 anchor，其余 mask；位置区间 [cached, start+block)。"""
    block = int(model.block_size)
    ids = torch.full((1, block), int(model.mask_token_id), dtype=torch.long)
    ids[0, 0] = int(anchor)
    positions = torch.arange(0, start + block, dtype=torch.long).unsqueeze(0)
    return ids, positions, block


def _dump(path: str, *, block_hidden, base_logits, tokens, start, anchor) -> None:
    torch.save(
        {
            "block_hidden": block_hidden.float().cpu(),
            "base_logits": base_logits.float().cpu(),
            "tokens": list(tokens),
            "start": int(start),
            "anchor": int(anchor),
        },
        path,
    )
    print(f"saved {path}: tokens={tokens}")


def run_ours(args) -> None:
    from src.spec.dspark.modeling import Qwen3DSparkModel

    model = Qwen3DSparkModel.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).eval()
    ids, positions, block = _block_setup(model, args.anchor, args.start)
    ctx = make_ctx(args.start, len(model.target_layer_ids) * model.config.hidden_size)

    from transformers.cache_utils import DynamicCache

    pkv = DynamicCache()
    with torch.inference_mode():
        hidden = model._forward_backbone(
            position_ids=positions,
            noise_embedding=model.embed_tokens(ids),
            target_hidden_states=ctx,
            attention_mask=None,
            past_key_values=pkv,
            use_cache=True,
            is_causal=False,
        )
        block_hidden = hidden[:, :block, :]
        base_logits = model.compute_logits(block_hidden)
        tokens, _ = model.sample_draft_tokens(
            base_logits,
            first_prev_token_ids=torch.tensor([args.anchor], dtype=torch.long),
            temperature=0.0,
            hidden_states=block_hidden,
        )
    _dump(
        args.out,
        block_hidden=block_hidden[0],
        base_logits=base_logits[0],
        tokens=tokens[0].tolist(),
        start=args.start,
        anchor=args.anchor,
    )


def run_theirs(args) -> None:
    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel

    model = Qwen3DSparkModel.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).eval()
    ids, positions, block = _block_setup(model, args.anchor, args.start)
    ctx = make_ctx(args.start, len(model.target_layer_ids) * model.config.hidden_size)

    from deepspec.eval.dspark.draft_ops import forward_dspark_draft_block
    from transformers import DynamicCache

    pkv = DynamicCache()
    with torch.inference_mode():
        hidden = forward_dspark_draft_block(
            model,
            draft_input_ids=ids,
            position_ids=positions,
            past_key_values_draft=pkv,
            target_hidden_states=ctx,
            start=args.start,
            block_size=block,
        )
        block_hidden = hidden[:, :block, :]
        base_logits = model.compute_logits(block_hidden)
        tokens, _ = model.sample_draft_tokens(
            base_logits,
            first_prev_token_ids=torch.tensor([args.anchor], dtype=torch.long),
            temperature=0.0,
            hidden_states=block_hidden,
        )
    _dump(
        args.out,
        block_hidden=block_hidden[0],
        base_logits=base_logits[0],
        tokens=tokens[0].tolist(),
        start=args.start,
        anchor=args.anchor,
    )


def run_compare(args) -> None:
    a = torch.load(args.a, map_location="cpu")
    b = torch.load(args.b, map_location="cpu")
    for k in ("start", "anchor"):
        assert a[k] == b[k], f"{k} mismatch: {a[k]} vs {b[k]}"

    dh = (a["block_hidden"] - b["block_hidden"]).abs()
    scale = b["block_hidden"].abs().max().item() or 1.0
    print(f"[1] block_hidden  max|diff|={dh.max().item():.6g}  rel={dh.max().item()/scale:.3%}")

    la = a["base_logits"].argmax(-1).tolist()
    lb = b["base_logits"].argmax(-1).tolist()
    dl = (a["base_logits"] - b["base_logits"]).abs()
    print(f"[2] base_logits   argmax ours={la}")
    print(f"                  argmax them={lb}   max|diff|={dl.max().item():.6g}")

    print(f"[3] tokens        ours={a['tokens']}")
    print(f"                  them={b['tokens']}")
    same = a["tokens"] == b["tokens"]
    print()
    if not same and la == lb:
        print("→ 结论：块前向一致、argmax 一致，**差异出在 markov 链**（块内采样）")
    elif la != lb and dh.max().item() / scale > 1e-2:
        print("→ 结论：**块前向**就不同（hidden 相对误差 >1%），先查位置/掩码/ctx 投影")
    elif la != lb:
        print("→ 结论：hidden 接近但 logits argmax 不同 —— 疑似数值噪声，看 max|diff| 是否贴近 bf16 尾数")
    else:
        print("→ 结论：两级都一致（此输入下无差异）")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=("ours", "theirs", "compare"))
    p.add_argument("--model", default="")
    p.add_argument("--start", type=int, default=68)
    p.add_argument("--anchor", type=int, default=1234)
    p.add_argument("--out", default="")
    p.add_argument("--a", default="")
    p.add_argument("--b", default="")
    args = p.parse_args()

    if args.mode == "ours":
        run_ours(args)
    elif args.mode == "theirs":
        run_theirs(args)
    else:
        run_compare(args)


if __name__ == "__main__":
    main()
