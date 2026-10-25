"""验证阶段的**纯函数**裁决逻辑（greedy）。

抽出来的理由：接受规则是整个投机解码里最容易 off-by-one 的地方，而它只在
GPU 上跑 e2e 才能验证的话，没有权重文件时就完全没有覆盖。这里把它写成
"给定 drafts 与 target 的逐位置 argmax，返回 (要追加的 token, 接受数)"，
可以在 CPU 上把边界全部枚举一遍。

位置约定（重要）：verify 一次前向处理位置 ``p .. p+m-1``，其中

    index 0     = anchor（上一步已经接受、KV 还没写的那个 token）
    index 1..k  = 草稿 d1..dk

位置 ``p+i`` 的 logits 预测的是 ``p+i+1`` 的 token，因此：

- ``preds[0]`` 决定 ``d1`` 是否被接受；
- ``preds[j]`` 决定 ``d_{j+1}``；
- 全接受时，``preds[m-1]`` 给出 bonus token。
"""

from __future__ import annotations

import torch


def accept_rejection(
    drafts: list[int],
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor | None = None,
    *,
    generator: torch.Generator | None = None,
) -> tuple[list[int], int]:
    """标准拒绝采样（Leviathan et al. / Chen et al.），保证输出分布**等于**目标分布。

    逐个位置 i 做检验：``u < min(1, p_i(d_i) / q_i(d_i))`` 则接受，继续下一位；
    一旦拒绝，就从 ``normalize(max(0, p_i − q_i))`` 采样一个 token 并停止。全部
    接受时从**最后一个位置的**目标分布取 bonus token。

    Args:
        drafts: 草稿 token（长度 k ≥ 1）。
        target_probs: 目标模型的**最终采样分布** ``[k+1, vocab]``（必须已经过
            温度 / top-k / top-p / min-p 截断并归一化，见
            ``Sampler.final_probs``）。
        draft_probs: 草稿分布 ``[k, vocab]``；``None`` 表示草稿是确定性的
            （one-hot，``q_i(d_i) = 1``）——ngram 与 greedy 草稿都属于这种。
        generator: 随机源（测试注入）。

    Returns:
        ``(tokens, accepted)``：``tokens`` = 接受前缀 + 1 个恢复/bonus token。

    为什么 ``None`` 时直接令 ``q = 1`` 是正确的：确定性草稿的分布是
    ``delta_{d_i}``，于是 ``q_i(d_i) = 1``、``p_i − q_i`` 等价于"把 d_i 那一维
    清零后重归一化"，这正是教科书里 greedy-draft 的接受/恢复公式。
    """
    k = len(drafts)
    if target_probs.shape[0] != k + 1:
        raise ValueError(
            f"target_probs must have len(drafts)+1 = {k + 1} rows, "
            f"got {target_probs.shape[0]}"
        )
    device = target_probs.device
    ids = torch.tensor(drafts, dtype=torch.long, device=device)
    rows = torch.arange(k, device=device)

    p_tok = target_probs[rows, ids]
    if draft_probs is None:
        ratio = p_tok.clamp(max=1.0)
        q_rows = None
    else:
        q_tok = draft_probs[rows, ids].clamp_min(1e-20)
        ratio = (p_tok / q_tok).clamp(max=1.0)
        q_rows = draft_probs

    u = torch.rand(k, device=device, generator=generator)
    rejected = u >= ratio
    hit = torch.nonzero(rejected, as_tuple=False)

    if hit.numel() == 0:
        # 全接受：bonus token 取自最后一个位置的目标分布
        token = int(torch.multinomial(target_probs[k], 1, generator=generator))
        return drafts + [token], k

    j = int(hit[0, 0])
    if q_rows is None:
        residual = target_probs[j].clone()
        residual[ids[j]] = 0.0
    else:
        residual = (target_probs[j] - q_rows[j]).clamp_min(0.0)
    total = float(residual.sum())
    if total <= 1e-12:
        # 数值退化（例如 p 与 q 几乎相同却被判拒绝）：回退到目标分布
        residual = target_probs[j]
        total = float(residual.sum())
        if total <= 1e-12:  # pragma: no cover - 目标分布本身非法
            raise ValueError("degenerate target distribution in rejection sampling")
    token = int(torch.multinomial(residual / total, 1, generator=generator))
    return drafts[:j] + [token], j


def accept_greedy(drafts: list[int], preds: list[int]) -> tuple[list[int], int]:
    """greedy 接受裁决。

    Args:
        drafts: 草稿 token（不含 anchor），长度 k。
        preds: 目标模型在 ``p..p+k`` 上的 argmax，长度必须是 ``k+1``。

    Returns:
        ``(tokens, accepted)``：``tokens`` 是本次要追加进序列的 token
        （= 被接受的草稿 + 1 个新 token），``accepted`` 是被接受的草稿数。

    Raises:
        ValueError: 长度不一致（verify 批次构造错位时会立刻暴露）。
    """
    k = len(drafts)
    if len(preds) != k + 1:
        raise ValueError(
            f"preds must have len(drafts)+1 = {k + 1} entries, got {len(preds)}"
        )
    j = 0
    while j < k and preds[j] == drafts[j]:
        j += 1
    # j < k：第 j+1 个草稿被拒，正确 token 就是 preds[j]（在 p+j 处的 argmax）
    # j == k：全部接受，bonus 取最后一个位置 p+k 的 argmax
    return drafts[:j] + [preds[j]], j
