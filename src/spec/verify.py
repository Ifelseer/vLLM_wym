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


def accept_greedy_device(
    drafts: torch.Tensor, preds: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """设备侧的 greedy 接受裁决（不经过主机）。

    为什么需要：CPU 版的 ``accept_greedy`` 必须先把 logits 的 argmax **读回主机**才能
    决定接受几个，形成一次全流水线同步点：

    ```
    GPU 算完 verify → .tolist()（等 GPU 排空）→ CPU 比较 → 下一轮
    ```

    设备版只做 elementwise 比较 + `cumprod` + 一次 gather，全程在 GPU 上，于是下一轮
    的输入可以**直接从设备拿**（这是 ZOS 能覆盖投机的前提）。

    一个能省掉大量工作的观察：``accept_greedy`` 返回的
    ``drafts[:j] + [preds[j]]`` 的**最后一个元素恒为** ``preds[j]``，所以下一轮的 anchor
    不需要把 tokens 拼出来，只需要 ``preds.gather(1, j)``。tokens 仍然打包返回，但只
    用于**延迟一拍**的 D2H（追加 `token_ids` / 统计）。

    Args:
        drafts: ``[B, k]`` int64，本拍草稿（k 可为 0）。
        preds: ``[B, k+1]`` int64，verify 各位置的 argmax。

    Returns:
        ``(next_token [B] int64, advances [B] int32, tokens [B, k+1] int64)``
        - ``next_token``：下一轮的输入 token（= ``preds[j]``）；
        - ``advances``：下一轮位置推进量（= ``1 + j``，直接喂 `advance_decode`）；
        - ``tokens``：本拍实际产出的 ``1+j`` 个 token（第 ``j+1`` 位之后是 0，按
          ``advances-1`` 切片）。

    与 CPU 版逐位等价由 ``tests/unit/test_zos_accept.py`` 全枚举保证。
    """
    if preds.dim() != 2 or drafts.dim() != 2:
        raise ValueError("accept_greedy_device expects 2-D tensors")
    b, k = drafts.shape
    if preds.shape[0] != b or preds.shape[1] != k + 1:
        raise ValueError(
            f"preds must be [B, k+1] = [{b}, {k + 1}], got {tuple(preds.shape)}"
        )

    rows = torch.arange(b, device=preds.device)
    if k == 0:
        j = torch.zeros(b, dtype=torch.long, device=preds.device)
    else:
        match = (preds[:, :k] == drafts).to(torch.int32)
        # 最长可接受前缀：任一位置不匹配后，后面的 cumprod 全为 0
        j = match.cumprod(dim=1).sum(dim=1).to(torch.long)

    next_token = preds[rows, j]
    advances = (j + 1).to(torch.int32)

    # 打包 tokens：pos < j 取草稿，pos == j 取 next_token，其余补 0
    drafts_pad = torch.cat(
        [drafts, torch.zeros((b, 1), dtype=drafts.dtype, device=drafts.device)], dim=1
    )
    idx = torch.arange(k + 1, device=preds.device).unsqueeze(0)
    tokens = torch.where(idx < j.unsqueeze(1), drafts_pad, torch.zeros_like(drafts_pad))
    tokens = tokens.scatter(1, j.unsqueeze(1), next_token.unsqueeze(1))
    return next_token, advances, tokens


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
