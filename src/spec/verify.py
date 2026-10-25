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
