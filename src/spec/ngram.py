from __future__ import annotations

from ..data.sequence import Sequence

# ==============================================================================
# N-gram（prompt lookup）草稿器。
#
# 用途：Phase 1 的**验证替身**。它零模型、纯 CPU，但能产生真实的草稿
# （重复文本 / 引用型 prompt 上命中率不低），用来把「多 token verify + KV
# 回滚 + 逐位一致」这条链路单独验证干净，避免与 DSpark 的算法问题混在一起。
#
# 为什么不直接上 DSpark：DSpark 的草稿要用目标模型的 5 层 hidden，如果
# 第一次就把「草稿」和「验证」两处新代码叠在一起，出错时无法判断是草稿错
# 还是 verify/回滚错。
# ==============================================================================


class NgramDrafter:
    """在 ``seq.token_ids`` 里找最近一次出现的后缀，取其后续 token 作为草稿。

    参数：
        n: 参与匹配的后缀长度（默认 3）。
        max_tokens: 单次最多给出的草稿数（与 ``--spec-tokens`` 对齐）。
        window: 只回看最近多少个位置（默认 256）。限制它是为了让复杂度与
            上下文长度解耦——prompt lookup 的经验窗口本来就不需要很长。
    """

    def __init__(
        self, *, n: int = 3, max_tokens: int = 7, window: int = 256
    ) -> None:
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        self.n = n
        self.max_tokens = max_tokens
        self.window = max(window, n + 1)

    def propose(self, seq: Sequence, k: int) -> list[int]:
        ids = seq.token_ids
        total = len(ids)
        n = self.n
        if n <= 0 or total <= n:
            return []
        k = min(k, self.max_tokens)
        # 生成长度预算：已到 max_tokens 就不再多给（给了也会被调度器丢弃）
        k = min(k, seq.max_tokens - seq.num_completion_tokens)
        if k <= 0:
            return []

        suffix = ids[total - n :]
        # 排除末尾这段自身：从 total-n-1 往前找
        lo = max(0, total - self.window)
        for pos in range(total - n - 1, lo - 1, -1):
            if ids[pos : pos + n] == suffix:
                follow = ids[pos + n : pos + n + k]
                return list(follow)
        return []
