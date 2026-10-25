"""投机解码（speculative decoding）的草稿侧抽象。

本模块只负责**产生候选 token**，不参与验证与 KV 账本：

- 验证在 ``compute/model_runner.py`` 的 verify 路径（目标模型一次前向算
  ``[anchor] + drafts`` 的 logits）；
- 账本在 ``control/scheduler.py``（接受 j 个 → ``num_cached_tokens += 1+j``，
  只把验证通过的 token 追加进 ``Sequence.token_ids``）。

依赖方向：``control`` 可依赖本模块；本模块**不**依赖 ``compute`` / ``control``，
只依赖 ``src/data/sequence.py`` 的类型（避免与调度器形成环）。

Phase 1 只实现 ``NgramDrafter``（零模型、纯 CPU）——它的作用是先把
"多 token verify + KV 回滚 + 逐位一致"这条链路验证干净，再让 DSpark 的
GPU 草稿接进来（Phase 3 实现同一个 ``Drafter`` 接口）。
"""

from __future__ import annotations

from .base import Drafter
from .ngram import NgramDrafter

__all__ = ["Drafter", "NgramDrafter", "make_drafter"]


class _WorkerSideDrafter:
    """标记：草稿由 **worker** 产出（GPU 草稿模型），调度器只做额度预留。

    DSpark 的草稿需要目标模型的 aux hidden、自己的 KV cache 和 GPU 前向，无法在
    调度器（CPU 侧）产出。所以调度器退化为"按上限预留"：``write = 1 + k``，
    块分配幂等，实际写几位由 worker 回报的 ``d_lens`` 决定。

    ``propose()`` 永远返回空——真实的草稿在 ``ModelRunner._propose_dspark`` 里。
    """

    worker_side = True

    def propose(self, seq, k: int) -> list[int]:  # noqa: ARG002 - 接口兼容
        return []


def make_drafter(name: str, *, num_spec_tokens: int) -> Drafter:
    """按配置名构造草稿器。

    - ``ngram``：后缀匹配（无模型，Phase 1 的验证替身）
    - ``dspark``：worker 侧 GPU 草稿（本函数只返回一个预留标记）
    """
    if name == "ngram":
        return NgramDrafter(max_tokens=num_spec_tokens)
    if name == "dspark":
        return _WorkerSideDrafter()
    raise NotImplementedError(
        f"spec draft {name!r} is not implemented (ngram|dspark)"
    )
