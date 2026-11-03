"""ZOS 在途拍的 KV 预留上界：必须覆盖"设备侧起点 + 本拍写入"两段。

真实事故（4090 上机，n=256/512，4/4 崩）：

    在途那拍的起点由设备侧 advances 决定，最大 1+k；
    最晚从 num_cached + (1+k) 开始，然后再写 (1+k) 个位置
    ⇒ 需要 num_cached + 2(1+k)

只按 `1+k` 预留时，序列跨过 KV 块边界（256 token）会让 slot 落到未分配的块
（表项 -1 → 负 slot），Triton kernel 直接 `illegal memory access`。

这条是**纯算术**，所以能在 CPU 上钉死：不测"跑起来崩不崩"，只测"预留量对不对"。
"""

from __future__ import annotations

from types import SimpleNamespace

from src.control.async_decode import AsyncDecode, DecodeTicket


def _ad_and_kv(spec_tokens: int):
    recorded: list[int] = []
    seq = SimpleNamespace(
        is_finished=False,
        num_completion_tokens=0,
        max_tokens=128,
        temperature=0.0,
        block_table=[0, 1],
        num_cached_tokens=250,  # 正好在第一个 256 block 的边界附近
    )
    blocks = SimpleNamespace(
        available_slots=100,
        needed_append=lambda s, n=1: 0,
    )
    kv = SimpleNamespace(
        blocks=blocks,
        may_append=lambda s, n=1: recorded.append(n),
    )
    core = SimpleNamespace(
        config=SimpleNamespace(spec_tokens=spec_tokens, max_num_seqs=4),
        scheduler=SimpleNamespace(waiting=[], running=[seq], eos={2}),
        kv=kv,
    )
    prev = DecodeTicket(sched=SimpleNamespace(decodes=[seq]), n_p=1, n_d=1)
    return AsyncDecode(core), prev, recorded


def test_spec_reserves_two_steps():
    """k=7：起点最多晚 8、再写 8 → 预留 16，而不是 8。"""
    ad, prev, recorded = _ad_and_kv(spec_tokens=7)
    assert ad._can_lookahead(prev) is True
    assert recorded == [16]


def test_plain_decode_reserves_two_positions():
    """普通 decode：起点晚 1、再写 1 → 预留 2。"""
    ad, prev, recorded = _ad_and_kv(spec_tokens=0)
    assert ad._can_lookahead(prev) is True
    assert recorded == [2]
