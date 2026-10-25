"""Phase 2 的 CPU 单测：采样器"最终分布" + 拒绝采样的分布等价性。

投机解码的无损性有两条口径：
- **greedy**：逐 token 相等（Phase 1 的 ``accept_greedy``）；
- **温度采样**：输出 token 的**分布**必须等于目标分布 p（本文件）。

第二条最容易写错：接受概率用 ``min(1, p/q)``、拒绝后用 ``normalize(p−q)`` 恢复、
全接受时取 bonus。这三处任何一处偏差都不会让程序崩，只会让分布悄悄跑偏——所以
必须用 Monte-Carlo 频次去验，而不是"跑通就行"。
"""

from __future__ import annotations

import pytest
import torch

from src.compute.sampler import Sampler
from src.spec.verify import accept_rejection

_TRIALS = 20000
_TOL = 0.02  # n=20000 时二项标准差 ≤0.0036，2% 约 5.6σ


def _freq(tokens: list[int], vocab: int) -> torch.Tensor:
    counts = torch.zeros(vocab, dtype=torch.float64)
    for t in tokens:
        counts[t] += 1
    return counts / max(len(tokens), 1)


# ---------------------------------------------------------------- #
# 1. Sampler.final_probs：必须是"采样器实际使用的分布"
# ---------------------------------------------------------------- #
class TestFinalProbs:
    def setup_method(self):
        self.sampler = Sampler()
        self.logits = torch.tensor([[1.0, 2.0, 3.0, 0.0]])
        self.temps = torch.tensor([1.0])

    def test_normalized(self):
        p = self.sampler.final_probs(self.logits, self.temps)
        assert p.shape == (1, 4)
        assert p.sum().item() == pytest.approx(1.0, abs=1e-6)
        expected = torch.softmax(self.logits, dim=-1)
        assert torch.allclose(p, expected, atol=1e-6)

    def test_temperature_scaling(self):
        p = self.sampler.final_probs(self.logits, torch.tensor([0.5]))
        expected = torch.softmax(self.logits / 0.5, dim=-1)
        assert torch.allclose(p, expected, atol=1e-6)

    def test_all_greedy_is_onehot(self):
        p = self.sampler.final_probs(self.logits, self.temps, all_greedy=True)
        assert p[0].tolist() == [0.0, 0.0, 1.0, 0.0]

    def test_top_k_one_is_onehot(self):
        p = self.sampler.final_probs(
            self.logits,
            self.temps,
            top_ks=torch.tensor([1]),
            use_top_k=True,
            max_top_k=1,
        )
        assert p[0].tolist() == [0.0, 0.0, 1.0, 0.0]

    def test_top_k_two_masks_tail(self):
        p = self.sampler.final_probs(
            self.logits,
            self.temps,
            top_ks=torch.tensor([2]),
            use_top_k=True,
            max_top_k=2,
        )
        assert p[0, 2] > 0 and p[0, 1] > 0
        assert p[0, 0] == 0.0 and p[0, 3] == 0.0
        assert p.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_top_p_keeps_first(self):
        # top_p 极小：只保留概率最大的那个（首项恒保留）
        p = self.sampler.final_probs(
            self.logits,
            self.temps,
            top_ps=torch.tensor([0.01]),
            use_top_p=True,
        )
        assert p[0].tolist() == [0.0, 0.0, 1.0, 0.0]

    def test_min_p_filters(self):
        # max_p=0.6439，min_p=0.5 → 只保留 p ≥ 0.32 的项（只有 argmax）
        p = self.sampler.final_probs(
            self.logits,
            self.temps,
            min_ps=torch.tensor([0.5]),
            use_min_p=True,
        )
        assert p[0, 2] == pytest.approx(1.0, abs=1e-6)
        assert p[0, 0] == 0.0

    def test_matches_forward_sampling(self):
        """Monte-Carlo：forward 的实际采样频次应逼近 final_probs。"""
        gen = torch.Generator().manual_seed(0)
        vocab = 4
        samples = torch.empty(_TRIALS, dtype=torch.long)
        for i in range(_TRIALS):
            samples[i] = self.sampler(
                self.logits,
                self.temps,
                generator=gen,
                all_greedy=False,
                any_greedy=False,
            )[0]
        got = _freq(samples.tolist(), vocab)
        want = self.sampler.final_probs(self.logits, self.temps)[0].double()
        assert torch.all((got - want).abs() < _TOL), f"got={got.tolist()} want={want.tolist()}"

    def _mc_compare(self, *, trials: int, tol: float, **kw) -> None:
        gen = torch.Generator().manual_seed(7)
        logits = torch.tensor([[2.2, 1.4, 1.1, 0.3, -0.5, -2.0]])
        temps = torch.tensor([1.0])
        vocab = logits.shape[1]
        out = torch.empty(trials, dtype=torch.long)
        for i in range(trials):
            out[i] = self.sampler(
                logits, temps, generator=gen, all_greedy=False, any_greedy=False, **kw
            )[0]
        got = _freq(out.tolist(), vocab)
        want = self.sampler.final_probs(logits, temps, **kw)[0].double()
        assert torch.all((got - want).abs() < tol), (
            f"forward 与 final_probs 不一致\n got={got.tolist()}\nwant={want.tolist()}"
        )

    def test_matches_forward_with_top_k(self):
        """top-k 截断后 forward 的频次仍要等于 final_probs（错位高发区）。"""
        self._mc_compare(
            trials=8000,
            tol=0.03,
            top_ks=torch.tensor([3]),
            use_top_k=True,
            max_top_k=3,
        )

    def test_matches_forward_with_top_p(self):
        self._mc_compare(
            trials=8000,
            tol=0.03,
            top_ps=torch.tensor([0.75]),
            use_top_p=True,
        )

    def test_matches_forward_with_top_k_and_top_p(self):
        """两者同时开启：final_probs 的掩码必须与 forward 的"先 k 后 p"等价。"""
        self._mc_compare(
            trials=8000,
            tol=0.03,
            top_ks=torch.tensor([4]),
            top_ps=torch.tensor([0.85]),
            use_top_k=True,
            use_top_p=True,
            max_top_k=4,
        )

    def test_matches_forward_with_min_p(self):
        self._mc_compare(
            trials=8000,
            tol=0.03,
            min_ps=torch.tensor([0.3]),
            use_min_p=True,
        )


# ---------------------------------------------------------------- #
# 2. accept_rejection：输出分布必须等于目标分布
# ---------------------------------------------------------------- #
class TestAcceptRejection:
    def test_length_invariant_and_full_accept(self):
        gen = torch.Generator().manual_seed(1)
        # p 在草稿 token 上接近 1 → 几乎所有试验都全接受（k=1 → 2 个 token）
        probs = torch.tensor([[0.99, 0.005, 0.005], [0.99, 0.005, 0.005]])
        lens = set()
        for _ in range(200):
            toks, acc = accept_rejection([0], probs, None, generator=gen)
            assert len(toks) == acc + 1
            lens.add(acc)
        assert 1 in lens  # 至少出现过全接受

    def test_always_reject_when_draft_unlikely(self):
        gen = torch.Generator().manual_seed(2)
        probs = torch.tensor([[0.0, 0.5, 0.5], [0.0, 0.5, 0.5]])
        probs[0, 0] = 1e-9  # 草稿 token 0 几乎不可能
        probs[0] /= probs[0].sum()
        accs = set()
        for _ in range(50):
            _, acc = accept_rejection([0], probs, None, generator=gen)
            accs.add(acc)
        assert accs == {0}, "draft with ~0 target prob must never be accepted"

    def test_distribution_equivalence_one_hot_draft(self):
        """q = one-hot(p0)，k=1：输出 token 的频次必须等于 p。"""
        gen = torch.Generator().manual_seed(3)
        p = torch.tensor([0.2, 0.5, 0.3])
        probs = torch.stack([p, p])  # [k+1=2, vocab]
        emitted = []
        for _ in range(_TRIALS):
            toks, acc = accept_rejection([0], probs, None, generator=gen)
            emitted.append(toks[0])  # 第一个位置就是第一位输出
        got = _freq(emitted, 3)
        assert torch.all((got - p.double()).abs() < _TOL), (
            f"emitted={got.tolist()} target={p.tolist()}"
        )

    def test_distribution_equivalence_multistep(self):
        """k=2：第一个输出位置的边际分布仍必须是 p（对后续位置做同样检验）。"""
        gen = torch.Generator().manual_seed(4)
        p = torch.tensor([0.1, 0.6, 0.3])
        probs = torch.stack([p, p, p])
        first = []
        for _ in range(_TRIALS):
            toks, _ = accept_rejection([0, 0], probs, None, generator=gen)
            first.append(toks[0])
        got = _freq(first, 3)
        assert torch.all((got - p.double()).abs() < _TOL), (
            f"first={got.tolist()} target={p.tolist()}"
        )

    def test_distribution_equivalence_general_draft(self):
        """q 非 one-hot（Phase 3 的 DSpark 草稿）：x 必须**从 q 采样**才成立。

        注意：算法的正确性前提是 ``x_i ~ q_i``。若把 x 固定成某个 token（不是
        从 q 抽），输出分布就不再等于 p——这是拒绝采样最容易被误用的地方，
        所以测试里必须真的按 q 抽样。
        """
        gen = torch.Generator().manual_seed(5)
        p = torch.tensor([0.2, 0.5, 0.3])
        q = torch.tensor([0.5, 0.3, 0.2])
        probs = torch.stack([p, p])
        emitted = []
        for _ in range(_TRIALS):
            x = int(torch.multinomial(q, 1, generator=gen))
            toks, acc = accept_rejection([x], probs, q.unsqueeze(0), generator=gen)
            assert len(toks) == acc + 1
            emitted.append(toks[0])
        got = _freq(emitted, 3)
        assert torch.all((got - p.double()).abs() < _TOL), (
            f"emitted={got.tolist()} target={p.tolist()}"
        )

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="len\\(drafts\\)\\+1"):
            accept_rejection([0, 1], torch.rand(2, 4), None)
