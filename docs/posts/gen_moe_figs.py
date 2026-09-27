"""生成 MoE 文档里的实现示意图。

用法（在项目根目录下）：
    python docs/posts/gen_moe_figs.py

输出到 docs/posts/：
    dense_vs_sparse.png
    naive_vs_grouped.png
    align_blocks.png
    tp_vs_ep.png
    shared_ar.png
    w13_layout.png
    moe_pipeline.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent

_FONT = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
fm.fontManager.addfont(_FONT)
plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False

C_MAIN = "#2563eb"
C_HL = "#f59e0b"
C_DIM = "#9ca3af"
C_BG = "#f8fafc"
C_OK = "#059669"
C_RED = "#dc2626"
C_PURPLE = "#7c3aed"


def _box(ax, x, y, w, h, text, fc=C_BG, ec=C_MAIN, fs=10, tc="black", lw=1.4, rounded=0.04):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={rounded}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(box)
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            color=tc,
            zorder=3,
        )


def _arrow(ax, p0, p1, color=C_MAIN, lw=1.6, style="-|>"):
    ar = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=style,
        mutation_scale=13,
        linewidth=lw,
        color=color,
        zorder=4,
        shrinkA=1.0,
        shrinkB=1.0,
    )
    ax.add_patch(ar)


def draw_dense_vs_sparse() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 5.8), dpi=160)
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    ax.text(2.6, 5.45, "稠密 MLP", ha="center", fontsize=13, color="#0f172a")
    ax.text(8.5, 5.45, "稀疏 MoE（top-2 / 8）", ha="center", fontsize=13, color="#0f172a")

    _box(ax, 1.55, 4.55, 2.1, 0.7, "token 隐藏向量", fc="#e0f2fe")
    _arrow(ax, (2.6, 4.55), (2.6, 3.95))
    _box(ax, 1.35, 2.55, 2.5, 1.35, "唯一的 FFN\ngate / up / down\n每个 token 都走它", fc="#dbeafe")
    _arrow(ax, (2.6, 2.55), (2.6, 1.95))
    _box(ax, 1.55, 1.15, 2.1, 0.7, "输出 hidden", fc="#e0f2fe")
    ax.text(2.6, 0.35, "参数少，每次全算", ha="center", fontsize=9.5, color="#64748b")

    _box(ax, 7.45, 4.55, 2.1, 0.7, "token 隐藏向量", fc="#e0f2fe")
    _arrow(ax, (8.5, 4.55), (8.5, 4.15))
    _box(ax, 7.55, 3.45, 1.9, 0.65, "router / gate", fc="#fef3c7", ec=C_HL)

    experts = ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]
    for i, name in enumerate(experts):
        x = 5.55 + (i % 4) * 1.45
        y = 2.35 if i < 4 else 1.45
        active = i in (1, 6)
        _box(
            ax,
            x,
            y,
            1.3,
            0.72,
            name,
            fc="#fde68a" if active else "#f1f5f9",
            ec=C_HL if active else C_DIM,
            fs=10,
        )
    _arrow(ax, (8.5, 3.45), (8.5, 3.15), color=C_HL)
    ax.annotate(
        "",
        xy=(6.2, 3.07),
        xytext=(8.5, 3.45),
        arrowprops=dict(arrowstyle="-|>", color=C_HL, lw=1.2),
    )
    ax.annotate(
        "",
        xy=(10.55, 2.17),
        xytext=(8.5, 3.45),
        arrowprops=dict(arrowstyle="-|>", color=C_HL, lw=1.2),
    )
    _box(ax, 7.35, 0.45, 2.3, 0.7, "加权求和", fc="#dcfce7", ec=C_OK)
    _arrow(ax, (8.2, 1.45), (8.2, 1.2), color=C_OK)
    _arrow(ax, (10.2, 1.45), (8.7, 1.15), color=C_OK)
    ax.text(8.5, 0.12, "参数多，每次只算选中的那几个", ha="center", fontsize=9.5, color="#64748b")

    ax.plot([5.15, 5.15], [0.2, 5.5], color="#e2e8f0", lw=1.0)
    fig.savefig(OUT / "dense_vs_sparse.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT / "dense_vs_sparse.png")


def draw_naive_vs_grouped() -> None:
    fig, ax = plt.subplots(figsize=(11.6, 5.6), dpi=160)
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    ax.text(2.7, 5.25, "朴素：按专家循环", ha="center", fontsize=13)
    ax.text(8.7, 5.25, "融合：先重排再 grouped GEMM", ha="center", fontsize=13)

    tokens = [("t0", 1), ("t1", 0), ("t2", 1), ("t3", 2)]
    for i, (name, e) in enumerate(tokens):
        _box(ax, 0.25, 4.2 - i * 0.85, 1.15, 0.65, f"{name}\n→E{e}", fc="#e0f2fe", fs=9)

    for i, e in enumerate([0, 1, 2]):
        y = 3.7 - i * 1.15
        _box(ax, 2.0, y, 2.35, 0.95, f"for expert {e}:\ngather → GEMM → scatter", fc="#fee2e2", ec=C_RED, fs=8.5)
    ax.text(3.15, 0.25, "三次小核，启动开销大", ha="center", fontsize=9.5, color=C_RED)

    order = [("t1", 0), ("t0", 1), ("t2", 1), ("t3", 2)]
    for i, (name, e) in enumerate(order):
        _box(
            ax,
            6.15,
            4.15 - i * 0.72,
            1.35,
            0.6,
            f"{name}  E{e}",
            fc="#fef3c7" if e == 1 else "#e0f2fe",
            ec=C_HL if e == 1 else C_MAIN,
            fs=9,
        )
    _box(ax, 8.0, 2.55, 3.15, 1.7, "一次 grouped GEMM\n同一专家的行挨在一起\n块大小对齐到 BLOCK_M", fc="#dbeafe", fs=10)
    _arrow(ax, (7.55, 3.1), (8.0, 3.3))
    ax.text(9.55, 0.25, "算力走矩阵乘，不再走 Python 循环", ha="center", fontsize=9.5, color=C_MAIN)

    ax.plot([5.55, 5.55], [0.15, 5.35], color="#e2e8f0", lw=1.0)
    fig.savefig(OUT / "naive_vs_grouped.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT / "naive_vs_grouped.png")


def draw_align_blocks() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 5.2), dpi=160)
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(5.7, 4.85, "moe_align_block_size：按专家打包，块长对齐到 BLOCK_M=4", ha="center", fontsize=12)

    raw = [1, 0, 1, 2, 1, 0]
    ax.text(0.3, 4.25, "topk_ids（展平）", fontsize=10, color="#334155")
    for i, e in enumerate(raw):
        _box(ax, 0.3 + i * 0.85, 3.35, 0.75, 0.7, f"t{i}\nE{e}", fc="#e0f2fe", fs=8.5)

    packed = [
        ("t1", 0, False),
        ("t5", 0, False),
        ("pad", 0, True),
        ("pad", 0, True),
        ("t0", 1, False),
        ("t2", 1, False),
        ("t4", 1, False),
        ("pad", 1, True),
        ("t3", 2, False),
        ("pad", 2, True),
        ("pad", 2, True),
        ("pad", 2, True),
    ]
    ax.text(0.3, 2.85, "sorted_ids（每个专家一段，不足补 pad）", fontsize=10, color="#334155")
    colors = {0: "#dbeafe", 1: "#fef3c7", 2: "#dcfce7"}
    ecs = {0: C_MAIN, 1: C_HL, 2: C_OK}
    for i, (name, e, pad) in enumerate(packed):
        x = 0.3 + (i % 12) * 0.9
        fc = "#f1f5f9" if pad else colors[e]
        ec = C_DIM if pad else ecs[e]
        label = "pad" if pad else f"{name}"
        _box(ax, x, 1.95, 0.8, 0.65, label, fc=fc, ec=ec, fs=8.5)

    ax.text(0.3, 1.45, "expert_ids（每个 BLOCK 一个专家号）", fontsize=10, color="#334155")
    for i, (lab, e) in enumerate([("blk0", 0), ("blk1", 1), ("blk2", 2)]):
        _box(ax, 0.3 + i * 2.6, 0.45, 2.3, 0.8, f"{lab}  →  expert {e}", fc=colors[e], ec=ecs[e], fs=10)

    fig.savefig(OUT / "align_blocks.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT / "align_blocks.png")


def draw_tp_vs_ep() -> None:
    fig, ax = plt.subplots(figsize=(11.6, 6.0), dpi=160)
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    ax.text(2.85, 5.6, "MoE 张量并行（切中间维）", ha="center", fontsize=12)
    ax.text(8.7, 5.6, "MoE 专家并行（切专家）", ha="center", fontsize=12)

    def gpu(x, y, title, body, fc="#eff6ff"):
        _box(ax, x, y, 4.7, 2.15, "", fc=fc, ec=C_MAIN, lw=1.6)
        ax.text(x + 0.18, y + 1.78, title, fontsize=10.5, color=C_MAIN, va="center")
        ax.text(x + 2.35, y + 0.85, body, ha="center", va="center", fontsize=9.5)

    gpu(0.25, 3.15, "GPU 0", "E0 E1 E2 E3\n每张专家只持有 inter/2")
    gpu(0.25, 0.55, "GPU 1", "E0 E1 E2 E3\n另外一半 inter")
    ax.text(2.6, 0.12, "每个 token 两张卡都算，最后 all-reduce", ha="center", fontsize=9, color="#64748b")

    gpu(6.55, 3.15, "GPU 0", "完整的 E0、E1\nremote 的 E2/E3 标成 -1", fc="#fff7ed")
    gpu(6.55, 0.55, "GPU 1", "完整的 E2、E3\nremote 的 E0/E1 标成 -1", fc="#fff7ed")
    ax.text(8.9, 0.12, "token 不搬家，本卡算本地专家，最后 all-reduce", ha="center", fontsize=9, color="#64748b")

    ax.plot([5.7, 5.7], [0.35, 5.7], color="#e2e8f0", lw=1.0)
    fig.savefig(OUT / "tp_vs_ep.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT / "tp_vs_ep.png")


def draw_shared_ar() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 4.8), dpi=160)
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    ax.text(2.7, 4.4, "EP：先对 routed 做 AR，再加共享专家", ha="center", fontsize=11)
    ax.text(8.6, 4.4, "TP：先加共享专家，再 AR", ha="center", fontsize=11)

    steps_ep = ["routed 局部输出", "all-reduce", "+ shared（完整一份）"]
    steps_tp = ["routed 局部输出", "+ shared（各算一片）", "all-reduce"]
    for i, s in enumerate(steps_ep):
        y = 3.35 - i * 1.15
        fc = "#dcfce7" if i == 1 else "#e0f2fe"
        _box(ax, 0.7, y, 4.0, 0.85, s, fc=fc, fs=10)
        if i < 2:
            _arrow(ax, (2.7, y), (2.7, y - 0.28))
    for i, s in enumerate(steps_tp):
        y = 3.35 - i * 1.15
        fc = "#dcfce7" if i == 2 else "#e0f2fe"
        _box(ax, 6.6, y, 4.0, 0.85, s, fc=fc, fs=10)
        if i < 2:
            _arrow(ax, (8.6, y), (8.6, y - 0.28))

    ax.plot([5.6, 5.6], [0.2, 4.55], color="#e2e8f0", lw=1.0)
    fig.savefig(OUT / "shared_ar.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT / "shared_ar.png")


def draw_w13_layout() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 4.6), dpi=160)
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    ax.text(2.7, 4.15, "HuggingFace / 训练侧", ha="center", fontsize=12)
    ax.text(8.6, 4.15, "本实现 / FlashInfer cutlass", ha="center", fontsize=12)

    _box(ax, 0.5, 2.35, 2.1, 1.35, "gate", fc="#dbeafe", fs=12)
    _box(ax, 2.7, 2.35, 2.1, 1.35, "up", fc="#fef3c7", ec=C_HL, fs=12)
    ax.text(2.65, 1.85, "w13 = [gate | up]", ha="center", fontsize=10, color="#334155")
    ax.text(2.65, 1.35, "silu(gate) * up", ha="center", fontsize=10, color="#64748b")

    _box(ax, 6.4, 2.35, 2.1, 1.35, "up", fc="#fef3c7", ec=C_HL, fs=12)
    _box(ax, 8.6, 2.35, 2.1, 1.35, "gate", fc="#dbeafe", fs=12)
    ax.text(8.55, 1.85, "w13 = [up | gate]", ha="center", fontsize=10, color="#334155")
    ax.text(8.55, 1.35, "up * silu(gate)  （mul_and_silu）", ha="center", fontsize=10, color="#64748b")

    _arrow(ax, (4.95, 3.0), (6.3, 3.0), color=C_PURPLE, lw=2.0)
    ax.text(5.6, 3.25, "加载时对调", ha="center", fontsize=9, color=C_PURPLE)

    ax.text(
        5.7,
        0.45,
        "同一套 SwiGLU，只是两半拼在中间维上的顺序不同。加载器里的 _hf_gate_up_to_cutlass 就是做这件事。",
        ha="center",
        fontsize=9.5,
        color="#334155",
    )
    fig.savefig(OUT / "w13_layout.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT / "w13_layout.png")


def draw_moe_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.8, 4.4), dpi=160)
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    nodes = [
        (0.2, "hidden\n[T, H]"),
        (2.15, "gate 线性\n[T, E]"),
        (4.15, "softmax\n+ top-k"),
        (6.15, "expert_map\n全局→本地"),
        (8.25, "fused experts\nalign / GEMM"),
        (10.25, "AR / shared"),
    ]
    for x, text in nodes:
        _box(ax, x, 1.7, 1.7, 1.35, text, fc="#e0f2fe", fs=9)
    for i in range(len(nodes) - 1):
        x0 = nodes[i][0] + 1.7
        x1 = nodes[i + 1][0]
        _arrow(ax, (x0, 2.35), (x1, 2.35))

    ax.text(
        5.9,
        0.85,
        "SparseMoeBlock.forward  →  Experts.forward  →  fused_moe_forward",
        ha="center",
        fontsize=10,
        color="#334155",
    )
    ax.text(
        5.9,
        0.35,
        "Triton 路径：grouped GEMM1 → mul_and_silu → GEMM2（乘 topk 权重）→ moe_sum",
        ha="center",
        fontsize=9.5,
        color="#64748b",
    )
    fig.savefig(OUT / "moe_pipeline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT / "moe_pipeline.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    draw_dense_vs_sparse()
    draw_naive_vs_grouped()
    draw_align_blocks()
    draw_tp_vs_ep()
    draw_shared_ar()
    draw_w13_layout()
    draw_moe_pipeline()
