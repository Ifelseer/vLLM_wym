#!/usr/bin/env python3
"""Generate IEEE-style figures for docs/parallel-benchmark.md.

All numeric values are copied from that document (single-run measurements).
Do not round or recompute published table entries.

Usage (from anywhere; matplotlib required):
    python3 docs/figures/gen_ieee_figs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

FIGDIR = Path(__file__).resolve().parent

# Okabe–Ito (colorblind-safe) + hatches for grayscale print.
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
HATCHES = ["", "///", "xx", "..", "OO", "\\\\"]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "STIXGeneral"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.1,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def style_axes(ax, ylabel):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, color="#cccccc")
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel)


def grouped_bar(ax, categories, series, values, bar_width=0.78, annotate=False, fmt="{:.1f}"):
    x = np.arange(len(categories), dtype=float)
    n = len(series)
    w = bar_width / n
    bars_out = []
    for i, s in enumerate(series):
        vals = [float(values.get((c, s), 0.0)) for c in categories]
        bars = ax.bar(
            x + (i - (n - 1) / 2.0) * w,
            vals,
            w * 0.92,
            label=s,
            color=COLORS[i % len(COLORS)],
            hatch=HATCHES[i % len(HATCHES)],
            edgecolor="black",
            linewidth=0.4,
        )
        bars_out.append(bars)
        if annotate:
            for rect, v in zip(bars, vals):
                if v <= 0:
                    continue
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    v,
                    fmt.format(v),
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=0,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(list(categories))
    ax.margins(x=0.06)
    return bars_out


def save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(str(FIGDIR / ("{}.{}".format(name, ext))))
    plt.close(fig)
    print("  -> {}".format(name))


def panel_label(ax, text):
    ax.set_title(text, loc="left", fontsize=9, pad=4)


# ---------------------------------------------------------------------------
# Fig. 1  Hardware topology
# ---------------------------------------------------------------------------
def fig01_hw():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7))

    def gpu_box(ax, x, y, w, h, title, sub, fc):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                facecolor=fc,
                edgecolor="black",
                linewidth=0.7,
            )
        )
        ax.text(x + w / 2.0, y + h * 0.62, title, ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(x + w / 2.0, y + h * 0.28, sub, ha="center", va="center", fontsize=7, color="#333333")

    def numa(ax, x, y, w, h, label):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.03",
                facecolor="#f4f4f4",
                edgecolor="#888888",
                linewidth=0.6,
                linestyle="--",
            )
        )
        ax.text(x + 0.12, y + h - 0.14, label, ha="left", va="top", fontsize=7, color="#555555")

    def hlink(ax, x0, x1, y, text, color):
        ax.annotate(
            "",
            xy=(x1, y),
            xytext=(x0, y),
            arrowprops=dict(arrowstyle="<->", color=color, lw=1.15),
        )
        ax.text((x0 + x1) / 2.0, y + 0.28, text, ha="center", va="bottom", fontsize=7, color=color)

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    panel_label(ax, "(a) Environment A — heterogeneous 2-GPU")
    numa(ax, 0.30, 1.05, 4.20, 3.45, "NUMA 0")
    numa(ax, 5.50, 1.05, 4.20, 3.45, "NUMA 1")
    gpu_box(ax, 0.70, 1.55, 3.40, 1.95, "RTX 4090 D", "24 GB", "#cfe8ff")
    gpu_box(ax, 5.90, 1.55, 3.40, 1.95, "RTX 4070 SUPER", "12 GB", "#ffe6c7")
    hlink(ax, 4.10, 5.90, 2.50, "SYS / no P2P", "#D55E00")
    ax.text(5.0, 0.28, "Host-memory bounce; no NVLink / P2P = NS", ha="center", fontsize=6.5, color="#555555")

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    panel_label(ax, "(b) Environment B — 4x RTX 3090")
    numa(ax, 0.20, 1.05, 4.55, 3.45, "NUMA 0")
    numa(ax, 5.25, 1.05, 4.55, 3.45, "NUMA 1")
    gpu_box(ax, 0.40, 1.55, 1.85, 1.95, "3090", "phys. GPU 1", "#cfe8ff")
    gpu_box(ax, 2.70, 1.55, 1.85, 1.95, "3090", "phys. GPU 2", "#cfe8ff")
    gpu_box(ax, 5.45, 1.55, 1.85, 1.95, "3090", "phys. GPU 4", "#cfe8ff")
    gpu_box(ax, 7.75, 1.55, 1.85, 1.95, "3090", "phys. GPU 5", "#cfe8ff")
    hlink(ax, 2.25, 2.70, 3.70, "PIX", "#0072B2")
    hlink(ax, 7.30, 7.75, 3.70, "PIX", "#0072B2")
    hlink(ax, 4.55, 5.45, 2.50, "SYS / no P2P", "#D55E00")
    ax.text(5.0, 0.28, "NCCL world=4: 2 KB ~82 us, 8 KB ~76 us; no NVLink", ha="center", fontsize=6.5, color="#555555")

    fig.tight_layout()
    save(fig, "fig01_hw")


# ---------------------------------------------------------------------------
# Fig. 2  Test coverage
# ---------------------------------------------------------------------------
def fig02_coverage():
    # 0 = not tested, 1 = Env A, 2 = Env B, 3 = both, 4 = OOM/skip
    rows = [
        "Qwen3-0.6B",
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3-14B",
        "Qwen3-32B",
    ]
    cols = ["Single", "TP=2", "TP=4", "PP=2", "DP", "vLLM", "compile"]
    # Encoded from the original test-plan narrative (not new experiments).
    mat = np.array(
        [
            [3, 1, 2, 1, 0, 1, 1],  # 0.6B
            [0, 0, 0, 0, 1, 0, 1],  # 4B DP only
            [3, 3, 2, 1, 2, 1, 1],  # 8B
            [0, 0, 0, 1, 0, 0, 1],  # 14B PP
            [4, 4, 2, 0, 0, 0, 0],  # 32B
        ],
        dtype=float,
    )
    cmap = matplotlib.colors.ListedColormap(
        ["#f0f0f0", "#0072B2", "#E69F00", "#009E73", "#D55E00"]
    )
    fig, ax = plt.subplots(figsize=(7.16, 2.7))
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=4, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(rows)
    labels = {0: "—", 1: "A", 2: "B", 3: "A+B", 4: "OOM"}
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, labels[int(mat[i, j])], ha="center", va="center", fontsize=8, color="white" if mat[i, j] > 0 else "#666666")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(length=0)
    handles = [
        mpatches.Patch(facecolor="#0072B2", edgecolor="black", linewidth=0.3, label="Env. A only"),
        mpatches.Patch(facecolor="#E69F00", edgecolor="black", linewidth=0.3, label="Env. B only"),
        mpatches.Patch(facecolor="#009E73", edgecolor="black", linewidth=0.3, label="Both"),
        mpatches.Patch(facecolor="#D55E00", edgecolor="black", linewidth=0.3, label="Attempted, OOM"),
        mpatches.Patch(facecolor="#f0f0f0", edgecolor="black", linewidth=0.3, label="Not in matrix"),
    ]
    ax.legend(handles=handles, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=7.5)
    save(fig, "fig02_coverage")


# ---------------------------------------------------------------------------
# Fig. 3  Env A 0.6B  (decode + TTFT + batch8)
# ---------------------------------------------------------------------------
def fig03_envA_06b():
    layouts = ["Single", "TP=2", "PP=2"]
    modes = ["Eager", "CUDA Graph", "Compile", "CG+Compile"]
    decode = {
        ("Single", "Eager"): 65.3,
        ("Single", "CUDA Graph"): 433.9,
        ("Single", "Compile"): 63.9,
        ("Single", "CG+Compile"): 432.2,
        ("TP=2", "Eager"): 42.5,
        ("TP=2", "CUDA Graph"): 354.4,
        ("TP=2", "Compile"): 39.5,
        ("TP=2", "CG+Compile"): 348.7,
        ("PP=2", "Eager"): 90.5,
        ("PP=2", "CUDA Graph"): 257.9,
        ("PP=2", "Compile"): 92.9,
        ("PP=2", "CG+Compile"): 236.2,
    }
    ttft = {
        ("Single", "Eager"): 19.43,
        ("Single", "CUDA Graph"): 18.70,
        ("Single", "Compile"): 4.78,
        ("Single", "CG+Compile"): 4.53,
        ("TP=2", "Eager"): 25.74,
        ("TP=2", "CUDA Graph"): 25.74,
        ("TP=2", "Compile"): 29.43,
        ("TP=2", "CG+Compile"): 27.96,
        ("PP=2", "Eager"): 15.55,
        ("PP=2", "CUDA Graph"): 27.15,
        ("PP=2", "Compile"): 13.06,
        ("PP=2", "CG+Compile"): 29.78,
    }
    batch = {
        ("Single", "Eager"): 487.3,
        ("Single", "CUDA Graph"): 2734.4,
        ("Single", "Compile"): 484.1,
        ("Single", "CG+Compile"): 2664.6,
        ("TP=2", "Eager"): 324.8,
        ("TP=2", "CUDA Graph"): 2156.9,
        ("TP=2", "Compile"): 331.4,
        ("TP=2", "CG+Compile"): 2041.1,
        ("PP=2", "Eager"): 664.8,
        ("PP=2", "CUDA Graph"): 1540.5,
        ("PP=2", "Compile"): 662.9,
        ("PP=2", "CG+Compile"): 1477.6,
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45))
    grouped_bar(axes[0], layouts, modes, decode)
    style_axes(axes[0], "Decode (tok/s)")
    panel_label(axes[0], "(a) Decode")
    grouped_bar(axes[1], layouts, modes, batch)
    style_axes(axes[1], "Batch-8 (tok/s)")
    panel_label(axes[1], "(b) Batch-8")
    grouped_bar(axes[2], layouts, modes, ttft)
    style_axes(axes[2], "TTFT (ms)")
    panel_label(axes[2], "(c) TTFT")
    axes[0].legend(frameon=False, loc="upper right", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig03_envA_06b")


# ---------------------------------------------------------------------------
# Fig. 4  Env A 8B
# ---------------------------------------------------------------------------
def fig04_envA_8b():
    layouts = ["Single", "TP=2", "PP=2"]
    modes = ["Eager", "CUDA Graph", "Compile", "CG+Compile"]
    decode = {
        ("Single", "Eager"): 38.4,
        ("Single", "CUDA Graph"): 57.6,
        ("Single", "Compile"): 50.3,
        ("Single", "CG+Compile"): 58.0,
        ("TP=2", "Eager"): 31.3,
        ("TP=2", "CUDA Graph"): 51.3,
        ("TP=2", "Compile"): 33.7,
        ("TP=2", "CG+Compile"): 51.0,
        ("PP=2", "Eager"): 23.1,
        ("PP=2", "CUDA Graph"): 35.2,
        ("PP=2", "Compile"): 20.9,
        ("PP=2", "CG+Compile"): 35.4,
    }
    ttft = {
        ("Single", "Eager"): 39.99,
        ("Single", "CUDA Graph"): 40.82,
        ("Single", "Compile"): 35.47,
        ("Single", "CG+Compile"): 35.47,
        ("TP=2", "Eager"): 47.88,
        ("TP=2", "CUDA Graph"): 72.84,
        ("TP=2", "Compile"): 47.33,
        ("TP=2", "CG+Compile"): 62.48,
        ("PP=2", "Eager"): 59.08,
        ("PP=2", "CUDA Graph"): 75.63,
        ("PP=2", "Compile"): 53.43,
        ("PP=2", "CG+Compile"): 66.11,
    }
    batch = {
        ("Single", "Eager"): 363.0,
        ("Single", "CUDA Graph"): 400.2,
        ("Single", "Compile"): 354.9,
        ("Single", "CG+Compile"): 399.5,
        ("TP=2", "Eager"): 229.9,
        ("TP=2", "CUDA Graph"): 334.3,
        ("TP=2", "Compile"): 240.1,
        ("TP=2", "CG+Compile"): 333.3,
        ("PP=2", "Eager"): 165.6,
        ("PP=2", "CUDA Graph"): 247.3,
        ("PP=2", "Compile"): 152.4,
        ("PP=2", "CG+Compile"): 247.5,
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45))
    grouped_bar(axes[0], layouts, modes, decode)
    style_axes(axes[0], "Decode (tok/s)")
    panel_label(axes[0], "(a) Decode")
    grouped_bar(axes[1], layouts, modes, batch)
    style_axes(axes[1], "Batch-8 (tok/s)")
    panel_label(axes[1], "(b) Batch-8")
    grouped_bar(axes[2], layouts, modes, ttft)
    style_axes(axes[2], "TTFT (ms)")
    panel_label(axes[2], "(c) TTFT")
    axes[1].legend(frameon=False, loc="upper right", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig04_envA_8b")


# ---------------------------------------------------------------------------
# Fig. 5  Optimization vs model scale
# ---------------------------------------------------------------------------
def fig05_opt_scale():
    models = ["0.6B\n(single)", "8B\n(single)", "14B\n(PP=2)"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))

    cg_eager = [65.3, 38.4, 20.2]
    cg_cg = [433.9, 57.6, 24.1]
    x = np.arange(3)
    w = 0.36
    axes[0].bar(x - w / 2, cg_eager, w, label="Eager", color=COLORS[0], hatch=HATCHES[0], edgecolor="black", lw=0.4)
    axes[0].bar(x + w / 2, cg_cg, w, label="CUDA Graph", color=COLORS[1], hatch=HATCHES[1], edgecolor="black", lw=0.4)
    for i, lab in enumerate(["6.6x", "1.5x", "1.2x"]):
        axes[0].text(i + w / 2, cg_cg[i], lab, ha="center", va="bottom", fontsize=7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    style_axes(axes[0], "Decode (tok/s)")
    panel_label(axes[0], "(a) CUDA Graph vs eager decode")
    axes[0].legend(frameon=False)

    ttft_e = [19.43, 39.99, 77.26]
    ttft_c = [4.78, 35.47, 74.41]
    axes[1].bar(x - w / 2, ttft_e, w, label="Eager", color=COLORS[0], hatch=HATCHES[0], edgecolor="black", lw=0.4)
    axes[1].bar(x + w / 2, ttft_c, w, label="Compile", color=COLORS[2], hatch=HATCHES[2], edgecolor="black", lw=0.4)
    for i, lab in enumerate(["4.1x", "1.13x", "1.04x"]):
        axes[1].text(i + w / 2, ttft_c[i], lab, ha="center", va="bottom", fontsize=7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    style_axes(axes[1], "TTFT (ms)")
    panel_label(axes[1], "(b) Compile vs eager TTFT")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig05_opt_scale")


# ---------------------------------------------------------------------------
# Fig. 6  Env A parallel scaling (relative to single GPU, same mode)
# ---------------------------------------------------------------------------
def fig06_scaling():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))
    cats = ["Single", "TP=2", "PP=2"]
    x = np.arange(3)
    w = 0.36

    # 0.6B decode relative to single of the same mode
    eager = np.array([65.3, 42.5, 90.5]) / 65.3
    cg = np.array([433.9, 354.4, 257.9]) / 433.9
    axes[0].bar(x - w / 2, eager, w, label="Eager", color=COLORS[0], hatch=HATCHES[0], edgecolor="black", lw=0.4)
    axes[0].bar(x + w / 2, cg, w, label="CUDA Graph", color=COLORS[1], hatch=HATCHES[1], edgecolor="black", lw=0.4)
    axes[0].axhline(1.0, color="black", lw=0.6, ls=":")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cats)
    style_axes(axes[0], "Relative decode")
    panel_label(axes[0], "(a) Qwen3-0.6B (decode vs single)")
    for i, v in enumerate(eager):
        axes[0].text(i - w / 2, v + 0.03, "{:.2f}x".format(v), ha="center", fontsize=6.5)
    for i, v in enumerate(cg):
        axes[0].text(i + w / 2, v + 0.03, "{:.2f}x".format(v), ha="center", fontsize=6.5)
    axes[0].legend(frameon=False)
    axes[0].set_ylim(0, 1.7)

    # 8B batch8 relative
    eager_b = np.array([363.0, 229.9, 165.6]) / 363.0
    cg_b = np.array([400.2, 334.3, 247.3]) / 400.2
    axes[1].bar(x - w / 2, eager_b, w, label="Eager", color=COLORS[0], hatch=HATCHES[0], edgecolor="black", lw=0.4)
    axes[1].bar(x + w / 2, cg_b, w, label="CUDA Graph", color=COLORS[1], hatch=HATCHES[1], edgecolor="black", lw=0.4)
    axes[1].axhline(1.0, color="black", lw=0.6, ls=":")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cats)
    style_axes(axes[1], "Relative batch-8")
    panel_label(axes[1], "(b) Qwen3-8B (batch-8 vs single)")
    for i, v in enumerate(eager_b):
        axes[1].text(i - w / 2, v + 0.03, "{:.2f}x".format(v), ha="center", fontsize=6.5)
    for i, v in enumerate(cg_b):
        axes[1].text(i + w / 2, v + 0.03, "{:.2f}x".format(v), ha="center", fontsize=6.5)
    axes[1].legend(frameon=False)
    axes[1].set_ylim(0, 1.35)
    fig.tight_layout()
    save(fig, "fig06_scaling")


# ---------------------------------------------------------------------------
# Fig. 7  14B PP + 4B DP
# ---------------------------------------------------------------------------
def fig07_14b_dp():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))
    modes = ["Eager", "CUDA Graph", "Compile", "CG+Compile"]
    x = np.arange(4)
    decode = [20.2, 24.1, 20.6, 24.0]
    ttft = [77.26, 76.72, 74.41, 75.21]
    w = 0.36
    axes[0].bar(x - w / 2, decode, w, label="Decode (tok/s)", color=COLORS[0], hatch=HATCHES[1], edgecolor="black", lw=0.4)
    ax0b = axes[0].twinx()
    ax0b.bar(x + w / 2, ttft, w, label="TTFT (ms)", color=COLORS[2], hatch=HATCHES[2], edgecolor="black", lw=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(modes, rotation=15, ha="right")
    style_axes(axes[0], "Decode (tok/s)")
    ax0b.set_ylabel("TTFT (ms)")
    ax0b.spines["top"].set_visible(False)
    panel_label(axes[0], "(a) Qwen3-14B, PP=2 (12+28 layers)")
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = ax0b.get_legend_handles_labels()
    axes[0].legend(h1 + h2, l1 + l2, frameon=False, fontsize=7)

    cats = ["DP=1 (4090)", "DP=2 (4090+4070)"]
    series = ["Eager", "Compile", "CUDA Graph"]
    values = {
        ("DP=1 (4090)", "Eager"): 1432.8,
        ("DP=1 (4090)", "Compile"): 1403.6,
        ("DP=1 (4090)", "CUDA Graph"): 2302.5,
        ("DP=2 (4090+4070)", "Eager"): 1146.7,
        ("DP=2 (4090+4070)", "Compile"): 1100.1,
        ("DP=2 (4090+4070)", "CUDA Graph"): 0.0,
    }
    grouped_bar(axes[1], cats, series, values)
    style_axes(axes[1], "32-way aggregate (tok/s)")
    panel_label(axes[1], "(b) Qwen3-4B data parallel")
    axes[1].text(1.0, 80, "CG: OOM\non 12 GB", ha="center", va="bottom", fontsize=7, color="#D55E00")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig07_14b_dp")


# ---------------------------------------------------------------------------
# Fig. 8  This engine vs vLLM
# ---------------------------------------------------------------------------
def fig08_vs_vllm():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55))
    cats = ["Single\nEager", "Single\nCG", "Single\nCompile", "TP=2\nEager", "TP=2\nCG"]
    series = ["Ours", "vLLM"]
    v06 = {
        ("Single\nEager", "Ours"): 65.3,
        ("Single\nEager", "vLLM"): 60.9,
        ("Single\nCG", "Ours"): 433.9,
        ("Single\nCG", "vLLM"): 231.8,
        ("Single\nCompile", "Ours"): 63.9,
        ("Single\nCompile", "vLLM"): 61.8,
        ("TP=2\nEager", "Ours"): 42.5,
        ("TP=2\nEager", "vLLM"): 45.1,
        ("TP=2\nCG", "Ours"): 354.4,
        ("TP=2\nCG", "vLLM"): 205.9,
    }
    grouped_bar(axes[0], cats, series, v06)
    style_axes(axes[0], "Decode (tok/s)")
    panel_label(axes[0], "(a) Qwen3-0.6B")
    axes[0].legend(frameon=False)

    cats8 = ["Single\nEager", "Single\nCG", "TP=2\nEager", "TP=2\nCG"]
    v08 = {
        ("Single\nEager", "Ours"): 38.4,
        ("Single\nEager", "vLLM"): 35.3,
        ("Single\nCG", "Ours"): 57.6,
        ("Single\nCG", "vLLM"): 59.4,
        ("TP=2\nEager", "Ours"): 31.3,
        ("TP=2\nEager", "vLLM"): 32.0,
        ("TP=2\nCG", "Ours"): 51.3,
        ("TP=2\nCG", "vLLM"): 57.7,
    }
    grouped_bar(axes[1], cats8, series, v08)
    style_axes(axes[1], "Decode (tok/s)")
    panel_label(axes[1], "(b) Qwen3-8B")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig08_vs_vllm")


# ---------------------------------------------------------------------------
# Fig. 9  Env B tensor parallel
# ---------------------------------------------------------------------------
def fig09_envB_tp():
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))
    series = ["Eager", "CUDA Graph"]

    cats06 = ["Single", "TP=4"]
    d06 = {
        ("Single", "Eager"): 48.2,
        ("Single", "CUDA Graph"): 369.2,
        ("TP=4", "Eager"): 35.2,
        ("TP=4", "CUDA Graph"): 257.3,
    }
    grouped_bar(axes[0], cats06, series, d06)
    style_axes(axes[0], "Decode (tok/s)")
    panel_label(axes[0], "(a) 0.6B decode")
    axes[0].legend(frameon=False, fontsize=6.5)

    cats8 = ["Single", "TP=2", "TP=4"]
    d8 = {
        ("Single", "Eager"): 36.6,
        ("Single", "CUDA Graph"): 50.2,
        ("TP=2", "Eager"): 27.6,
        ("TP=2", "CUDA Graph"): 73.5,
        ("TP=4", "Eager"): 27.1,
        ("TP=4", "CUDA Graph"): 105.4,
    }
    grouped_bar(axes[1], cats8, series, d8)
    style_axes(axes[1], "Decode (tok/s)")
    panel_label(axes[1], "(b) 8B decode")
    axes[1].legend(frameon=False, fontsize=6.5)

    b8 = {
        ("Single", "Eager"): 243.2,
        ("Single", "CUDA Graph"): 333.2,
        ("TP=2", "Eager"): 187.7,
        ("TP=2", "CUDA Graph"): 384.1,
        ("TP=4", "Eager"): 182.7,
        ("TP=4", "CUDA Graph"): 451.0,
    }
    grouped_bar(axes[2], cats8, series, b8)
    style_axes(axes[2], "Batch-8 (tok/s)")
    panel_label(axes[2], "(c) 8B batch-8")
    axes[2].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig09_envB_tp")


# ---------------------------------------------------------------------------
# Fig. 10  Env B 8B four-GPU packing (DP vs hybrid)
# ---------------------------------------------------------------------------
def fig10_envB_dp():
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    cats = ["DP=4", "DP=2 x TP=2", "DP=2 x PP=2"]
    series = ["Eager", "CUDA Graph"]
    values = {
        ("DP=4", "Eager"): 864.8,
        ("DP=4", "CUDA Graph"): 1166.3,
        ("DP=2 x TP=2", "Eager"): 668.6,
        ("DP=2 x TP=2", "CUDA Graph"): 1088.5,
        ("DP=2 x PP=2", "Eager"): 638.3,
        ("DP=2 x PP=2", "CUDA Graph"): 994.9,
    }
    grouped_bar(ax, cats, series, values, annotate=True, fmt="{:.1f}")
    style_axes(ax, "32-way aggregate (tok/s)")
    ax.legend(frameon=False)
    ax.set_xticklabels(["DP=4", "DP=2×TP=2", "DP=2×PP=2"])
    save(fig, "fig10_envB_dp")


# ---------------------------------------------------------------------------
# Fig. 11  32B: 4x3090 TP=4 vs A100 single
# ---------------------------------------------------------------------------
def fig11_32b():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))
    cats = ["4x3090\nTP=4", "A100 80GB\nsingle"]
    series = ["Eager", "CUDA Graph"]
    dec = {
        ("4x3090\nTP=4", "Eager"): 12.1,
        ("4x3090\nTP=4", "CUDA Graph"): 34.8,
        ("A100 80GB\nsingle", "Eager"): 20.4,
        ("A100 80GB\nsingle", "CUDA Graph"): 23.2,
    }
    ttft = {
        ("4x3090\nTP=4", "Eager"): 201.11,
        ("4x3090\nTP=4", "CUDA Graph"): 214.50,
        ("A100 80GB\nsingle", "Eager"): 105.58,
        ("A100 80GB\nsingle", "CUDA Graph"): 107.96,
    }
    grouped_bar(axes[0], cats, series, dec, annotate=True, fmt="{:.1f}")
    style_axes(axes[0], "Decode (tok/s)")
    panel_label(axes[0], "(a) Decode")
    axes[0].legend(frameon=False)
    grouped_bar(axes[1], cats, series, ttft, annotate=True, fmt="{:.0f}")
    style_axes(axes[1], "TTFT (ms)")
    panel_label(axes[1], "(b) TTFT")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig11_32b")


# ---------------------------------------------------------------------------
# Fig. 12  Cross-environment 8B single-GPU
# ---------------------------------------------------------------------------
def fig12_cross():
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45))
    cats = ["Env. A\n4090 D", "Env. B\n3090"]
    series = ["Eager", "CUDA Graph"]
    ttft = {
        ("Env. A\n4090 D", "Eager"): 39.99,
        ("Env. A\n4090 D", "CUDA Graph"): 40.82,
        ("Env. B\n3090", "Eager"): 67.50,
        ("Env. B\n3090", "CUDA Graph"): 69.01,
    }
    dec = {
        ("Env. A\n4090 D", "Eager"): 38.4,
        ("Env. A\n4090 D", "CUDA Graph"): 57.6,
        ("Env. B\n3090", "Eager"): 36.6,
        ("Env. B\n3090", "CUDA Graph"): 50.2,
    }
    batch = {
        ("Env. A\n4090 D", "Eager"): 363.0,
        ("Env. A\n4090 D", "CUDA Graph"): 400.2,
        ("Env. B\n3090", "Eager"): 243.2,
        ("Env. B\n3090", "CUDA Graph"): 333.2,
    }
    grouped_bar(axes[0], cats, series, dec, annotate=True, fmt="{:.1f}")
    style_axes(axes[0], "Decode (tok/s)")
    panel_label(axes[0], "(a) Decode")
    grouped_bar(axes[1], cats, series, batch, annotate=True, fmt="{:.0f}")
    style_axes(axes[1], "Batch-8 (tok/s)")
    panel_label(axes[1], "(b) Batch-8")
    grouped_bar(axes[2], cats, series, ttft, annotate=True, fmt="{:.1f}")
    style_axes(axes[2], "TTFT (ms)")
    panel_label(axes[2], "(c) TTFT")
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig12_cross_8b")


def main():
    fig01_hw()
    fig02_coverage()
    fig03_envA_06b()
    fig04_envA_8b()
    fig05_opt_scale()
    fig06_scaling()
    fig07_14b_dp()
    fig08_vs_vllm()
    fig09_envB_tp()
    fig10_envB_dp()
    fig11_32b()
    fig12_cross()
    print("done ->", FIGDIR)


if __name__ == "__main__":
    main()
