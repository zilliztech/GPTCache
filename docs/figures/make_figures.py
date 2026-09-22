"""Generate the two write-up figures.

fig_request_flow.png  -> Section 6.1 / 3.1: the GPTCache request path.
fig_crossover.png     -> Section 7.4: cost-weighted advantage over LRU across regimes.

Numbers in the crossover figure are the paired-mean deltas verified via bench_lmsys/paired.py
(lmsys, n=7 stationary; decay-off/on drift at cs100). Regenerate with `python make_figures.py`.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))


def request_flow():
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")

    steps = ["Pre-embed\n(extract key)", "Embed\n(vector)", "Search\n(ANN)",
             "Evaluate\n(threshold)", "Post-process\n(select)"]
    x = 0.2; w = 1.55; gap = 0.35; y = 1.6; h = 0.9
    centers = []
    for s in steps:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                             linewidth=1.2, edgecolor="#333", facecolor="#eef3fb")
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, s, ha="center", va="center", fontsize=8.5)
        centers.append(x + w / 2)
        x += w + gap
    right = x - gap

    # forward arrows between steps
    for i in range(len(steps) - 1):
        a = centers[i] + w / 2
        b = centers[i + 1] - w / 2
        ax.add_patch(FancyArrowPatch((a, y + h / 2), (b, y + h / 2),
                     arrowstyle="-|>", mutation_scale=12, color="#333", lw=1.2))

    # HIT path out of post-process
    ax.add_patch(FancyArrowPatch((right, y + h / 2), (right + 0.9, y + h / 2),
                 arrowstyle="-|>", mutation_scale=12, color="#1a7f37", lw=1.4))
    ax.text(right + 0.95, y + h / 2, "hit:\nreturn\ncached", ha="left", va="center",
            fontsize=8, color="#1a7f37")

    # MISS path: drop down, call LLM, save back
    miss_y = 0.5
    ax.add_patch(FancyArrowPatch((centers[3], y), (centers[3], miss_y + 0.35),
                 arrowstyle="-|>", mutation_scale=12, color="#b3261e", lw=1.4))
    ax.text(centers[3] + 0.1, (y + miss_y) / 2, "miss", ha="left", va="center",
            fontsize=8, color="#b3261e")
    llm = FancyBboxPatch((centers[3] - 0.9, miss_y - 0.05), 1.8, 0.5,
                         boxstyle="round,pad=0.03", linewidth=1.2,
                         edgecolor="#b3261e", facecolor="#fdecea")
    ax.add_patch(llm)
    ax.text(centers[3], miss_y + 0.2, "call LLM", ha="center", va="center", fontsize=8.5)
    # save arrow back to the first store (leftwards)
    ax.add_patch(FancyArrowPatch((centers[3] - 0.9, miss_y + 0.2), (centers[0], miss_y + 0.2),
                 arrowstyle="-|>", mutation_scale=12, color="#b3261e", lw=1.2,
                 linestyle=(0, (4, 2))))
    ax.add_patch(FancyArrowPatch((centers[0], miss_y + 0.2), (centers[0], y),
                 arrowstyle="-|>", mutation_scale=12, color="#b3261e", lw=1.2,
                 linestyle=(0, (4, 2))))
    ax.text((centers[0] + centers[3]) / 2, miss_y + 0.05, "save (question, answer, embedding, cost)",
            ha="center", va="top", fontsize=7.5, color="#b3261e")

    fig.tight_layout()
    p = os.path.join(OUT, "fig_request_flow.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def crossover():
    # paired-mean cost_wt delta over LRU (pp), lmsys, verified via paired.py
    labels = ["Drift\ndecay OFF\n(fast churn)",
              "Drift\ndecay ON\n(fast, sh0.10)",
              "Drift\ndecay ON\n(gentle, sh0.02)",
              "Stationary\nsharp skew\n(z15)",
              "Stationary\nflat skew\n(z11)"]
    vals = [-5.64, 3.63, 4.10, 7.64, 14.42]  # last two = mean over cs25/50/100
    notes = ["ADAPT-LRU", "ADAPT-LRU", "ADAPT-LRU", "CA-LRU (mean cs)", "CA-LRU (mean cs)"]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = ["#b3261e" if v < 0 else "#1a7f37" for v in vals]
    bars = ax.bar(range(len(vals)), vals, color=colors, width=0.6, edgecolor="#222", linewidth=0.6)
    ax.axhline(0, color="#222", lw=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Cost-weighted hit-rate advantage over LRU (pp)", fontsize=9.5)
    ax.set_title("Crossover surface: where value-aware eviction beats LRU", fontsize=11)
    for i, (b, v, n) in enumerate(zip(bars, vals, notes)):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.4 if v >= 0 else -0.9),
                f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=9, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, -0.6 if v >= 0 else 0.4, n,
                ha="center", va="top" if v >= 0 else "bottom", fontsize=6.5, color="#555")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#1a7f37", edgecolor="#222", label="value-aware eviction wins"),
                       Patch(facecolor="#b3261e", edgecolor="#222", label="LRU wins")],
              loc="upper left", fontsize=8.5, frameon=False)
    ax.margins(y=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_crossover.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def gdsf_headtohead():
    # CA - GDSF paired cost_wt delta (pp) at cs100, lmsys, n=7, verified via paired.py.
    # CI = paired-t 95% half-width. Green only where the CI clears zero (a real win).
    labels = ["Drift\nflat (z11)", "Drift\ndefault (z12)",
              "Stationary\nsharp (z15)", "Stationary\nflat (z11)"]
    vals = [-0.43, 0.39, -1.50, 4.56]
    cis = [2.68, 4.71, 4.90, 3.41]
    wins = [(v - c) > 0 for v, c in zip(vals, cis)]  # CI excludes zero

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    colors = ["#1a7f37" if w else "#9aa0a6" for w in wins]
    bars = ax.bar(range(len(vals)), vals, yerr=cis, capsize=5, color=colors,
                  width=0.6, edgecolor="#222", linewidth=0.6,
                  error_kw=dict(ecolor="#444", lw=1.1))
    ax.axhline(0, color="#222", lw=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("CA $-$ GDSF cost-weighted advantage (pp)", fontsize=9.5)
    ax.set_title("Head-to-head vs GDSF: CA overtakes the cost-aware baseline\n"
                 "only under flat stationary skew (cs100, paired $n=7$, 95% CI)", fontsize=10.5)
    for b, v, c in zip(bars, vals, cis):
        ax.text(b.get_x() + b.get_width() / 2, v + c + 0.25 if v >= 0 else v - c - 0.25,
                f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=9, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#1a7f37", edgecolor="#222", label="CA beats GDSF (CI excludes 0)"),
                       Patch(facecolor="#9aa0a6", edgecolor="#222", label="tie (CI spans 0)")],
              loc="upper left", fontsize=8.5, frameon=False)
    ax.margins(y=0.20)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_gdsf.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    request_flow()
    crossover()
    gdsf_headtohead()
