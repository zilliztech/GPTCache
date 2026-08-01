"""Turn the benchmark CSVs into figures and a summary table.

Pure CSV -> output: this script computes no cache behaviour of its own, so the
figures can always be regenerated from committed data without rerunning the
sweep.

Statistics
----------
Policies are compared with a **paired bootstrap**. Every policy sees the same
ten seeds on the same trace, and a seed fixes the arrival sequence, so the runs
pair up seed-by-seed. For each resample we draw ten seeds with replacement and
average the *per-seed difference* against the baseline, which cancels the
between-seed variance that would otherwise swamp a few points of hit rate.
Intervals are 95% percentile intervals over 10,000 resamples.

Outputs (to ``benchmarks/results/``)
------------------------------------
==================================  ========================================
``fig1_hitrate_vs_capacity.png``    hit rate vs capacity, per trace, CI bands
``fig2_latency_cdf.png``            per-request latency CDF at one capacity
``fig3_ablation.png``               ARC vs ARC-exact -- the one-flag ablation
``fig4_p_over_time.png``            p trajectory; the flat line is the point
``fig5_sketch_inertness.png``       exact-key vs LSH-keyed frequency sketch
``fig6_false_hit_rate.png``         wrong-cluster hits, the underreported cost
``summary.md`` / ``summary.csv``    relative improvement vs LRU, with CIs
==================================  ========================================

Usage::

    python benchmarks/analyze.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# One fixed colour per policy, held across every figure: colour follows the
# entity, never its rank. Slots are taken in order from a categorical palette
# validated for CVD separation and contrast on a light surface.
COLORS = {
    "ARC":       "#2a78d6",   # slot 1 blue
    "ARC-exact": "#eb6834",   # slot 2 orange
    "LRU":       "#1baf7a",   # slot 3 aqua
    "LFU":       "#eda100",   # slot 4 yellow
    "FIFO":      "#e87ba4",   # slot 5 magenta
    "RR":        "#008300",   # slot 6 green
    "LRU-batch": "#4a3aa7",   # slot 7 violet
}
# marker shape is a second, redundant channel, so identity never rests on
# colour alone (three of these hues sit below 3:1 on a light surface)
MARKERS = {"ARC": "o", "ARC-exact": "s", "LRU": "^", "LFU": "D",
           "FIFO": "v", "RR": "P", "LRU-batch": "X"}
ORDER = ["ARC", "ARC-exact", "LRU", "LRU-batch", "LFU", "FIFO", "RR"]
TRACE_LABEL = {
    "quora-stationary": "Quora, stationary popularity",
    "quora-drift": "Quora, popularity drift",
    "wildchat": "WildChat, real arrival order",
}

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dcdbd6"
BASELINE = "LRU"
N_BOOT = 10_000

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# --------------------------------------------------------------------------
def bootstrap_paired(values, base, n_boot=N_BOOT, seed=0):
    """95% CI for ``mean(values - base)`` over paired seeds.

    :param values: per-seed metric for the policy under test.
    :param base: per-seed metric for the baseline, same seed order.
    """
    values = np.asarray(values, dtype=float)
    base = np.asarray(base, dtype=float)
    diff = values - base
    n = len(diff)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def bootstrap_mean(values, n_boot=N_BOOT, seed=0):
    """95% CI for the mean of a single policy's per-seed metric."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def _legend(ax, ncol=4):
    ax.legend(frameon=False, ncol=ncol, loc="lower right",
              handlelength=1.8, columnspacing=1.2, labelcolor=INK)


# --------------------------------------------------------------------------
def fig_hitrate(df):
    tr = [t for t in TRACE_LABEL if t in set(df.trace)]
    fig, axes = plt.subplots(1, len(tr), figsize=(13, 4.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, trace in zip(axes, tr):
        sub = df[df.trace == trace]
        caps = sorted(sub.capacity.unique())
        for pol in ORDER:
            if pol not in set(sub.policy):
                continue
            mid, lo, hi = [], [], []
            for c in caps:
                v = sub[(sub.policy == pol) & (sub.capacity == c)].hit_rate
                m, l, h = bootstrap_mean(v.values)
                mid.append(m * 100)
                lo.append(l * 100)
                hi.append(h * 100)
            ax.plot(caps, mid, color=COLORS[pol], marker=MARKERS[pol],
                    markersize=5, linewidth=2, label=pol)
            ax.fill_between(caps, lo, hi, color=COLORS[pol], alpha=0.15,
                            linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(caps)
        ax.set_xticklabels([str(c) for c in caps])
        ax.set_xlabel("cache capacity (entries)")
        ax.set_title(TRACE_LABEL[trace])
    axes[0].set_ylabel("hit rate (%)")
    _legend(axes[0], ncol=2)
    fig.suptitle("Hit rate vs capacity  (mean of 10 seeds, 95% bootstrap CI)",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    out = os.path.join(RESULTS, "fig1_hitrate_vs_capacity.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def fig_latency(path):
    if not os.path.exists(path):
        print("  (skipped fig2: lat_cdf.csv missing)")
        return
    d = pd.read_csv(path)
    cap = int(d.capacity.iloc[0])
    trace = d.trace.iloc[0]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for pol in ORDER:
        s = d[d.policy == pol]
        if s.empty:
            continue
        g = s.groupby("quantile").latency_us.mean()
        ax.plot(g.values, g.index.values / 100.0, color=COLORS[pol],
                linewidth=2, label=pol)
    ax.set_xscale("log")
    ax.set_xlabel("per-request latency (microseconds, log scale)")
    ax.set_ylabel("fraction of requests <= x")
    ax.set_ylim(0, 1.005)
    ax.set_title(f"Per-request latency CDF\n{TRACE_LABEL.get(trace, trace)}, "
                 f"capacity {cap}, mean of 3 seeds")
    _legend(ax, ncol=2)
    fig.tight_layout()
    out = os.path.join(RESULTS, "fig2_latency_cdf.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def fig_ablation(df):
    tr = [t for t in TRACE_LABEL if t in set(df.trace)]
    fig, axes = plt.subplots(1, len(tr), figsize=(13, 4.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, trace in zip(axes, tr):
        sub = df[df.trace == trace]
        caps = sorted(sub.capacity.unique())
        x = np.arange(len(caps))
        width = 0.36
        for k, pol in enumerate(("ARC-exact", "ARC")):
            mid, err = [], [[], []]
            for c in caps:
                v = sub[(sub.policy == pol) & (sub.capacity == c)].hit_rate
                m, lo, hi = bootstrap_mean(v.values)
                mid.append(m * 100)
                err[0].append((m - lo) * 100)
                err[1].append((hi - m) * 100)
            ax.bar(x + (k - 0.5) * width, mid, width * 0.94,
                   color=COLORS[pol], label=pol,
                   yerr=err, ecolor=INK_2, capsize=2,
                   error_kw={"linewidth": 1})
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in caps])
        ax.set_xlabel("cache capacity (entries)")
        ax.set_title(TRACE_LABEL[trace])
    axes[0].set_ylabel("hit rate (%)")
    axes[0].legend(frameon=False, loc="upper left", labelcolor=INK)
    fig.suptitle("Ablation: semantic vs exact-key ghost lists  "
                 "(one constructor flag, everything else identical)",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    out = os.path.join(RESULTS, "fig3_ablation.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def fig_p_over_time(path):
    if not os.path.exists(path):
        print("  (skipped fig4: p_trace.csv missing)")
        return
    d = pd.read_csv(path)
    cap = int(d.capacity.iloc[0])
    trace = d.trace.iloc[0]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for pol in ("ARC", "ARC-exact"):
        s = d[d.policy == pol].sort_values("step")
        if s.empty:
            continue
        ax.plot(s.step, s.p, color=COLORS[pol], linewidth=2, label=pol)
    ax.axhline(0, color=GRID, linewidth=1)
    ax.set_xlabel("request number")
    ax.set_ylabel("p  (adaptive target size for T1)")
    ax.set_ylim(bottom=-cap * 0.03)
    ax.set_title(f"ARC's adaptation state over time\n"
                 f"{TRACE_LABEL.get(trace, trace)}, capacity {cap}")
    ax.annotate("exact-key ghosts: pinned at 0 for the entire trace.\n"
                "The ghost lists fill up, but an id-keyed ghost can only match\n"
                "an id that by construction never returns, so it never fires.",
                xy=(len(d[d.policy == 'ARC-exact']) and
                    d[d.policy == "ARC-exact"].step.max() * 0.42 or 0, 0),
                xytext=(0.30, 0.20), textcoords="axes fraction",
                fontsize=10, color=INK_2,
                arrowprops={"arrowstyle": "->", "color": INK_2,
                            "linewidth": 1})
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    out = os.path.join(RESULTS, "fig4_p_over_time.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def fig_sketch(path):
    if not os.path.exists(path):
        print("  (skipped fig5: gate_screening.json missing)")
        return
    with open(path) as fh:
        gate = json.load(fh)
    sk = gate["check1"]["sketch"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    keys = ["exact", "lsh"]
    labels = ["exact key\n(as TinyLFU keys it)", "LSH of the embedding\n(contrast only)"]
    corr = [sk[k]["corr_with_true_popularity"] for k in keys]
    cols = [COLORS["ARC-exact"], COLORS["ARC"]]
    a1.bar(labels, corr, color=cols, width=0.55)
    a1.axhline(0, color=INK_2, linewidth=1)
    for i, c in enumerate(corr):
        a1.text(i, c + (0.03 if c >= 0 else -0.06), f"{c:+.3f}",
                ha="center", fontsize=11, color=INK)
    a1.set_ylabel("corr(sketch estimate, true cluster popularity)")
    a1.set_ylim(-0.15, 1.0)
    a1.set_title("Does the frequency estimate\ncarry any signal?")

    est = [sk[k]["mean_estimate"] for k in keys]
    a2.bar(labels, est, color=cols, width=0.55)
    for i, e in enumerate(est):
        a2.text(i, e + 0.08, f"{e:.2f}", ha="center", fontsize=11, color=INK)
    a2.set_ylabel("mean frequency estimate")
    a2.set_title("...and does it accumulate\nat all?")

    fig.suptitle("The same failure mode in a second policy family: an "
                 "exact-key frequency sketch is inert on a semantic workload",
                 y=1.03, fontsize=12)
    fig.tight_layout()
    out = os.path.join(RESULTS, "fig5_sketch_inertness.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def fig_false_hits(df):
    sub = df[df.trace.str.startswith("quora")]
    if sub.empty:
        print("  (skipped fig6: no trace with ground-truth clusters)")
        return
    tr = [t for t in TRACE_LABEL if t in set(sub.trace)]
    fig, axes = plt.subplots(1, len(tr), figsize=(9.6, 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, trace in zip(axes, tr):
        s = sub[sub.trace == trace]
        caps = sorted(s.capacity.unique())
        for pol in ORDER:
            if pol not in set(s.policy):
                continue
            mid = [s[(s.policy == pol) & (s.capacity == c)]
                   .false_hit_rate.mean() * 100 for c in caps]
            ax.plot(caps, mid, color=COLORS[pol], marker=MARKERS[pol],
                    markersize=5, linewidth=2, label=pol)
        ax.set_xscale("log", base=2)
        ax.set_xticks(caps)
        ax.set_xticklabels([str(c) for c in caps])
        ax.set_xlabel("cache capacity (entries)")
        ax.set_title(TRACE_LABEL[trace])
    axes[0].set_ylabel("false hit rate (% of hits)")
    # these curves rise left-to-right, so the upper-left corner is the free one
    axes[0].legend(frameon=False, ncol=2, loc="upper left", handlelength=1.8,
                   columnspacing=1.2, labelcolor=INK)
    fig.suptitle("False hit rate: hits served from a different ground-truth "
                 "cluster (a wrong answer)", y=1.02, fontsize=12.5)
    fig.tight_layout()
    out = os.path.join(RESULTS, "fig6_false_hit_rate.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


# --------------------------------------------------------------------------
def summary(df):
    """Relative improvement vs LRU at each capacity, with paired CIs."""
    rows = []
    for trace in df.trace.unique():
        for cap in sorted(df.capacity.unique()):
            s = df[(df.trace == trace) & (df.capacity == cap)]
            base = s[s.policy == BASELINE].sort_values("seed").hit_rate.values
            for pol in ORDER:
                v = s[s.policy == pol].sort_values("seed")
                if v.empty:
                    continue
                mean, lo, hi = bootstrap_paired(v.hit_rate.values, base)
                rows.append({
                    "trace": trace, "capacity": cap, "policy": pol,
                    "hit_rate": v.hit_rate.mean(),
                    "delta_vs_lru_pp": mean * 100,
                    "ci_lo_pp": lo * 100, "ci_hi_pp": hi * 100,
                    "significant": not (lo <= 0 <= hi),
                    "false_hit_rate": v.false_hit_rate.mean(),
                    "lat_mean_us": v.lat_mean_us.mean(),
                    "lat_p99_us": v.lat_p99_us.mean(),
                    "evict_us_per_req": v.evict_us_per_req.mean(),
                    "peak_heap_kb": v.peak_heap_kb.mean(),
                    "policy_vectors": v.policy_vectors.mean(),
                    "resident_mean": v.resident_mean.mean(),
                    "throughput_qps": v.throughput_qps.mean(),
                })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "summary.csv"), index=False)

    lines = ["# Semantic-ARC benchmark summary", ""]
    lines.append(f"Baseline for all deltas: **{BASELINE}**. "
                 f"Deltas are percentage points of hit rate, with 95% paired "
                 f"bootstrap CIs over 10 seeds ({N_BOOT:,} resamples). "
                 f"**Bold** = interval excludes zero.")
    lines.append("")

    for trace in out.trace.unique():
        lines.append(f"## {TRACE_LABEL.get(trace, trace)}")
        lines.append("")
        lines.append("| capacity | " + " | ".join(ORDER) + " |")
        lines.append("|---" * (len(ORDER) + 1) + "|")
        for cap in sorted(out.capacity.unique()):
            cells = []
            for pol in ORDER:
                r = out[(out.trace == trace) & (out.capacity == cap)
                        & (out.policy == pol)]
                if r.empty:
                    cells.append("-")
                    continue
                r = r.iloc[0]
                txt = (f"{r.hit_rate * 100:.2f}%<br>"
                       f"{r.delta_vs_lru_pp:+.2f} "
                       f"[{r.ci_lo_pp:+.2f}, {r.ci_hi_pp:+.2f}]")
                cells.append(f"**{txt}**" if r.significant and pol != BASELINE
                             else txt)
            lines.append(f"| {cap} | " + " | ".join(cells) + " |")
        lines.append("")

    # worst case across regimes -- the robustness claim
    lines.append("## Worst case across all three regimes")
    lines.append("")
    lines.append("The claim for ARC is robustness, not a uniform win: the best "
                 "*worst case*, with no tuning knob to set per workload.")
    lines.append("")
    lines.append("| capacity | " + " | ".join(ORDER) + " |")
    lines.append("|---" * (len(ORDER) + 1) + "|")
    for cap in sorted(out.capacity.unique()):
        cells = []
        for pol in ORDER:
            r = out[(out.capacity == cap) & (out.policy == pol)]
            cells.append(f"{r.hit_rate.min() * 100:.2f}%" if not r.empty
                         else "-")
        lines.append(f"| {cap} | " + " | ".join(cells) + " |")
    lines.append("")

    # cost table
    lines.append("## Cost of the ghost scan")
    lines.append("")
    lines.append("| trace | capacity | policy | mean lat (us) | p99 lat (us) "
                 "| evict decision (us/req) | policy vectors | peak heap (KB) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for trace in out.trace.unique():
        for cap in (400, 1600):
            for pol in ("LRU", "ARC"):
                r = out[(out.trace == trace) & (out.capacity == cap)
                        & (out.policy == pol)]
                if r.empty:
                    continue
                r = r.iloc[0]
                lines.append(
                    f"| {trace} | {cap} | {pol} | {r.lat_mean_us:.2f} | "
                    f"{r.lat_p99_us:.2f} | {r.evict_us_per_req:.2f} | "
                    f"{r.policy_vectors:.0f} | {r.peak_heap_kb:.0f} |")
    lines.append("")

    path = os.path.join(RESULTS, "summary.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  {path}")
    print(f"  {os.path.join(RESULTS, 'summary.csv')}")
    return out


def main():
    bench = os.path.join(RESULTS, "bench.csv")
    if not os.path.exists(bench):
        print("benchmarks/results/bench.csv missing -- run run_bench.py first.")
        return 2
    df = pd.read_csv(bench)
    print(f"loaded {len(df)} rows: {df.trace.nunique()} traces, "
          f"{df.policy.nunique()} policies, {df.capacity.nunique()} capacities, "
          f"{df.seed.nunique()} seeds")
    print("writing:")
    fig_hitrate(df)
    fig_latency(os.path.join(RESULTS, "lat_cdf.csv"))
    fig_ablation(df)
    fig_p_over_time(os.path.join(RESULTS, "p_trace.csv"))
    fig_sketch(os.path.join(RESULTS, "gate_screening.json"))
    fig_false_hits(df)
    summary(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
