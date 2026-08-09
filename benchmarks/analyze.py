"""Turn the benchmark CSVs into figures and a summary table.

Pure CSV -> output: this script computes no cache behaviour of its own, so the
figures can always be regenerated from the sweep CSVs without rerunning the
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
``fig12_isomemory.png``             hit rate vs bytes, ARC charged for ghosts
``fig13_charge_sensitivity.png``    where the ARC win survives that charge
``summary.md`` / ``summary.csv``    relative improvement vs LRU, with CIs
==================================  ========================================

The iso-memory figures need ``results/payload.json`` from
``measure_payload.py``; without it they are skipped and the rest still builds.

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
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

# Overridable so a reduced run (smoke/quick) cannot overwrite the committed
# reference artifacts; run_pipeline.sh points those at a scratch subdirectory.
RESULTS = (os.environ.get("BENCH_RESULTS_DIR")
           or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))

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
# this dict is the filter *and* the ordering for every per-trace figure: a
# trace missing from it is silently dropped from figs 1/2/3/6/12/13
TRACE_LABEL = {
    "quora-stationary": "Quora, stationary popularity",
    "quora-drift": "Quora, popularity drift",
    "wildchat": "WildChat, real arrival order",
    "wikianswers-stationary": "WikiAnswers, stationary popularity",
    "wikianswers-drift": "WikiAnswers, popularity drift",
}
# traces built on a corpus with ground-truth cluster ids, so a hit can be
# checked against the cluster it came from and false hit rate is meaningful
GROUND_TRUTH_TRACES = {"quora-stationary", "quora-drift",
                       "wikianswers-stationary", "wikianswers-drift"}
# panel rows are laid out per trace; these keep a 3-panel figure at its
# original size while letting a 5-panel one stay legible
PANEL_W = 4.35
PANEL_W_NARROW = 4.8

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
    fig, axes = plt.subplots(1, len(tr),
                             figsize=(PANEL_W * len(tr), 4.2), sharey=False)
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
    fig, axes = plt.subplots(1, len(tr),
                             figsize=(PANEL_W * len(tr), 4.2), sharey=False)
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
    sub = df[df.trace.isin(GROUND_TRUTH_TRACES)]
    if sub.empty:
        print("  (skipped fig6: no trace with ground-truth clusters)")
        return
    tr = [t for t in TRACE_LABEL if t in set(sub.trace)]
    fig, axes = plt.subplots(1, len(tr),
                             figsize=(PANEL_W_NARROW * len(tr), 4.2),
                             sharey=True)
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
# Iso-memory: charging ARC for its ghosts
#
# fig1 plots hit rate against *entries*, which assumes every policy pays the
# same price per entry. ARC does not -- it keeps up to c embedding-carrying
# ghosts on top of its c residents. measure_payload.py measures the real charge
# from the WildChat responses; these two figures re-run the comparison in bytes
# and then show how the verdict moves if that charge is wrong.

# capacity is an ordered variable, so it gets a sequential single-hue ramp
# (light -> dark), not categorical hues
CAP_RAMP = ["#a9c9ef", "#7fabe4", "#5389d6", "#2f6ac0", "#17508f"]
GHOST_POLICIES = ("ARC", "ARC-exact")   # the only policies charged a surcharge


def load_payload():
    """Measured byte model from ``measure_payload.py``, or ``None``."""
    path = os.path.join(RESULTS, "payload.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _interp_loglinear(pts, x):
    """Hit rate at fractional capacity ``x``, linear in ``log(capacity)``.

    Hit rate is close to linear against log capacity over a doubling grid, so
    this reads a policy's curve between two measured capacities rather than
    requiring a rerun at 1.45x every point. Outside the grid it clamps, which
    is why the sensitivity figure stops at the capacity where 2.0x still lands
    inside the measured range.
    """
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (c0, h0), (c1, h1) in zip(pts, pts[1:]):
        if c0 <= x <= c1:
            w = (np.log(x) - np.log(c0)) / (np.log(c1) - np.log(c0))
            return h0 + w * (h1 - h0)
    return pts[-1][1]


def _seed_curve(sub, policy, seed):
    """``[(capacity, hit_rate)]`` for one policy and one seed, capacity-sorted.

    Seeds pair across capacities -- a seed fixes the arrival sequence -- so
    interpolating within a seed keeps the paired bootstrap valid.
    """
    s = sub[(sub.policy == policy) & (sub.seed == seed)]
    return sorted(zip(s.capacity.values, s.hit_rate.values))


def _iso_delta_per_seed(sub, cap, ratio, policy="ARC", base=BASELINE):
    """Per-seed ``policy(cap) - base(ratio*cap)`` hit-rate difference.

    Empty when ``ratio * cap`` falls outside the measured capacity grid:
    ``_interp_loglinear`` would clamp, which silently reports the 1.0x number
    at every charge instead of admitting the baseline was never measured that
    large.
    """
    if cap * ratio > max(sub.capacity):
        return np.empty(0)
    out = []
    for seed in sorted(sub.seed.unique()):
        pol_pts = dict(_seed_curve(sub, policy, seed))
        base_pts = _seed_curve(sub, base, seed)
        if cap not in pol_pts or not base_pts:
            continue
        out.append(pol_pts[cap] - _interp_loglinear(base_pts, cap * ratio))
    return np.asarray(out, dtype=float)


def fig_isomemory(df, payload):
    """Hit rate against bytes rather than entries."""
    resident = payload["resident_bytes_mean"]
    ratio = payload["ratio"]
    tr = [t for t in TRACE_LABEL if t in set(df.trace)]
    fig, axes = plt.subplots(1, len(tr),
                             figsize=(PANEL_W * len(tr), 4.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, trace in zip(axes, tr):
        sub = df[df.trace == trace]
        caps = sorted(sub.capacity.unique())
        for pol in ORDER:
            if pol not in set(sub.policy):
                continue
            charge = ratio if pol in GHOST_POLICIES else 1.0
            xs = [c * resident * charge / 1e6 for c in caps]
            mid, lo, hi = [], [], []
            for c in caps:
                v = sub[(sub.policy == pol) & (sub.capacity == c)].hit_rate
                m, l, h = bootstrap_mean(v.values)
                mid.append(m * 100)
                lo.append(l * 100)
                hi.append(h * 100)
            ax.plot(xs, mid, color=COLORS[pol], marker=MARKERS[pol],
                    markersize=5, linewidth=2, label=pol)
            ax.fill_between(xs, lo, hi, color=COLORS[pol], alpha=0.15,
                            linewidth=0)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("cache memory (MB)")
        ax.set_title(TRACE_LABEL[trace])
    axes[0].set_ylabel("hit rate (%)")
    _legend(axes[0], ncol=2)
    fig.suptitle(
        f"Hit rate vs memory  (ARC charged {ratio:.2f}x for its ghosts; "
        f"resident entry {resident / 1024:.1f} KB, measured)",
        y=1.02, fontsize=13)
    fig.tight_layout()
    out = os.path.join(RESULTS, "fig12_isomemory.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def fig_charge_sensitivity(df, payload):
    """ARC's iso-memory margin as a function of the assumed ghost charge.

    The measured charge depends on response length, which varies by
    deployment. This sweeps the charge from 1.0 (ghosts free) to 2.0 (the
    empty-payload worst case, and what a naive entry-count comparison implies)
    so the reader can see where the verdict flips rather than taking one
    number on faith.
    """
    ratio = payload["ratio"]
    ratios = np.linspace(1.0, 2.0, 41)
    tr = [t for t in TRACE_LABEL if t in set(df.trace)]
    fig, axes = plt.subplots(1, len(tr),
                             figsize=(PANEL_W * len(tr), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, trace in zip(axes, tr):
        sub = df[df.trace == trace]
        caps = sorted(sub.capacity.unique())
        # only capacities whose 2.0x charge stays inside the measured grid,
        # so no point on the curve is a clamped extrapolation
        caps = [c for c in caps if c * ratios[-1] <= max(caps)]
        for i, c in enumerate(caps):
            ys = [_iso_delta_per_seed(sub, c, r).mean() * 100 for r in ratios]
            ax.plot(ratios, ys, color=CAP_RAMP[i % len(CAP_RAMP)],
                    linewidth=2, label=f"c = {c}")
        ax.axhline(0, color=INK_2, linewidth=1, linestyle="-", zorder=1)
        ax.axvline(ratio, color=INK, linewidth=1.2, linestyle="--", zorder=1)
        ax.annotate(f"measured\n{ratio:.2f}x", xy=(ratio, ax.get_ylim()[1]),
                    xytext=(3, -4), textcoords="offset points",
                    va="top", ha="left", fontsize=9, color=INK)
        ax.axvline(2.0, color=INK_2, linewidth=1, linestyle=":", zorder=1)
        ax.annotate("empty\npayload", xy=(2.0, ax.get_ylim()[1]),
                    xytext=(-3, -4), textcoords="offset points",
                    va="top", ha="right", fontsize=9, color=INK_2)
        ax.set_xlabel("memory charged to ARC (x capacity)")
        ax.set_title(TRACE_LABEL[trace])
    axes[0].set_ylabel("ARC - LRU at equal memory (pp)")
    axes[0].legend(frameon=False, ncol=1, loc="lower left",
                   handlelength=1.8, labelcolor=INK)
    fig.suptitle("Where the ARC advantage survives its own memory cost  "
                 "(above zero = ARC wins at equal bytes)",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    out = os.path.join(RESULTS, "fig13_charge_sensitivity.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def isomemory_section(df, payload):
    """Markdown for the iso-memory comparison, appended to summary.md."""
    ratio = payload["ratio"]
    resident = payload["resident_bytes_mean"]
    lines = ["## Iso-memory: ARC charged for its ghosts", ""]
    lines.append(
        f"ARC(c) holds c residents plus up to c embedding-only ghosts, so it "
        f"cannot be compared to {BASELINE}(c) entry-for-entry. The charge is "
        f"`1 + ghost/resident`, not 2x: a ghost is the "
        f"{payload['embedding_bytes']} B embedding alone, while a resident "
        f"also carries the question and the cached response. Measured over the "
        f"{payload['n_entries']:,} WildChat entries "
        f"(response mean {payload['response_bytes']['mean']:.0f} B, median "
        f"{payload['response_bytes']['median']:.0f} B; question mean "
        f"{payload['question_bytes']['mean']:.0f} B), a resident entry is "
        f"{resident:.0f} B and the charge is **{ratio:.3f}x**. The 2.0x figure "
        f"corresponds to an empty payload, which is what the simulator holds "
        f"and what an entry-count plot implicitly assumes.")
    lines.append("")
    lines.append(f"Deltas below are ARC(c) minus {BASELINE}(charge x c), "
                 f"{BASELINE} interpolated log-linearly within each seed, "
                 f"95% paired bootstrap CI. **Bold** = interval excludes zero. "
                 f"`-` means charge x c lands past the largest capacity in the "
                 f"sweep, so {BASELINE} was never measured there.")
    lines.append("")

    charges = [("1.00x (ghosts free)", 1.0),
               (f"{ratio:.2f}x (measured)", ratio),
               ("2.00x (empty payload)", 2.0)]
    for trace in [t for t in TRACE_LABEL if t in set(df.trace)]:
        sub = df[df.trace == trace]
        caps = sorted(sub.capacity.unique())
        lines.append(f"### {TRACE_LABEL[trace]}")
        lines.append("")
        lines.append("| charge | " + " | ".join(f"c={c}" for c in caps) + " |")
        lines.append("|---" * (len(caps) + 1) + "|")
        for label, r in charges:
            cells = []
            for c in caps:
                d = _iso_delta_per_seed(sub, c, r)
                if len(d) == 0:
                    cells.append("-")
                    continue
                m, lo, hi = bootstrap_mean(d)
                txt = f"{m * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]"
                cells.append(f"**{txt}**" if not lo <= 0 <= hi else txt)
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


# --------------------------------------------------------------------------
def load_timing():
    """The single-process timing pass, if run_bench.py has produced one."""
    path = os.path.join(RESULTS, "timing.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def cost_section(out):
    """Time and space cost, from the uncontended pass rather than the sweep.

    The sweep's own timing columns are measured inside a worker pool, so they
    scale with ``--jobs`` and with whatever else the machine was doing; the same
    cell has moved threefold between runs while its hit rate held to four
    decimals. Everything here comes from ``timing.csv`` instead, which
    re-measures these cells one at a time.
    """
    timing = load_timing()
    contended = timing is None
    src = out if contended else (
        timing.groupby(["trace", "capacity", "policy"], as_index=False).mean(
            numeric_only=True))

    lines = ["## Cost of the ghost scan", ""]
    if contended:
        lines.append("> **These numbers are not quotable.** `timing.csv` is "
                     "missing, so the table falls back to the sweep, whose "
                     "timing columns are measured under `--jobs` contention. "
                     "Run `run_bench.py --timing-only` and re-run this script.")
    else:
        lines.append("Measured by the single-process timing pass "
                     "(`run_bench.py` -> `timing.csv`), mean of "
                     f"{timing.seed.nunique()} seeds. "
                     "Latency and throughput are properties of this host; the "
                     "ARC/LRU *ratio* is the portable quantity. Hit rate is "
                     "deterministic and unaffected.")
    lines.append("")
    lines.append("| trace | capacity | policy | mean lat (us) | p95 lat (us) "
                 "| p99 lat (us) | throughput (q/s) | evict decision (us/req) "
                 "| policy vectors | peak heap (KB) |")
    lines.append("|---" * 10 + "|")
    for trace in out.trace.unique():
        for cap in (400, 1600):
            for pol in ("LRU", "ARC"):
                r = src[(src.trace == trace) & (src.capacity == cap)
                        & (src.policy == pol)]
                if r.empty:
                    continue
                r = r.iloc[0]
                lines.append(
                    f"| {trace} | {cap} | {pol} | {r.lat_mean_us:.2f} | "
                    f"{r.lat_p95_us:.2f} | {r.lat_p99_us:.2f} | "
                    f"{r.throughput_qps:,.0f} | {r.evict_us_per_req:.2f} | "
                    f"{r.policy_vectors:.0f} | {r.peak_heap_kb:.0f} |")
    lines.append("")
    return lines


def summary(df, payload=None):
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
                    "lat_p50_us": v.lat_p50_us.mean(),
                    "lat_p95_us": v.lat_p95_us.mean(),
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
    n_tr = len(set(out.trace))
    lines.append(f"## Worst case across all {n_tr} regimes")
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

    # memory cost, then time cost
    if payload is not None:
        lines.extend(isomemory_section(df, payload))

    # cost table
    lines.extend(cost_section(out))

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
    payload = load_payload()
    if payload is None:
        print("note: results/payload.json missing -- skipping the iso-memory "
              "figures. Run measure_payload.py to generate it.")
    print("writing:")
    fig_hitrate(df)
    fig_latency(os.path.join(RESULTS, "lat_cdf.csv"))
    fig_ablation(df)
    fig_p_over_time(os.path.join(RESULTS, "p_trace.csv"))
    fig_sketch(os.path.join(RESULTS, "gate_screening.json"))
    fig_false_hits(df)
    if payload is not None:
        fig_isomemory(df, payload)
        fig_charge_sensitivity(df, payload)
    summary(df, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
