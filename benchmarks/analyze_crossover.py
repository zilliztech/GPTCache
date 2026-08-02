"""Turn the crossover sweep into figures and a readable verdict.

Reads ``crossover_*.csv`` from ``sweep_crossover.py`` and answers one question:
under what traffic does ARC beat LRU, and by how much.

Everything is a **paired** comparison -- ARC and LRU see the identical arrival
sequence for a given (corpus, drift, skew, capacity, seed), so the difference
is taken per seed and bootstrapped over seeds, the same estimator
``analyze.py`` uses for the headline tables.

Colour
------
``ARC - LRU`` is a *polarity* quantity: its sign is the whole message. So the
heatmaps use a diverging scale -- one cool hue for "ARC ahead", one warm hue
for "LRU ahead", a neutral grey at exactly zero, and an explicit zero contour.
Cells whose 95% CI straddles zero are cross-hatched, because a coloured cell
that is not statistically distinguishable from zero is the single easiest way
to mislead with this figure.

Outputs (``benchmarks/results/``)
---------------------------------
``fig7_crossover_drift_skew.png``      ARC-LRU over drift x skew
``fig8_crossover_drift_capacity.png``  ARC-LRU over drift x capacity
``fig9_crossover_curves.png``          ARC-LRU vs drift, with CI bands
``fig10_real_traces.png``              every policy on the real streams
``crossover.md``                       tables and the verdict
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import (  # noqa: E402
    COLORS, GRID, INK, INK_2, MARKERS, N_BOOT, ORDER, bootstrap_mean,
    bootstrap_paired,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

CORPUS_LABEL = {
    "quora": "Quora paraphrase clusters",
    "stackexchange": "StackExchange duplicate titles",
    "wildchat": "WildChat (2 months, real order)",
    "wildchat-long": "WildChat (full window, real order)",
}

# diverging pair: cool = ARC ahead, warm = LRU ahead, neutral grey at zero.
# Blue/orange rather than red/green -- the latter is the classic CVD failure.
DIVERGING = LinearSegmentedColormap.from_list(
    "arc_lru",
    ["#7a3b0a", "#c1743a", "#e8c9a8", "#efeeea", "#a9c9e0", "#3d7ea6", "#12455f"],
)


# --------------------------------------------------------------------------
def paired_grid(df, row_key, col_key, base="LRU", test="ARC"):
    """(mean delta, ci_lo, ci_hi) grids in percentage points, per (row, col)."""
    rows = sorted(df[row_key].unique())
    cols = sorted(df[col_key].unique())
    mean = np.full((len(rows), len(cols)), np.nan)
    lo = np.full_like(mean, np.nan)
    hi = np.full_like(mean, np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            cell = df[(df[row_key] == r) & (df[col_key] == c)]
            a = cell[cell.policy == test].sort_values("seed").hit_rate.values
            b = cell[cell.policy == base].sort_values("seed").hit_rate.values
            if len(a) == 0 or len(a) != len(b):
                continue
            m, l, h = bootstrap_paired(a * 100, b * 100, n_boot=N_BOOT)
            mean[i, j], lo[i, j], hi[i, j] = m, l, h
    return rows, cols, mean, lo, hi


def heatmap(ax, rows, cols, mean, lo, hi, xlabel, ylabel, title,
            xfmt=str, yfmt=str):
    """Diverging heatmap with a zero contour and CI-straddles-zero hatching."""
    lim = float(np.nanmax(np.abs(mean))) or 1.0
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    im = ax.imshow(mean, cmap=DIVERGING, norm=norm, aspect="auto",
                   origin="lower", interpolation="nearest")

    ns = (lo <= 0) & (hi >= 0)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isnan(mean[i, j]):
                continue
            if ns[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////",
                    edgecolor="#8a8a86", linewidth=0.0))
            # value labels stay in ink, never in the series colour
            shade = INK if abs(mean[i, j]) < 0.62 * lim else "white"
            ax.text(j, i, f"{mean[i, j]:+.1f}", ha="center", va="center",
                    fontsize=8.5, color=shade)

    # the crossover itself: where the surface changes sign
    if np.nanmin(mean) < 0 < np.nanmax(mean):
        ax.contour(np.arange(len(cols)), np.arange(len(rows)), mean,
                   levels=[0.0], colors=[INK], linewidths=1.6)

    ax.set_xticks(range(len(cols)), [xfmt(c) for c in cols])
    ax.set_yticks(range(len(rows)), [yfmt(r) for r in rows])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(False)
    return im


def _finish_heatfig(fig, axes, im, suptitle, note):
    cb = fig.colorbar(im, ax=axes, fraction=0.045, pad=0.02, aspect=28)
    cb.set_label("ARC - LRU  (percentage points of hit rate)")
    cb.outline.set_edgecolor(GRID)
    # reserve strips top and bottom so neither the title nor the caption can
    # land on a panel title or an axis label
    fig.get_layout_engine().set(rect=(0.006, 0.055, 0.988, 0.88))
    fig.suptitle(suptitle, x=0.012, y=0.985, ha="left", va="top", fontsize=13)
    fig.text(0.012, 0.018, note, fontsize=9, color=INK_2, ha="left")


# --------------------------------------------------------------------------
def fig_drift_skew(df):
    corpora = sorted(df.corpus.unique())
    fig, axes = plt.subplots(1, len(corpora), figsize=(7.0 * len(corpora), 5.6),
                             squeeze=False, layout="constrained")
    im = None
    for ax, corpus in zip(axes[0], corpora):
        sub = df[df.corpus == corpus]
        rows, cols, mean, lo, hi = paired_grid(sub, "drift_rate", "zipf_s")
        im = heatmap(ax, rows, cols, mean, lo, hi,
                     "popularity skew  (Zipf s)",
                     "drift rate  (fraction of ranking reshuffled per epoch)",
                     CORPUS_LABEL.get(corpus, corpus),
                     xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    cap = int(df.capacity.iloc[0])
    _finish_heatfig(
        fig, axes[0].tolist(), im,
        f"When does ARC beat LRU?  Drift x skew at capacity {cap}",
        "Hatched = 95% paired bootstrap CI includes zero (no measurable "
        "difference).  Black line = crossover contour.")
    out = os.path.join(RESULTS_DIR, "fig7_crossover_drift_skew.png")
    fig.savefig(out)
    plt.close(fig)
    print("  wrote", os.path.basename(out))


def fig_drift_capacity(df):
    corpora = sorted(df.corpus.unique())
    fig, axes = plt.subplots(1, len(corpora), figsize=(7.0 * len(corpora), 5.6),
                             squeeze=False, layout="constrained")
    im = None
    for ax, corpus in zip(axes[0], corpora):
        sub = df[df.corpus == corpus]
        rows, cols, mean, lo, hi = paired_grid(sub, "drift_rate", "capacity")
        im = heatmap(ax, rows, cols, mean, lo, hi,
                     "cache capacity (entries)",
                     "drift rate",
                     CORPUS_LABEL.get(corpus, corpus),
                     xfmt=lambda v: f"{int(v)}", yfmt=lambda v: f"{v:g}")
    skew = df.zipf_s.iloc[0]
    _finish_heatfig(
        fig, axes[0].tolist(), im,
        f"Drift x capacity at Zipf s={skew:g}",
        "Hatched = 95% paired bootstrap CI includes zero.  "
        "Black line = crossover contour.")
    out = os.path.join(RESULTS_DIR, "fig8_crossover_drift_capacity.png")
    fig.savefig(out)
    plt.close(fig)
    print("  wrote", os.path.basename(out))


def fig_curves(ds, dc):
    """ARC-LRU against drift, with CI bands -- the heatmap's key rows, exactly."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))

    # left: one line per skew, at the grid capacity
    corpus = "quora" if "quora" in set(ds.corpus) else sorted(ds.corpus)[0]
    sub = ds[ds.corpus == corpus]
    skews = sorted(sub.zipf_s.unique())
    ramp = plt.get_cmap("viridis")(np.linspace(0.12, 0.86, len(skews)))
    ax = axes[0]
    for colour, s in zip(ramp, skews):
        cell = sub[sub.zipf_s == s]
        rows, _, mean, lo, hi = paired_grid(cell, "drift_rate", "zipf_s")
        m, l, h = mean[:, 0], lo[:, 0], hi[:, 0]
        ax.fill_between(rows, l, h, color=colour, alpha=0.16, linewidth=0)
        ax.plot(rows, m, color=colour, linewidth=2, marker="o", markersize=5,
                label=f"s={s:g}")
        ax.annotate(f"s={s:g}", (rows[-1], m[-1]), textcoords="offset points",
                    xytext=(6, 0), fontsize=9, color=INK_2, va="center")
    ax.set_title(f"{CORPUS_LABEL.get(corpus, corpus)} — by popularity skew "
                 f"(capacity {int(sub.capacity.iloc[0])})", loc="left")
    ax.set_xlabel("drift rate")
    ax.set_ylabel("ARC - LRU  (pp of hit rate)")

    # right: one line per capacity, at the grid skew
    sub = dc[dc.corpus == corpus]
    caps = sorted(sub.capacity.unique())
    ramp = plt.get_cmap("viridis")(np.linspace(0.12, 0.86, len(caps)))
    ax = axes[1]
    for colour, c in zip(ramp, caps):
        cell = sub[sub.capacity == c]
        rows, _, mean, lo, hi = paired_grid(cell, "drift_rate", "capacity")
        m, l, h = mean[:, 0], lo[:, 0], hi[:, 0]
        ax.fill_between(rows, l, h, color=colour, alpha=0.16, linewidth=0)
        ax.plot(rows, m, color=colour, linewidth=2, marker="o", markersize=5,
                label=f"cap {int(c)}")
        ax.annotate(f"{int(c)}", (rows[-1], m[-1]), textcoords="offset points",
                    xytext=(6, 0), fontsize=9, color=INK_2, va="center")
    ax.set_title(f"{CORPUS_LABEL.get(corpus, corpus)} — by capacity "
                 f"(Zipf s={sub.zipf_s.iloc[0]:g})", loc="left")
    ax.set_xlabel("drift rate")
    ax.set_ylabel("ARC - LRU  (pp of hit rate)")

    # both panels fall left-to-right, so the free corner differs: the skew
    # panel bottoms out above zero, the capacity panel crosses it
    for ax, loc in zip(axes, ("upper right", "lower left")):
        ax.axhline(0, color=INK, linewidth=1.2, zorder=1)
        ax.legend(frameon=False, ncol=2, fontsize=9, loc=loc)
        ax.margins(x=0.10)
    fig.suptitle("ARC's advantage over LRU shrinks as traffic drifts",
                 y=0.99, x=0.012, ha="left", fontsize=13)
    fig.text(0.012, 0.015,
             "Shaded band = 95% paired bootstrap CI. Zero line = LRU. "
             "Above the line ARC wins, below it LRU wins.",
             fontsize=9, color=INK_2, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    out = os.path.join(RESULTS_DIR, "fig9_crossover_curves.png")
    fig.savefig(out)
    plt.close(fig)
    print("  wrote", os.path.basename(out))


def working_set(n_clusters, s, frac=0.90):
    """Ranks covering ``frac`` of the Zipf mass -- the hot set worth caching."""
    w = 1.0 / np.arange(1, n_clusters + 1) ** s
    w /= w.sum()
    return int(np.searchsorted(np.cumsum(w), frac) + 1)


def _n_clusters(corpus):
    import traces as _t
    cid = np.load(os.path.join(_t.DATA_DIR, f"{corpus}_cid.npy"))
    return int(cid[cid >= 0].max()) + 1


def fig_working_set(ds, dc):
    """The unifying variable: cache size relative to the hot set.

    Drift moves ARC's advantage, but capacity-relative-to-working-set moves it
    far more, and it collapses both corpora and every skew onto one curve.
    """
    nc = {c: _n_clusters(c) for c in set(ds.corpus) | set(dc.corpus)}
    pts = []
    for df in (ds, dc):
        for (corpus, drift, skew, cap), g in df.groupby(
                ["corpus", "drift_rate", "zipf_s", "capacity"]):
            a = g[g.policy == "ARC"].sort_values("seed").hit_rate.values
            b = g[g.policy == "LRU"].sort_values("seed").hit_rate.values
            if len(a) == 0 or len(a) != len(b):
                continue
            m, lo, hi = bootstrap_paired(a * 100, b * 100)
            pts.append(dict(corpus=corpus, drift=drift, skew=skew, cap=cap,
                            ratio=cap / working_set(nc[corpus], skew),
                            delta=m, lo=lo, hi=hi))
    pts = pd.DataFrame(pts).drop_duplicates(
        subset=["corpus", "drift", "skew", "cap"])

    fig, ax = plt.subplots(figsize=(9.6, 5.6), layout="constrained")
    drifts = sorted(pts.drift.unique())
    # drift is a magnitude -> one hue, light to dark. Never a categorical cycle.
    ramp = plt.get_cmap("PuBu")(np.linspace(0.32, 0.97, len(drifts)))
    shapes = {"quora": "o", "stackexchange": "s"}
    # Scatter, not a line plot: consecutive points in ratio order come from
    # different (skew, capacity) families, so joining them would draw a
    # sawtooth that reads as structure and is really just interleaving.
    for colour, dr in zip(ramp, drifts):
        sub = pts[pts.drift == dr]
        for corpus, mk in shapes.items():
            s2 = sub[sub.corpus == corpus]
            ax.scatter(s2.ratio, s2.delta, color=colour, marker=mk, s=30,
                       edgecolor="white", linewidth=0.6, zorder=3)

    # binned median: the trend the correlation is describing, drawn honestly
    lo_e, hi_e = np.log10(pts.ratio.min()), np.log10(pts.ratio.max())
    edges = np.linspace(lo_e, hi_e, 11)
    mids, meds = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = pts[(np.log10(pts.ratio) >= a) & (np.log10(pts.ratio) < b)]
        if len(m) >= 3:
            mids.append(10 ** ((a + b) / 2))
            meds.append(float(m.delta.median()))
    ax.plot(mids, meds, color=INK, linewidth=2.2, zorder=5,
            label="binned median")

    ax.axhline(0, color=INK, linewidth=1.3, linestyle="--", zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("cache capacity / working set   "
                  "(working set = Zipf ranks covering 90% of traffic)")
    ax.set_ylabel("ARC - LRU  (pp of hit rate)")
    ax.set_title("The cache-to-working-set ratio predicts the crossover "
                 "better than drift does", loc="left", fontsize=13)

    sm = plt.cm.ScalarMappable(cmap=LinearSegmentedColormap.from_list(
        "d", ramp), norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("drift rate")
    cb.outline.set_edgecolor(GRID)

    handles = [plt.Line2D([], [], color=INK_2, marker=m, linestyle="none",
                          markersize=7, label=CORPUS_LABEL.get(c, c))
               for c, m in shapes.items()]
    handles.append(plt.Line2D([], [], color=INK, linewidth=2.2,
                              label="binned median"))
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper right")
    ax.set_xlabel("cache capacity / working set   "
                  "(working set = Zipf ranks covering 90% of traffic)")
    # the vertical spread at fixed ratio is skew -- say so on the axes, where
    # it cannot collide with the x label
    ax.text(0.015, 0.03, "spread at fixed ratio is popularity skew:\n"
            "the collapse is real but partial",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")
    out = os.path.join(RESULTS_DIR, "fig11_working_set.png")
    fig.savefig(out)
    plt.close(fig)
    print("  wrote", os.path.basename(out))
    return pts


def fig_real(df):
    corpora = sorted(df.corpus.unique())
    fig, axes = plt.subplots(1, len(corpora), figsize=(6.4 * len(corpora), 4.9),
                             squeeze=False)
    for ax, corpus in zip(axes[0], corpora):
        sub = df[df.corpus == corpus]
        for pol in ORDER:
            cell = sub[sub.policy == pol]
            if cell.empty:
                continue
            # No CI band here on purpose. Across seeds a "seed" is a different
            # window of the year, and real seasonality makes the *absolute*
            # level swing far more than any policy gap -- unpaired bands span
            # 15-35% and hide the lines while answering the wrong question.
            # The comparable quantity is the paired delta, which is in
            # crossover.md and is tight.
            caps, ms = [], []
            for cap, grp in cell.groupby("capacity"):
                caps.append(cap)
                ms.append(grp.hit_rate.mean() * 100)
            ax.plot(caps, ms, color=COLORS[pol], marker=MARKERS[pol],
                    markersize=6, linewidth=2, label=pol)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(sub.capacity.unique()),
                      [str(int(c)) for c in sorted(sub.capacity.unique())])
        ax.set_xlabel("capacity (entries)")
        ax.set_ylabel("hit rate (%)")
        ax.set_title(CORPUS_LABEL.get(corpus, corpus), loc="left")
        ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper left")
    fig.suptitle("Real prompt streams: every policy, every capacity",
                 y=0.99, x=0.012, ha="left", fontsize=13)
    fig.text(0.012, 0.015,
             "Mean over seeds; each seed is a different window of the stream. "
             "Absolute level swings with real seasonality — the comparable\n"
             "quantity is the paired per-seed delta in crossover.md, not the "
             "gap between these lines.",
             fontsize=9, color=INK_2, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    out = os.path.join(RESULTS_DIR, "fig10_real_traces.png")
    fig.savefig(out)
    plt.close(fig)
    print("  wrote", os.path.basename(out))


# --------------------------------------------------------------------------
def crossover_point(rows, mean, lo):
    """Largest drift rate at which ARC is still *significantly* ahead."""
    best = None
    for r, m, l in zip(rows, mean, lo):
        if l > 0:
            best = r
    return best


def _ratio_section(pts):
    """Bucket every grid cell by cache/working-set and report the win rate."""
    out = ["## The unifying variable: cache size vs working set", "",
           "Every synthetic cell, bucketed by `capacity / working_set`, where "
           "the working set is the number of Zipf ranks covering 90% of "
           "traffic. This single ratio orders the results better than drift "
           "does.", "",
           "| capacity / working set | cells | mean ARC - LRU | ARC "
           "significantly ahead | ARC significantly behind |",
           "|---|---|---|---|---|"]
    edges = [0, 0.02, 0.05, 0.15, 0.4, 1.0, np.inf]
    labels = ["< 0.02", "0.02 - 0.05", "0.05 - 0.15", "0.15 - 0.40",
              "0.40 - 1.0", "> 1.0"]
    pts = pts.copy()
    pts["bucket"] = pd.cut(pts.ratio, bins=edges, labels=labels)
    for label, g in pts.groupby("bucket", observed=True):
        win = float((g.lo > 0).mean()) * 100
        lose = float((g.hi < 0).mean()) * 100
        out.append(f"| {label} | {len(g)} | {g.delta.mean():+.2f} pp "
                   f"| {win:.0f}% | {lose:.0f}% |")
    r = np.corrcoef(np.log10(pts.ratio), pts.delta)[0, 1]
    rd = np.corrcoef(pts.drift, pts.delta)[0, 1]
    out += ["",
            f"Correlation with `ARC - LRU`: **log10(capacity/working set) "
            f"r = {r:+.2f}**, drift rate r = {rd:+.2f}. Both push the same "
            "way, but the capacity ratio is the stronger term.", ""]
    return out


def _density_section(dens):
    """Is the long trace's larger ARC margin about span, or about sparsity?"""
    out = ["## Control: is `wildchat-long`'s bigger margin real drift, or "
           "just a thinner stream?", "",
           "`wildchat-long` spans a year but is subsampled to 150k prompts, "
           "so it is **both** longer and ~2.2x sparser in time than "
           "`wildchat`. This control thins the original two-month stream by "
           "`stride` and re-measures, holding the span fixed. If sparsity "
           "alone moves ARC, the extra span cannot be credited.", "",
           "ARC - LRU (pp), on the 2-month corpus, by thinning factor.", "",
           "| capacity | " + " | ".join(
               f"stride {s}" for s in sorted(dens.stride.unique())) + " |",
           "|---" * (len(dens.stride.unique()) + 1) + "|"]
    for cap, g in dens.groupby("capacity"):
        cells = []
        for st, gg in g.groupby("stride"):
            a = gg[gg.policy == "ARC"].sort_values("seed").hit_rate.values
            b = gg[gg.policy == "LRU"].sort_values("seed").hit_rate.values
            m, lo, hi = bootstrap_paired(a * 100, b * 100)
            txt = f"{m:+.2f}"
            if lo > 0 or hi < 0:
                txt = f"**{txt}**"
            cells.append(txt)
        out.append(f"| {int(cap)} | " + " | ".join(cells) + " |")
    out += ["", "Absolute LRU hit rate falls steeply with thinning "
            "(much of WildChat's hit rate is short-range repetition), and "
            "ARC's margin grows. So part of the long trace's larger margin "
            "is sparsity, not span -- the two are confounded and this "
            "benchmark cannot fully separate them.", ""]
    return out


def write_markdown(ds, dc, real, pts=None, dens=None):
    out = [
        "# When does ARC actually beat LRU?",
        "",
        "Generated by `benchmarks/analyze_crossover.py` from "
        "`crossover_*.csv`. Every number is a **paired** difference: ARC and "
        "LRU see the identical arrival sequence per seed, differences are "
        f"taken per seed and bootstrapped over seeds ({N_BOOT:,} resamples, "
        "95% CI). **Bold** = interval excludes zero.",
        "",
        "`drift_rate` is the fraction of the popularity ranking re-permuted "
        "at each of 6 epochs: 0.0 is a fixed ranking for the whole trace, "
        "1.0 is a total reshuffle every epoch (the `quora-drift` regime of "
        "the original benchmark).",
        "",
    ]

    for corpus in sorted(ds.corpus.unique()):
        sub = ds[ds.corpus == corpus]
        cap = int(sub.capacity.iloc[0])
        out += [f"## {CORPUS_LABEL.get(corpus, corpus)} — drift x skew "
                f"(capacity {cap})", "",
                "ARC - LRU, percentage points.", ""]
        rows, cols, mean, lo, hi = paired_grid(sub, "drift_rate", "zipf_s")
        out.append("| drift \\ skew | " + " | ".join(f"s={c:g}" for c in cols)
                   + " |")
        out.append("|---" * (len(cols) + 1) + "|")
        for i, r in enumerate(rows):
            cells = []
            for j in range(len(cols)):
                txt = f"{mean[i, j]:+.2f}"
                if lo[i, j] > 0 or hi[i, j] < 0:
                    txt = f"**{txt}**"
                cells.append(txt)
            out.append(f"| {r:g} | " + " | ".join(cells) + " |")
        out.append("")
        for j, c in enumerate(cols):
            xp = crossover_point(rows, mean[:, j], lo[:, j])
            out.append(f"- skew s={c:g}: ARC is significantly ahead up to "
                       + (f"drift **{xp:g}**" if xp is not None
                          else "**no drift rate at all**"))
        out.append("")

    for corpus in sorted(dc.corpus.unique()):
        sub = dc[dc.corpus == corpus]
        out += [f"## {CORPUS_LABEL.get(corpus, corpus)} — drift x capacity "
                f"(Zipf s={sub.zipf_s.iloc[0]:g})", "",
                "ARC - LRU, percentage points.", ""]
        rows, cols, mean, lo, hi = paired_grid(sub, "drift_rate", "capacity")
        out.append("| drift \\ capacity | "
                   + " | ".join(str(int(c)) for c in cols) + " |")
        out.append("|---" * (len(cols) + 1) + "|")
        for i, r in enumerate(rows):
            cells = []
            for j in range(len(cols)):
                txt = f"{mean[i, j]:+.2f}"
                if lo[i, j] > 0 or hi[i, j] < 0:
                    txt = f"**{txt}**"
                cells.append(txt)
            out.append(f"| {r:g} | " + " | ".join(cells) + " |")
        out.append("")

    if pts is not None and not pts.empty:
        out += _ratio_section(pts)

    if real is not None and not real.empty:
        out += ["## Real prompt streams", "",
                "Hit rate, mean over seeds; ARC and LFU deltas are paired "
                "against LRU.", ""]
        for corpus in sorted(real.corpus.unique()):
            sub = real[real.corpus == corpus]
            out += [f"### {CORPUS_LABEL.get(corpus, corpus)}", "",
                    "| capacity | LRU | ARC | ARC - LRU | LFU - LRU |",
                    "|---|---|---|---|---|"]
            for cap, grp in sub.groupby("capacity"):
                base = grp[grp.policy == "LRU"].sort_values("seed")
                arc = grp[grp.policy == "ARC"].sort_values("seed")
                lfu = grp[grp.policy == "LFU"].sort_values("seed")
                if base.empty or arc.empty:
                    continue
                m, l, h = bootstrap_paired(arc.hit_rate.values * 100,
                                           base.hit_rate.values * 100)
                dtxt = f"{m:+.2f} [{l:+.2f}, {h:+.2f}]"
                if l > 0 or h < 0:
                    dtxt = f"**{dtxt}**"
                if lfu.empty:
                    ltxt = "—"
                else:
                    lm, ll, lh = bootstrap_paired(lfu.hit_rate.values * 100,
                                                  base.hit_rate.values * 100)
                    ltxt = f"{lm:+.2f} [{ll:+.2f}, {lh:+.2f}]"
                    if ll > 0 or lh < 0:
                        ltxt = f"**{ltxt}**"
                out.append(f"| {int(cap)} | {base.hit_rate.mean() * 100:.2f}% "
                           f"| {arc.hit_rate.mean() * 100:.2f}% | {dtxt} "
                           f"| {ltxt} |")
            out.append("")

    if dens is not None and not dens.empty:
        out += _density_section(dens)

    path = os.path.join(RESULTS_DIR, "crossover.md")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("  wrote", os.path.basename(path))


# --------------------------------------------------------------------------
def main():
    def load(name):
        p = os.path.join(RESULTS_DIR, f"crossover_{name}.csv")
        return pd.read_csv(p) if os.path.exists(p) else None

    ds, dc = load("drift_skew"), load("drift_capacity")
    real, dens = load("real"), load("density")
    if ds is None or dc is None:
        sys.exit("run benchmarks/sweep_crossover.py first")

    print("figures:")
    fig_drift_skew(ds)
    fig_drift_capacity(dc)
    fig_curves(ds, dc)
    pts = fig_working_set(ds, dc)
    if real is not None and not real.empty:
        fig_real(real)
    write_markdown(ds, dc, real, pts, dens)


if __name__ == "__main__":
    main()
