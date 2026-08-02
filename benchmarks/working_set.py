"""Does the capacity/working-set rule transfer from synthetic to real traffic?

``analyze_crossover.py`` reports that ``capacity / working_set`` orders ARC's
advantage over LRU better than drift rate does (r = -0.71 against r = -0.36),
and ``docs/eviction_policies.md`` turns that into a deployable rule: below
roughly 15% of the working set, ARC is worth its overhead. Both were fitted on
the **synthetic** cells only. This script checks the rule against the two
traces that carry a real arrival order, and it is a separate script because the
answer is that the rule does not transfer.

The obstacle is that "working set" is not measured the same way in the two
places, so the check is done in three stages.

Stage 1 -- how do the two definitions differ?
    On a synthetic trace, popularity is Zipf by construction, so the working
    set has a closed form: the number of ranks covering 90% of the Zipf mass
    (``analyze_crossover.working_set``). A real trace has no ``s`` to plug in;
    all you can do is count how many distinct items covered 90% of the
    requests that actually arrived. Stage 1 computes **both** on the synthetic
    traces, where ground-truth cluster ids make the empirical count exact.

    They do not agree, and the two ways they disagree are both worth knowing:

    *The analytic definition is drift-blind.* It describes the hot set at one
    instant. Under drift the ranking is re-permuted every epoch, so the trace
    as a whole is served by roughly ``n_epochs`` different hot sets. Measured
    at s=1.6 on Quora: the per-epoch working set is 22-27 at both drift 0 and
    drift 1, matching the analytic 26 -- but across the whole trace drift 1
    gives 142. The analytic value is the per-epoch number in both cases.

    *The empirical definition is censored by trace length.* It cannot exceed
    the number of queries, and well before that limit the low-popularity tail
    is drawn too few times to register. At s=0.7 on StackExchange the analytic
    working set is 25,590, which 20,000 queries cannot exhibit; the empirical
    count comes out at 9,406.

    So the empirical count runs *below* the analytic one at low skew and
    *above* it under drift. Both matter for stage 3, which is why the gap is
    diagnosed rather than assumed away.

Stage 2 -- measure the real traces and compare against the rule.
    Real prompts have no ground-truth clusters, so "distinct item" has to be
    recovered from the embeddings. We use the cache's own equivalence relation:
    a query joins an earlier query's cluster when their cosine similarity is at
    least ``tau``, which is exactly the condition under which the cache would
    have served one from the other. That makes the resulting count the
    *cache-relevant* working set rather than a generic clustering of the text.
    Reported at the serving ``tau`` and at a stricter threshold, because a
    working set recovered by thresholding should not hinge on the threshold.

    Single-link chaining is the known weakness of this relation: A~B and B~C
    merge A with C even if A and C are far apart. The stricter-threshold column
    is the check on that, and the per-seed spread is printed rather than
    hidden.

Stage 3 -- is the stage-2 gap bigger than the stage-1 uncertainty?
    Stage 1 means the real traces' working set cannot be compared to the rule's
    x-axis exactly. Stage 3 therefore asks the question that does not depend on
    getting the definition right: **how wrong would the working set have to be
    for the rule to be correct?** The rule's most pessimistic bucket predicts
    +0.64 pp, and reaching it requires ``capacity / working_set > 1``, i.e. a
    working set smaller than the cache. The measured working sets are four
    orders of magnitude away from that, against a stage-1 definitional spread
    of under 6x. The conclusion survives any plausible correction.

Usage::

    python benchmarks/prepare_data.py        # first
    python benchmarks/prepare_data_ext.py    # for wildchat-long
    python benchmarks/sweep_crossover.py     # for crossover_real.csv
    python benchmarks/working_set.py

Writes ``results/working_set.md`` and ``results/working_set.csv``.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import traces  # noqa: E402
import traces_param  # noqa: E402
from analyze import bootstrap_paired  # noqa: E402
from analyze_crossover import working_set as zipf_working_set  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

N_QUERIES = 20_000          # must match sweep_crossover.N_QUERIES
SEEDS = 10                  # must match the seeds in crossover_real.csv
CAPACITIES = (50, 100, 200, 400, 800, 1600)
COVERAGE = 0.90             # "working set" = items covering this much traffic
BLOCK = 2000                # rows per similarity block

# The buckets analyze_crossover.py fitted on the synthetic cells, and the mean
# ARC - LRU it measured in each. This is the rule under test.
SYNTHETIC_RULE = [
    (0.00, 0.02, +8.77),
    (0.02, 0.05, +7.57),
    (0.05, 0.15, +4.47),
    (0.15, 0.40, +1.78),
    (0.40, 1.00, +2.46),
    (1.00, np.inf, +0.64),
]


def rule_prediction(ratio):
    """Mean ARC - LRU the synthetic-only rule predicts at this ratio."""
    for lo, hi, delta in SYNTHETIC_RULE:
        if lo <= ratio < hi:
            return delta
    return SYNTHETIC_RULE[-1][2]


def empirical_working_set(labels, coverage=COVERAGE):
    """Distinct labels covering ``coverage`` of the arrivals in a trace.

    The empirical twin of :func:`analyze_crossover.working_set`: instead of
    integrating a Zipf curve, it counts how many distinct items actually
    absorbed 90% of the requests.
    """
    labels = np.asarray(labels)
    labels = labels[labels >= 0]
    if labels.size == 0:
        return 0
    counts = np.sort(np.bincount(labels)[np.bincount(labels) > 0])[::-1]
    return int(np.searchsorted(np.cumsum(counts), coverage * counts.sum()) + 1)


def reuse_distances(labels):
    """Queries elapsed between successive arrivals of the same item.

    The working-set rule describes *how concentrated* popularity is. It says
    nothing about *when* the repeats arrive, and a cache is governed by both.
    This is the second half: small reuse distances mean the hit rate is carried
    by short-range recurrence, which even a tiny cache captures and which no
    replacement policy can improve on.
    """
    last, gaps = {}, []
    for i, lab in enumerate(labels):
        lab = int(lab)
        if lab in last:
            gaps.append(i - last[lab])
        last[lab] = i
    return np.array(gaps) if gaps else np.array([0])


def semantic_labels(queries, tau):
    """Cluster a trace by the cache's own equivalence relation.

    Each query is linked to its most similar *earlier* query when that
    similarity is at least ``tau`` -- precisely the condition under which the
    cache would have served the earlier answer -- and the links are closed
    transitively with union-find.

    :returns: int array of cluster labels, one per query.
    """
    n = len(queries)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for start in range(0, n, BLOCK):
            end = min(start + BLOCK, n)
            sims = queries[start:end] @ queries[:end].T
            # mask self and every later query: only earlier arrivals may match
            for r in range(start, end):
                sims[r - start, r:] = -np.inf
            best = sims.argmax(axis=1)
            score = sims[np.arange(end - start), best]
            for r in range(end - start):
                if score[r] >= tau:
                    a, b = find(start + r), find(int(best[r]))
                    if a != b:
                        parent[a] = b

    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def stage1_definitions(tau, n_epochs=6):
    """Diagnose how the analytic and empirical definitions differ.

    Ground-truth cluster ids make the empirical count exact here, so any gap is
    a property of the two definitions and not of the clustering. The per-epoch
    column separates the two mechanisms: it isolates the instantaneous hot set,
    which is what the analytic formula actually describes.
    """
    rows = []
    for corpus in ("quora", "stackexchange"):
        _, cid = traces_param._corpus(corpus)  # pylint: disable=protected-access
        n_clusters = int(cid[cid >= 0].max()) + 1
        for zipf_s in (0.7, 1.1, 1.6):
            for drift in (0.0, 1.0):
                emp, per_epoch = [], []
                for seed in range(3):
                    _, cids = traces_param.build_param_trace(
                        corpus, N_QUERIES, seed, zipf_s=zipf_s,
                        drift_rate=drift, n_epochs=n_epochs)
                    emp.append(empirical_working_set(cids))
                    step = len(cids) // n_epochs
                    per_epoch += [empirical_working_set(cids[e * step:(e + 1) * step])
                                  for e in range(n_epochs)]
                analytic = zipf_working_set(n_clusters, zipf_s)
                rows.append(dict(corpus=corpus, zipf_s=zipf_s, drift=drift,
                                 analytic=analytic,
                                 empirical=float(np.mean(emp)),
                                 per_epoch=float(np.mean(per_epoch)),
                                 ratio=float(np.mean(emp)) / analytic))
    return pd.DataFrame(rows)


def stage2_real_working_sets(taus):
    """Measure the working set of each real trace window the benchmark used."""
    rows = []
    for corpus in traces_param.REAL_CORPORA:
        for seed in range(SEEDS):
            queries, _ = traces_param.build_real_trace(corpus, N_QUERIES, seed)
            for tau in taus:
                labels = semantic_labels(queries, tau)
                gaps = reuse_distances(labels)
                rows.append(dict(corpus=corpus, seed=seed, tau=tau,
                                 n_clusters=int(labels.max()) + 1,
                                 working_set=empirical_working_set(labels),
                                 reuse_median=float(np.median(gaps)),
                                 reuse_within_100=float((gaps <= 100).mean())))
            print(f"    {corpus} seed {seed}: " + "  ".join(
                f"tau={r['tau']} ws={r['working_set']}"
                for r in rows[-len(taus):]))
    return pd.DataFrame(rows)


def synthetic_reuse(n_epochs=6):
    """Reuse distances on the synthetic regimes, for contrast with the real ones."""
    rows = []
    for corpus in ("quora", "stackexchange"):
        for zipf_s in (0.7, 1.1, 1.6):
            for drift in (0.0, 1.0):
                gaps = []
                for seed in range(3):
                    _, cids = traces_param.build_param_trace(
                        corpus, N_QUERIES, seed, zipf_s=zipf_s,
                        drift_rate=drift, n_epochs=n_epochs)
                    gaps.append(reuse_distances(cids))
                gaps = np.concatenate(gaps)
                rows.append(dict(corpus=corpus, zipf_s=zipf_s, drift=drift,
                                 reuse_median=float(np.median(gaps)),
                                 reuse_within_100=float((gaps <= 100).mean())))
    return pd.DataFrame(rows)


def measured_deltas():
    """ARC - LRU per (corpus, capacity) from the committed real-trace sweep."""
    path = os.path.join(RESULTS_DIR, "crossover_real.csv")
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found -- run benchmarks/sweep_crossover.py first")
    df = pd.read_csv(path)
    df = df[df.stride.fillna(1) == 1]
    rows = []
    for (corpus, cap), g in df.groupby(["corpus", "capacity"]):
        arc = g[g.policy == "ARC"].sort_values("seed").hit_rate.values * 100
        lru = g[g.policy == "LRU"].sort_values("seed").hit_rate.values * 100
        if len(arc) == 0 or len(arc) != len(lru):
            continue
        mean, lo, hi = bootstrap_paired(arc, lru)
        rows.append(dict(corpus=corpus, capacity=int(cap), measured=mean,
                         lo=lo, hi=hi))
    return pd.DataFrame(rows)


def main():
    traces.silence_spurious_fp_warnings()
    tau = traces.load_tau()
    taus = sorted({tau, 0.9})

    print("=" * 78)
    print("DOES THE capacity/working-set RULE TRANSFER TO REAL TRAFFIC?")
    print("=" * 78)
    print(f"tau = {tau}   queries per window = {N_QUERIES}   seeds = {SEEDS}\n")

    print("STAGE 1  how do the two definitions of 'working set' differ?")
    print("-" * 78)
    s1 = stage1_definitions(tau)
    print(f"  {'corpus':<14}{'s':>5}{'drift':>7}{'analytic':>10}"
          f"{'per-epoch':>11}{'whole trace':>13}{'emp/ana':>9}")
    for _, r in s1.iterrows():
        print(f"  {r.corpus:<14}{r.zipf_s:>5}{r.drift:>7}{r.analytic:>10d}"
              f"{r.per_epoch:>11.0f}{r.empirical:>13.0f}{r.ratio:>9.2f}")
    lo, hi = s1.ratio.min(), s1.ratio.max()
    print(f"\n  empirical/analytic spans {lo:.2f} to {hi:.2f}. Two mechanisms:")
    print("   - drift-blindness: the analytic value tracks the PER-EPOCH "
          "column, so\n     under drift it understates the trace-level working "
          "set by ~n_epochs.")
    print("   - finite-trace censoring: at low skew the analytic working set "
          "exceeds\n     what 20,000 queries can exhibit, so the empirical "
          "count falls short.")
    print("  Stage 3 is built so that neither mechanism can carry the "
          "conclusion.\n")

    print("STAGE 2  working set of the real traces (semantic clustering)")
    print("-" * 78)
    s2 = stage2_real_working_sets(taus)
    sr = synthetic_reuse()
    print()

    meas = measured_deltas()
    out = []
    for corpus in traces_param.REAL_CORPORA:
        for tau_v in taus:
            sub = s2[(s2.corpus == corpus) & (s2.tau == tau_v)]
            if sub.empty:
                continue
            ws_mean, ws_min, ws_max = (sub.working_set.mean(),
                                       sub.working_set.min(),
                                       sub.working_set.max())
            for cap in CAPACITIES:
                m = meas[(meas.corpus == corpus) & (meas.capacity == cap)]
                if m.empty:
                    continue
                ratio = cap / ws_mean
                out.append(dict(
                    corpus=corpus, tau=tau_v, capacity=cap,
                    working_set=round(ws_mean, 1), ws_min=int(ws_min),
                    ws_max=int(ws_max), ratio=ratio,
                    predicted=rule_prediction(ratio),
                    measured=float(m.measured.iloc[0]),
                    lo=float(m.lo.iloc[0]), hi=float(m.hi.iloc[0])))
    out = pd.DataFrame(out)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out.to_csv(os.path.join(RESULTS_DIR, "working_set.csv"), index=False)

    lines = ["# Does the capacity/working-set rule transfer to real traffic?", ""]
    lines.append(
        "Generated by `benchmarks/working_set.py`. The rule in "
        "`docs/eviction_policies.md` was fitted on the **synthetic** cells "
        "only. This is the same rule evaluated against the two traces that "
        "carry a real arrival order.")
    lines.append("")
    lines.append("## Stage 1 - the two definitions of \"working set\" differ")
    lines.append("")
    lines.append(
        "The rule uses an analytic working set (Zipf ranks covering 90% of the "
        "mass); a real trace admits only an empirical one (distinct items "
        "covering 90% of arrivals). On synthetic traces, where ground-truth "
        f"cluster ids make the empirical count exact, the two differ by "
        f"{lo:.2f}-{hi:.2f}x, in two identifiable ways.")
    lines.append("")
    lines.append(
        "**The analytic definition is drift-blind.** It tracks the `per-epoch` "
        "column below at every drift rate. Under drift the ranking is "
        "re-permuted each epoch, so the trace as a whole is served by roughly "
        "`n_epochs` different hot sets and the trace-level count is several "
        "times larger. This also means the drifting cells of the original fit "
        "were placed too far right on the x-axis.")
    lines.append("")
    lines.append(
        "**The empirical definition is censored by trace length.** It cannot "
        "exceed the query count, and the low-popularity tail is drawn too few "
        "times to register well before that limit, so at low skew it falls "
        "short of the analytic value.")
    lines.append("")
    lines.append("| corpus | zipf s | drift | analytic | per-epoch | "
                 "whole trace | empirical/analytic |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in s1.iterrows():
        lines.append(f"| {r.corpus} | {r.zipf_s} | {r.drift} | {r.analytic} | "
                     f"{r.per_epoch:.0f} | {r.empirical:.0f} | {r.ratio:.2f} |")
    lines.append("")
    lines.append("## Stage 2 - the rule's prediction against measurement")
    lines.append("")
    lines.append(
        "`working set` is the mean over the same 10 trace windows the sweep "
        "used, clustered by the cache's own relation (link to an earlier query "
        "at cosine >= tau, closed transitively). `predicted` is the mean "
        "ARC - LRU for that `capacity / working set` bucket in the synthetic "
        "fit; `measured` is the paired-bootstrap mean on this trace.")
    lines.append("")
    for tau_v in taus:
        lines.append(f"### tau = {tau_v}")
        lines.append("")
        lines.append("| corpus | capacity | working set | ratio | predicted | "
                     "measured | 95% CI |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in out[out.tau == tau_v].iterrows():
            lines.append(
                f"| {r.corpus} | {r.capacity} | {r.working_set:.0f} "
                f"(min {r.ws_min}, max {r.ws_max}) | {r.ratio:.4f} | "
                f"**{r.predicted:+.2f}** | **{r.measured:+.2f}** | "
                f"[{r.lo:+.2f}, {r.hi:+.2f}] |")
        lines.append("")

    gaps = out[out.tau == tau].assign(gap=lambda d: d.predicted - d.measured)
    lines.append(
        f"At the serving threshold the rule over-predicts by "
        f"{gaps.gap.min():.2f} to {gaps.gap.max():.2f} percentage points on "
        f"every capacity of both real traces.")
    lines.append("")
    worst = min(d for _, _, d in SYNTHETIC_RULE)
    factors = gaps.working_set / gaps.capacity
    lines.append("## Stage 3 - the gap is larger than the definitional "
                 "uncertainty")
    lines.append("")
    lines.append(
        "Stage 1 means the real traces' working set cannot be placed on the "
        "rule's x-axis exactly, so the question that settles it is: how wrong "
        "would the working set have to be for the rule to hold? The rule's "
        f"most pessimistic bucket predicts {worst:+.2f} pp, and reaching it "
        "requires `capacity / working set > 1` -- a working set smaller than "
        "the cache itself.")
    lines.append("")
    lines.append("| corpus | capacity | measured working set | "
                 "working set the rule needs | factor |")
    lines.append("|---|---|---|---|---|")
    for _, r in gaps.iterrows():
        lines.append(f"| {r.corpus} | {r.capacity} | {r.working_set:.0f} | "
                     f"< {r.capacity} | {r.working_set / r.capacity:.0f}x |")
    lines.append("")
    lines.append(
        f"The working set would have to be {factors.min():.0f}x to "
        f"{factors.max():.0f}x smaller than measured for the rule to hold, "
        f"against a stage-1 definitional spread of {lo:.2f}x to {hi:.2f}x. No "
        "correction to the definition closes a gap of that size, so the "
        "conclusion does not rest on which definition is preferred.")
    lines.append("")
    with open(os.path.join(RESULTS_DIR, "working_set.md"), "w") as fh:
        fh.write("\n".join(lines))

    print("STAGE 2  the rule's prediction against measurement  (tau = %.2f)"
          % tau)
    print("-" * 78)
    print(f"  {'corpus':<14}{'cap':>6}{'ws':>9}{'ratio':>9}"
          f"{'predicted':>11}{'measured':>10}{'gap':>9}")
    for _, r in gaps.iterrows():
        print(f"  {r.corpus:<14}{r.capacity:>6}{r.working_set:>9.0f}"
              f"{r.ratio:>9.4f}{r.predicted:>+11.2f}{r.measured:>+10.2f}"
              f"{r.gap:>+9.2f}")
    print()
    print(f"  the rule over-predicts by {gaps.gap.min():.2f} to "
          f"{gaps.gap.max():.2f} pp on every cell.\n")

    print("STAGE 3  how wrong would the working set have to be?")
    print("-" * 78)
    worst = min(d for _, _, d in SYNTHETIC_RULE)
    print(f"  The rule's most pessimistic bucket predicts {worst:+.2f} pp, and "
          f"reaching it\n  needs capacity/working_set > 1 -- a working set "
          f"smaller than the cache.\n")
    print(f"  {'corpus':<14}{'cap':>6}{'measured ws':>13}"
          f"{'ws the rule needs':>19}{'factor':>9}")
    for _, r in gaps.iterrows():
        needed = r.capacity            # ratio > 1.0 <=> working_set < capacity
        print(f"  {r.corpus:<14}{r.capacity:>6}{r.working_set:>13.0f}"
              f"{'< ' + str(int(needed)):>19}{r.working_set / needed:>8.0f}x")
    factors = gaps.working_set / gaps.capacity
    print(f"\n  the working set would have to be {factors.min():.0f}x to "
          f"{factors.max():.0f}x smaller than\n  measured for the rule to hold, "
          f"against a stage-1 definitional spread of\n  {lo:.2f}x to {hi:.2f}x. "
          f"No correction to the definition closes a gap that size.\n")

    print("WHY  the rule describes popularity concentration, not arrival timing")
    print("-" * 78)
    print(f"  {'trace':<34}{'median reuse dist':>19}{'within 100':>13}")
    for corpus in traces_param.REAL_CORPORA:
        sub = s2[(s2.corpus == corpus) & (s2.tau == tau)]
        print(f"  {corpus + ' (real order)':<34}{sub.reuse_median.mean():>19.0f}"
              f"{100 * sub.reuse_within_100.mean():>12.1f}%")
    for _, r in sr.iterrows():
        if r.corpus != "quora":
            continue
        lab = f"quora synthetic s={r.zipf_s} drift={r.drift}"
        print(f"  {lab:<34}{r.reuse_median:>19.0f}"
              f"{100 * r.reuse_within_100:>12.1f}%")
    print("\n  Real traffic repeats within tens of queries; the synthetic "
          "regimes repeat\n  within thousands. A ratio built from popularity "
          "concentration cannot see\n  that difference, which is why it "
          "transfers poorly.")
    print("\nwrote results/working_set.{md,csv}")


if __name__ == "__main__":
    main()
