"""Paired cross-seed delta between two policies (collapses cross-seed variance).

The seeds are paired: seed k feeds the SAME query stream to every policy, so the
right confidence statement on "policy A beats policy B" is the per-seed paired
delta A-B, not the difference of unpaired means. With few seeds we report the
paired mean +/- std and the sign-consistency count (positive in N/seeds).

Usage:
  python bench_lmsys/paired.py "bench_lmsys/win_z11_seed*.json" CA_W_TINYLFU LRU
"""

import glob
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict

METRICS = [("cost_weighted_hit_rate", "cost_wt"), ("hit_rate", "hit"), ("token_saving_ratio", "tok")]

# t_{.975} critical values by df (two-sided 95%); falls back to 1.96 for large df.
# ponytail: small hardcoded table, no scipy needed for the n<=~15 paired runs here.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145}


def wilcoxon_p(deltas):
    """Exact two-sided Wilcoxon signed-rank p-value. Enumerates all 2^n sign
    assignments over the fixed (tie-averaged) ranks, so it is exact for small n
    and needs no scipy. Zeros are dropped (Wilcoxon convention)."""
    xs = [d for d in deltas if d != 0.0]
    n = len(xs)
    if n == 0:
        return 1.0
    mags = sorted(abs(d) for d in xs)
    # average ranks for tied magnitudes
    rank = {}
    i = 0
    while i < n:
        j = i
        while j < n and mags[j] == mags[i]:
            j += 1
        avg = (i + 1 + j) / 2.0  # mean of ranks i+1..j
        rank[mags[i]] = avg
        i = j
    ranks = [rank[abs(d)] for d in xs]
    w_obs = sum(r for r, d in zip(ranks, xs) if d > 0)
    total = sum(ranks)
    center = total / 2.0
    dev = abs(w_obs - center)
    extreme = 0
    for signs in itertools.product((0, 1), repeat=n):
        w = sum(r for r, s in zip(ranks, signs) if s)
        if abs(w - center) >= dev - 1e-9:
            extreme += 1
    return extreme / (2 ** n)


def ci95(v):
    """Paired-t 95% CI half-width on the mean (pp)."""
    n = len(v)
    if n < 2:
        return 0.0
    sem = statistics.stdev(v) / math.sqrt(n)
    t = _T975.get(n - 1, 1.96)
    return t * sem


def main():
    pattern, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files match {pattern!r}")
    # per[cs][metric] = list of per-seed (A-B) deltas, in pp
    per = defaultdict(lambda: defaultdict(list))
    for fp in files:
        d = json.load(open(fp))
        by = {(r["policy"], r["cache_size"]): r for r in d["results"]}
        css = {cs for (_, cs) in by}
        for cs in css:
            if (a, cs) not in by or (b, cs) not in by:
                continue
            for mkey, _ in METRICS:
                per[cs][mkey].append((by[(a, cs)][mkey] - by[(b, cs)][mkey]) * 100.0)

    n = len(files)
    print(f"Paired {a} - {b}  ({n} seeds)\n")
    print(f"  {'cache':<7}" + " ".join(f"{lbl+'(pp)':>30}" for _, lbl in METRICS))
    print(f"  {'':<7}" + " ".join(f"{'mean [k/n]  CI95  W-p':>30}" for _ in METRICS))
    for cs in sorted(per):
        cells = []
        for mkey, _ in METRICS:
            v = per[cs][mkey]
            m = statistics.mean(v)
            pos = sum(x > 0 for x in v)
            cells.append(f"{m:+6.2f} [{pos}/{n}] +/-{ci95(v):4.2f} p={wilcoxon_p(v):.3f}")
        print(f"  {cs:<7}" + " ".join(f"{c:>30}" for c in cells))


if __name__ == "__main__":
    main()
