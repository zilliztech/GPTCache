"""Paired cross-seed statistics for the embedding-cache benchmark.

Machine wall-clock noise can vary the *absolute* baseline/cached times a lot
between runs (background processes, thermal throttling, OneDrive sync, etc).
The right question is not "what's the exact average speedup" but "does the
cache consistently win, seed after seed" -- a paired comparison, exactly like
bench_lmsys/paired.py uses for CA_W_TINYLFU vs LRU. Absolute-magnitude noise
mostly cancels out because baseline and cached are measured back-to-back
within the same run (same background conditions for both arms of a seed).

Usage:
  python bench_embedding_cache/paired_stats.py "bench_embedding_cache/results_seed*.json"
"""

import glob
import itertools
import json
import sys


def wilcoxon_p(deltas):
    """Exact two-sided Wilcoxon signed-rank p-value (no scipy needed for
    small n). Same algorithm as bench_lmsys/paired.py. Zeros are dropped."""
    xs = [d for d in deltas if d != 0.0]
    n = len(xs)
    if n == 0:
        return 1.0
    mags = sorted(abs(d) for d in xs)
    rank = {}
    i = 0
    while i < n:
        j = i
        while j < n and mags[j] == mags[i]:
            j += 1
        avg = (i + 1 + j) / 2.0
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


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} <glob pattern for results_seed*.json>")
    pattern = sys.argv[1]
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files match {pattern!r}")

    rows = []
    for fp in files:
        d = json.load(open(fp))
        baseline_t = d["baseline"]["total_seconds"]
        cached_t = d["with_cache"]["total_seconds"]
        speedup = baseline_t / cached_t if cached_t > 0 else float("inf")
        # Paired delta on a scale that's stable across noisy absolute times:
        # log-speedup. delta > 0 means cache was faster in this seed.
        import math
        log_speedup = math.log(speedup) if speedup > 0 else 0.0
        rows.append({
            "file": fp,
            "seed": d["args"].get("seed"),
            "dup_ratio": d["exact_duplicate_ratio"],
            "hit_rate": d["cache_stats"]["hit_rate"],
            "baseline_s": baseline_t,
            "cached_s": cached_t,
            "speedup": speedup,
            "log_speedup": log_speedup,
        })

    n = len(rows)
    wins = sum(1 for r in rows if r["speedup"] > 1.0)
    speedups = [r["speedup"] for r in rows]
    log_speedups = [r["log_speedup"] for r in rows]
    mean_speedup = sum(speedups) / n
    mean_log = sum(log_speedups) / n
    # geometric mean speedup -- the right "average" for a ratio metric,
    # much less sensitive to one noisy outlier run than the arithmetic mean.
    import math
    geo_mean_speedup = math.exp(mean_log)

    p = wilcoxon_p(log_speedups)

    print("=" * 72)
    print(f"Paired embedding-cache statistics over {n} seeds")
    print("=" * 72)
    for r in rows:
        print(f"  seed={r['seed']}: dup_ratio={r['dup_ratio']*100:5.1f}%  "
              f"hit_rate={r['hit_rate']*100:5.1f}%  "
              f"baseline={r['baseline_s']:7.2f}s  cached={r['cached_s']:7.2f}s  "
              f"speedup={r['speedup']:5.2f}x")
    print()
    print(f"  Cache faster than baseline in {wins}/{n} seeds")
    print(f"  Arithmetic mean speedup : {mean_speedup:.2f}x")
    print(f"  Geometric mean speedup  : {geo_mean_speedup:.2f}x  (robust to one noisy run)")
    print(f"  Wilcoxon signed-rank p  : {p:.4f}  "
          f"({'significant at alpha=0.05' if p < 0.05 else 'NOT significant at alpha=0.05'})")

    out = {
        "n_seeds": n,
        "wins": wins,
        "mean_speedup": mean_speedup,
        "geo_mean_speedup": geo_mean_speedup,
        "wilcoxon_p": p,
        "per_seed": rows,
    }
    out_path = "bench_embedding_cache/paired_stats.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()