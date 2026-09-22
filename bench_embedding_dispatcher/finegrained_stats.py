"""Aggregate EmbeddingDispatcher benchmark runs into a per-concurrency-level
mean +/- std table.

Usage:
  python bench_embedding_dispatcher/finegrained_stats.py "bench_embedding_dispatcher/results_fine_run*.json"
"""

import glob
import json
import statistics
import sys
from collections import defaultdict


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} <glob pattern for results_fine_run*.json>")
    pattern = sys.argv[1]
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files match {pattern!r}")

    by_concurrency = defaultdict(list)
    rss_by_concurrency = defaultdict(list)

    for fp in files:
        d = json.load(open(fp))
        for row in d["results"]:
            c = row["concurrency"]
            by_concurrency[c].append(row["speedup"])
            rss_by_concurrency[c].append(row["dispatcher"]["rss_mb"])

    print("=" * 88)
    print(f"Fine-grained EmbeddingDispatcher speedup over {len(files)} runs")
    print("=" * 88)
    print(f"{'Concurrency':>12} | {'n':>3} | {'mean speedup':>13} | {'std':>6} | {'min':>6} | {'max':>6} | verdict")
    print("-" * 88)

    for c in sorted(by_concurrency):
        vals = by_concurrency[c]
        n = len(vals)
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if n > 1 else 0.0
        verdict = "wins" if mean > 1.0 else "loses"
        print(f"{c:>12} | {n:>3} | {mean:>13.2f} | {std:>6.2f} | {min(vals):>6.2f} | {max(vals):>6.2f} | {verdict}")

    out = {
        "n_runs": len(files),
        "per_concurrency": {
            str(c): {
                "speedups": by_concurrency[c],
                "mean_speedup": statistics.mean(by_concurrency[c]),
                "std_speedup": statistics.stdev(by_concurrency[c]) if len(by_concurrency[c]) > 1 else 0.0,
                "mean_rss_mb": statistics.mean(rss_by_concurrency[c]) if rss_by_concurrency[c][0] is not None else None,
            }
            for c in by_concurrency
        },
    }
    out_path = "bench_embedding_dispatcher/finegrained_stats.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
