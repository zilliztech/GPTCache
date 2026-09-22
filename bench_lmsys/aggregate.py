"""Aggregate multi-seed benchmark_lmsys runs into paper-ready tables.

Each input JSON (one per seed) already holds repeats-averaged metrics per
(policy, cache_size). This script pools those per-seed means and reports
mean +/- std ACROSS SEEDS, which is the confidence interval we cite in the
paper. It also emits the cost-isolation ablation (CA vs WTINYLFU_FREQ).

Read-only: it never writes back to the benchmark JSONs.

Usage:
  python bench_lmsys/aggregate.py                         # default glob below
  python bench_lmsys/aggregate.py "bench_lmsys/drift_e2_seed*.json"
"""

import glob
import json
import statistics
import sys
from collections import defaultdict

DEFAULT_GLOB = "bench_lmsys/drift_e2_seed*.json"

# Metrics to aggregate (json key -> display header), all reported as percentages.
METRICS = [
    ("cost_weighted_hit_rate", "cost_wt%"),
    ("hit_rate", "hit%"),
    ("token_saving_ratio", "tok%"),
]

POLICY_ORDER = ["LRU", "LFU", "WTINYLFU_FREQ", "CA_W_TINYLFU", "CA_W_TINYLFU_ADAPT"]


def _policy_rank(p):
    return POLICY_ORDER.index(p) if p in POLICY_ORDER else len(POLICY_ORDER)


def load(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files match {pattern!r}")
    # samples[(policy, cs)][metric_key] = [per-seed mean, ...]
    samples = defaultdict(lambda: defaultdict(list))
    seeds = []
    for fp in files:
        d = json.load(open(fp))
        seeds.append(d.get("args", {}).get("seed", "?"))
        for r in d["results"]:
            key = (r["policy"], r["cache_size"])
            for mkey, _ in METRICS:
                samples[key][mkey].append(r[mkey] * 100.0)
    return files, seeds, samples


def msd(vals):
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GLOB
    files, seeds, samples = load(pattern)

    cache_sizes = sorted({cs for (_, cs) in samples})
    policies = sorted({p for (p, _) in samples}, key=_policy_rank)

    print(f"Aggregated {len(files)} file(s), seeds={seeds}")
    print("=" * 78)

    # ---- Table 1: main results, mean +/- std across seeds ----
    print("\nTABLE 1 - cost-weighted hit rate (primary), hit%, tok%  [mean +/- std]\n")
    for cs in cache_sizes:
        print(f"  cache_size = {cs}")
        print(f"    {'policy':<15} " + " ".join(f"{h:>16}" for _, h in METRICS))
        # find per-metric winner for bolding (asterisk)
        best = {}
        for mkey, _ in METRICS:
            best[mkey] = max(
                (p for p in policies if (p, cs) in samples),
                key=lambda p: statistics.mean(samples[(p, cs)][mkey]),
            )
        for p in policies:
            if (p, cs) not in samples:
                continue
            cells = []
            for mkey, _ in METRICS:
                m, s = msd(samples[(p, cs)][mkey])
                star = "*" if best[mkey] == p else " "
                cells.append(f"{m:6.2f}+/-{s:4.2f}{star}")
            print(f"    {p:<15} " + " ".join(f"{c:>16}" for c in cells))
        print()

    # ---- Table 2: cost-isolation ablation (CA vs frequency-only twin) ----
    print("TABLE 2 - cost-isolation ablation: CA_W_TINYLFU - WTINYLFU_FREQ\n")
    print(f"  {'cache_size':<12} {'d_cost_wt(pp)':>16} {'d_hit(pp)':>16} {'d_tok(pp)':>16}")
    for cs in cache_sizes:
        ca = samples.get(("CA_W_TINYLFU", cs))
        fr = samples.get(("WTINYLFU_FREQ", cs))
        if not ca or not fr:
            continue
        d_cost = statistics.mean(ca["cost_weighted_hit_rate"]) - statistics.mean(fr["cost_weighted_hit_rate"])
        d_hit = statistics.mean(ca["hit_rate"]) - statistics.mean(fr["hit_rate"])
        d_tok = statistics.mean(ca["token_saving_ratio"]) - statistics.mean(fr["token_saving_ratio"])
        print(f"  {cs:<12} {d_cost:>+16.2f} {d_hit:>+16.2f} {d_tok:>+16.2f}")


if __name__ == "__main__":
    main()
