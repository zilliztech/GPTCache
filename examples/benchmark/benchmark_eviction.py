"""Eviction-policy benchmark — Zipf workload, heterogeneous costs.

Drives `EvictionBase` directly (no FAISS, no SQLite, no embedder) so the
runtime is bounded by Python loop overhead, not I/O or model inference.
Goal: produce a clean, fast, reproducible comparison of eviction policies
on a workload that *actually* exercises eviction (small cache vs. large
item pool, with re-access).

Pass `--policies LRU,LFU,FIFO,CA_W_TINYLFU` to compare the cost-aware policy
against the classic baselines and diff the JSON output.

Why this exists separately from `benchmark_qqp.py`:
  - `benchmark_qqp.py` sets `max_size = max(scale * 2, 100_000)`, so
    eviction *never fires* — useless for comparing eviction policies.
  - QQP queries each TP/FP exactly once per repeat — no re-access
    pattern, so eviction quality (re-access prediction) is unobservable.
  - The embedder dominates QQP runtime; for eviction policy
    comparison the embedder is irrelevant.

This harness instead uses:
  - A fixed pool of N synthetic items, each with a synthetic LLMCost.
  - A Zipf-distributed query stream over the pool (re-access heavy on
    a hot head, sparse on a long tail).
  - A small cache (`max_size << N`) that forces evictions.
  - Two metrics: plain hit-rate and *cost-weighted* hit-rate.

Usage
-----
  python benchmark_eviction.py                          # defaults, ~seconds
  python benchmark_eviction.py --zipf-alpha 1.5         # sharper skew
  python benchmark_eviction.py --max-size 500           # tighter cache
  python benchmark_eviction.py --policies LRU,LFU       # subset

Output
------
JSON to `<workdir>/results.json` with one row per policy:
  hit_rate, cost_weighted_hit_rate, evictions, elapsed_seconds.
"""

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from gptcache.manager.eviction.manager import EvictionBase
from gptcache.manager.eviction.ca_w_tinylfu import LLMCost


@dataclass
class Item:
    """Synthetic cache item with heterogeneous regeneration cost.

    `cost` follows the same shape as the `LLMCost.cost` property:
        latency_ms × model_tier × (1 + tokens / 1000)
    so CA_W_TINYLFU's utility function operates on the same units.
    """

    key: int
    cost: float
    size: int


def synthesize_items(n_items: int, expensive_fraction: float, seed: int) -> list:
    """Generate N items split into two cost tiers (cheap vs. expensive).

    Default split is 10% expensive (GPT-4-like: 8-15s, 500-2000 tokens,
    tier=20) and 90% cheap (GPT-3.5-like: 200ms-1.5s, 20-300 tokens,
    tier=1). The spread across items is ~600x, which is realistic.
    """
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n_items):
        if rng.random() < expensive_fraction:
            latency_ms = float(rng.uniform(8000, 15000))
            tokens = int(rng.uniform(500, 2000))
            model_tier = 20.0
            size = int(rng.uniform(1000, 5000))
        else:
            latency_ms = float(rng.uniform(200, 1500))
            tokens = int(rng.uniform(20, 300))
            model_tier = 1.0
            size = int(rng.uniform(50, 500))
        cost = latency_ms * model_tier * (1 + tokens / 1000.0)
        items.append(Item(key=i, cost=cost, size=size))
    return items


def zipf_query_stream(n_items: int, n_queries: int, alpha: float, seed: int) -> list:
    """Generate a Zipf-distributed query stream.

    `alpha > 1` is required by numpy.random.zipf. Higher alpha = sharper
    skew (hotter hot head). alpha=1.2 gives a moderate "80/20"-ish curve.
    Samples are taken with oversampling and modulo'd into [0, n_items)
    to fit the pool without crashing on large outliers.
    """
    rng = np.random.default_rng(seed)
    samples = rng.zipf(alpha, size=n_queries * 4)
    indices = (samples - 1) % n_items
    return indices[:n_queries].tolist()


def run_policy(
    policy: str,
    max_size: int,
    items: list,
    query_stream: list,
    clean_size: int,
) -> dict:
    """Drive a single eviction policy through the query stream.

    Hit/miss is decided by eviction.get() — returns None on a miss,
    truthy on a hit (also updates LRU order / frequency state as a
    side effect, which is correct: a real access should update the policy).
    """
    eviction_count = 0

    def on_evict(keys):
        nonlocal eviction_count
        eviction_count += len(keys)

    eviction = EvictionBase.get(
        name="memory",
        policy=policy,
        maxsize=max_size,
        clean_size=clean_size,
        on_evict=on_evict,
    )

    hits = 0
    cost_hit_sum = 0.0
    cost_total_sum = 0.0

    _is_ca = policy == "CA_W_TINYLFU"

    t0 = time.perf_counter()
    for qi in query_stream:
        item = items[qi]
        cost_total_sum += item.cost
        if eviction.get(qi) is not None:
            hits += 1
            cost_hit_sum += item.cost
        else:
            if _is_ca:
                # Pass the item's synthetic cost so the admission scoring is
                # actually cost-aware, not just frequency-aware.
                eviction.put([qi], costs=[LLMCost(
                    generation_latency_ms=item.cost,
                    token_count=0,
                    model_tier=1.0,
                )])
            else:
                eviction.put([qi])
    elapsed = time.perf_counter() - t0

    return {
        "policy": policy,
        "max_size": max_size,
        "clean_size": clean_size,
        "queries": len(query_stream),
        "hits": hits,
        "misses": len(query_stream) - hits,
        "hit_rate": hits / max(len(query_stream), 1),
        "cost_weighted_hit_rate": cost_hit_sum / max(cost_total_sum, 1e-9),
        "evictions": eviction_count,
        "elapsed_seconds": elapsed,
    }


def describe_workload(items, stream):
    costs = sorted(it.cost for it in items)
    counts = Counter(stream)
    top10 = sum(c for _, c in counts.most_common(10))
    unique_touched = len(counts)
    return {
        "n_items": len(items),
        "n_queries": len(stream),
        "cost_min": costs[0],
        "cost_median": costs[len(costs) // 2],
        "cost_max": costs[-1],
        "cost_p99": costs[int(len(costs) * 0.99)],
        "unique_items_touched": unique_touched,
        "top10_share_pct": 100.0 * top10 / max(len(stream), 1),
    }


def main():
    p = argparse.ArgumentParser(description="Eviction policy benchmark (Zipf workload)")
    p.add_argument("--n-items", type=int, default=5000,
                   help="Size of the unique item pool (default: 5000)")
    p.add_argument("--n-queries", type=int, default=50_000,
                   help="Length of the query stream (default: 50000)")
    p.add_argument("--max-size", type=int, default=1000,
                   help="Cache capacity; must be << n-items to trigger eviction (default: 1000)")
    p.add_argument("--clean-size", type=int, default=None,
                   help="Items evicted per eviction event (default: 20%% of max-size)")
    p.add_argument("--zipf-alpha", type=float, default=1.2,
                   help="Zipf skew (must be > 1; higher = sharper) (default: 1.2)")
    p.add_argument("--expensive-fraction", type=float, default=0.10,
                   help="Fraction of items in the 'expensive' tier (default: 0.10)")
    p.add_argument("--policies", default="LRU,LFU,FIFO,RR",
                   help="Comma-separated policies to compare. "
                        "CA_W_TINYLFU is supported too.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for reproducibility (default: 0)")
    p.add_argument("--workdir", default="bench_eviction_baseline",
                   help="Output directory (default: bench_eviction_baseline)")
    p.add_argument("--out", default=None,
                   help="Path to results JSON (default: <workdir>/results.json)")
    args = p.parse_args()

    if args.clean_size is None:
        args.clean_size = max(1, int(args.max_size * 0.2))
    if args.zipf_alpha <= 1.0:
        raise SystemExit("--zipf-alpha must be > 1 (numpy.random.zipf constraint)")

    print("=" * 72)
    print("Eviction-policy benchmark (Zipf workload, heterogeneous costs)")
    print("=" * 72)
    print(f"  Item pool       : {args.n_items}")
    print(f"  Query stream    : {args.n_queries}")
    print(f"  Cache max_size  : {args.max_size}  "
          f"({100.0 * args.max_size / args.n_items:.1f}% of pool)")
    print(f"  Clean size      : {args.clean_size}  (items evicted per event)")
    print(f"  Zipf alpha      : {args.zipf_alpha}")
    print(f"  Expensive frac  : {args.expensive_fraction}")
    print(f"  Seed            : {args.seed}")
    print(f"  Policies        : {args.policies}")

    items = synthesize_items(args.n_items, args.expensive_fraction, args.seed)
    stream = zipf_query_stream(args.n_items, args.n_queries, args.zipf_alpha, args.seed)
    workload = describe_workload(items, stream)

    print()
    print("  Workload stats:")
    print(f"    cost spread        : min={workload['cost_min']:.0f}  "
          f"med={workload['cost_median']:.0f}  "
          f"p99={workload['cost_p99']:.0f}  "
          f"max={workload['cost_max']:.0f}  "
          f"(unitless utility)")
    print(f"    unique items hit   : {workload['unique_items_touched']} / {args.n_items}  "
          f"({100.0 * workload['unique_items_touched'] / args.n_items:.1f}% of pool)")
    print(f"    top-10 item share  : {workload['top10_share_pct']:.1f}% of queries")
    print()
    print(f"  {'policy':<14}{'hit_rate':>11}{'cost_weighted':>16}"
          f"{'evictions':>12}{'time(s)':>10}")
    print("  " + "-" * 70)

    results = []
    for policy in (s.strip() for s in args.policies.split(",")):
        if not policy:
            continue
        try:
            r = run_policy(policy, args.max_size, items, stream, args.clean_size)
        except ValueError as e:
            print(f"  [skip] {policy}: {e}")
            continue
        results.append(r)
        print(
            f"  {policy:<14}"
            f"{r['hit_rate'] * 100:>10.2f}%"
            f"{r['cost_weighted_hit_rate'] * 100:>15.2f}%"
            f"{r['evictions']:>12}"
            f"{r['elapsed_seconds']:>10.3f}"
        )

    out_path = args.out or os.path.join(args.workdir, "results.json")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "workload": workload,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n  Wrote results to {out_path}")
    print("\n  TIP: pass --policies LRU,LFU,FIFO,CA_W_TINYLFU to compare against the")
    print("       cost-aware policy, and diff the JSON.")


if __name__ == "__main__":
    main()
