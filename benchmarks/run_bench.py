"""Sweep eviction policies across capacities, traces and seeds; emit tidy CSV.

One row per ``(trace, policy, capacity, seed)``. Every policy row is produced by
driving the *real* GPTCache eviction classes through
:class:`benchmarks.simcache.SemanticCacheSim`, so what is measured is shipped
code and not a re-implementation.

Policies
--------
``LRU`` ``LFU`` ``FIFO`` ``RR``
    ``cachetools``-backed, exactly as GPTCache configures them, but with
    ``clean_size=1``.
``ARC``
    The new policy with semantic ghost lists.
``ARC-exact``
    The same code with ``ghost_matching="exact"`` -- classic ARC, whose ghosts
    cannot fire on a semantic workload. This one flag is the ablation.
``LRU-batch``
    GPTCache's *default* configuration: LRU with ``clean_size = 0.2 * maxsize``,
    which drops a fifth of the cache per eviction.

On ``clean_size``
-----------------
GPTCache's default releases ``0.2 * maxsize`` entries at once, so a
cachetools-backed cache spends its life between 80% and 100% full while ARC,
which releases one entry at a time, sits at 100%. Left uncontrolled that
occupancy gap would show up as a hit-rate win for ARC that has nothing to do
with the replacement decision. So the headline comparison pins every
cachetools policy to ``clean_size=1``, and ``LRU-batch`` is carried alongside
to show what the untouched default actually does. ``resident_mean`` is recorded
on every row so the reader can check the control held.

Metrics
-------
hit rate; false hit rate (a hit served from a different ground-truth cluster --
a wrong answer, measurable only on Quora); per-request latency mean/p50/p95/p99;
eviction-decision latency; peak heap; vectors retained by the policy;
throughput; and ARC's ``p`` trajectory.

Peak heap is measured with :mod:`tracemalloc` rather than RSS. RSS on this
workload is dominated by the shared corpus and by allocator behaviour, is not
attributable to the policy, and is not reproducible across runs or parallel
workers; tracemalloc peak is all three. The analytic cost is reported too:
``policy_vectors`` is the number of embeddings the policy retains, which for
ARC is its ghost overhead.

Usage
-----
::

    python benchmarks/prepare_data.py     # once
    python benchmarks/run_bench.py                    # full sweep
    python benchmarks/run_bench.py --quick            # smoke test
    python benchmarks/run_bench.py --jobs 8           # parallel

Writes ``benchmarks/results/bench.csv`` and ``benchmarks/results/p_trace.csv``.
"""

import argparse
import csv
import itertools
import os
import platform
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traces  # noqa: E402
from simcache import SemanticCacheSim  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# name -> (policy passed to GPTCache, extra kwargs)
POLICIES = {
    "LRU":       ("LRU", {"clean_size": 1}),
    "LFU":       ("LFU", {"clean_size": 1}),
    "FIFO":      ("FIFO", {"clean_size": 1}),
    "RR":        ("RR", {"clean_size": 1}),
    "ARC":       ("ARC", {"ghost_matching": "semantic"}),
    "ARC-exact": ("ARC", {"ghost_matching": "exact"}),
    "LRU-batch": ("LRU", {}),          # clean_size defaults to 0.2 * maxsize
}

CAPACITIES = [50, 100, 200, 400, 800, 1600]
REGIMES = ["quora-stationary", "quora-drift", "wildchat"]
N_SEEDS = 10
N_QUERIES = 20_000

FIELDS = [
    "trace", "policy", "capacity", "seed", "n_queries",
    "hit_rate", "false_hit_rate", "n_hits", "n_false_hits",
    "lat_mean_us", "lat_p50_us", "lat_p95_us", "lat_p99_us",
    "evict_us_per_req", "throughput_qps", "wall_seconds",
    "peak_heap_kb", "policy_vectors", "resident_mean", "n_evicted",
    "p_final", "p_max", "p_mean", "tau",
]


def run_one(regime, policy_name, capacity, seed, n_queries, tau, keep_p=False):
    """Run a single ``(trace, policy, capacity, seed)`` cell."""
    queries, cids = traces.build_trace(regime, n_queries, seed)
    policy, kwargs = POLICIES[policy_name]
    kwargs = dict(kwargs)   # SemanticCacheSim mutates it

    lat = np.empty(len(queries), dtype=np.float64)
    resident = np.empty(len(queries), dtype=np.int32)
    p_trace = [] if keep_p else None
    hits = false_hits = 0
    # p is summarised on every row; the full trajectory is kept only for the
    # ablation figure, since storing 20k floats per cell would dominate the CSV
    p_max = p_sum = 0.0
    p_seen = False

    tracemalloc.start()
    base_heap = tracemalloc.get_traced_memory()[0]
    sim = SemanticCacheSim(policy, capacity, tau, queries.shape[1], **kwargs)

    wall0 = time.perf_counter()
    for i in range(len(queries)):
        t0 = time.perf_counter()
        hit, false_hit = sim.request(queries[i], int(cids[i]))
        lat[i] = (time.perf_counter() - t0) * 1e6
        hits += hit
        false_hits += false_hit
        resident[i] = len(sim._id_at)  # pylint: disable=protected-access
        p_now = sim.p
        if p_now is not None:
            p_seen = True
            p_sum += p_now
            if p_now > p_max:
                p_max = p_now
            if keep_p:
                p_trace.append(p_now)
    wall = time.perf_counter() - wall0

    peak_heap = tracemalloc.get_traced_memory()[1] - base_heap
    tracemalloc.stop()

    n = len(queries)
    row = {
        "trace": regime, "policy": policy_name, "capacity": capacity,
        "seed": seed, "n_queries": n,
        "hit_rate": hits / n,
        "false_hit_rate": (false_hits / hits) if hits else 0.0,
        "n_hits": hits, "n_false_hits": false_hits,
        "lat_mean_us": float(lat.mean()),
        "lat_p50_us": float(np.percentile(lat, 50)),
        "lat_p95_us": float(np.percentile(lat, 95)),
        "lat_p99_us": float(np.percentile(lat, 99)),
        "evict_us_per_req": sim.evict_seconds / n * 1e6,
        "throughput_qps": n / wall,
        "wall_seconds": wall,
        "peak_heap_kb": peak_heap / 1024.0,
        "policy_vectors": sim.n_vectors,
        "resident_mean": float(resident.mean()),
        "n_evicted": sim.evicted,
        "p_final": sim.p if p_seen else "",
        "p_max": p_max if p_seen else "",
        "p_mean": (p_sum / n) if p_seen else "",
        "tau": tau,
    }
    return row, p_trace


def _worker(args):
    regime, policy, cap, seed, n_queries, tau = args
    row, _ = run_one(regime, policy, cap, seed, n_queries, tau)
    return row


# quantile grid dense enough to draw a latency CDF without storing 20k samples
CDF_Q = ([0.1, 0.5, 1, 2] + list(range(5, 100, 5)) + [98, 99, 99.5, 99.9])


def emit_latency_cdf(path, regime, capacity, n_queries, tau, seeds=3):
    """Per-policy latency quantile grids at one capacity, for the CDF figure."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trace", "policy", "capacity", "seed", "quantile",
                    "latency_us"])
        for pol in POLICIES:
            for seed in range(seeds):
                queries, cids = traces.build_trace(regime, n_queries, seed)
                policy, kwargs = POLICIES[pol]
                sim = SemanticCacheSim(policy, capacity, tau,
                                       queries.shape[1], **dict(kwargs))
                lat = np.empty(len(queries))
                for i in range(len(queries)):
                    t0 = time.perf_counter()
                    sim.request(queries[i], int(cids[i]))
                    lat[i] = (time.perf_counter() - t0) * 1e6
                for q, v in zip(CDF_Q, np.percentile(lat, CDF_Q)):
                    w.writerow([regime, pol, capacity, seed, q, float(v)])
    print(f"wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="tiny sweep for smoke-testing the harness")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes")
    ap.add_argument("--n-queries", type=int, default=N_QUERIES)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "bench.csv"))
    args = ap.parse_args(argv)

    traces.silence_spurious_fp_warnings()
    if not traces.data_available():
        print("benchmarks/data/ is empty -- run prepare_data.py first.")
        return 2

    tau = traces.load_tau()
    capacities = CAPACITIES
    regimes = REGIMES
    policies = list(POLICIES)
    n_queries, n_seeds = args.n_queries, args.seeds
    if args.quick:
        capacities = [100, 400]
        regimes = ["quora-drift"]
        n_queries, n_seeds = 3000, 2

    cells = list(itertools.product(regimes, policies, capacities,
                                   range(n_seeds)))
    print(f"tau        : {tau}")
    print(f"queries    : {n_queries}")
    print(f"traces     : {regimes}")
    print(f"policies   : {policies}")
    print(f"capacities : {capacities}")
    print(f"seeds      : {n_seeds}")
    print(f"cells      : {len(cells)}   jobs: {args.jobs}")
    print(f"platform   : {platform.platform()} / {platform.processor()}")

    jobs = [(r, p, c, s, n_queries, tau) for r, p, c, s in cells]
    rows = []
    t0 = time.perf_counter()

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for i, row in enumerate(pool.map(_worker, jobs, chunksize=1), 1):
                rows.append(row)
                if i % 20 == 0 or i == len(jobs):
                    el = time.perf_counter() - t0
                    print(f"  {i}/{len(jobs)}  {el:6.1f}s elapsed, "
                          f"eta {el / i * (len(jobs) - i):6.1f}s")
    else:
        for i, job in enumerate(jobs, 1):
            rows.append(_worker(job))
            if i % 20 == 0 or i == len(jobs):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(jobs)}  {el:6.1f}s elapsed, "
                      f"eta {el / i * (len(jobs) - i):6.1f}s")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}  ({len(rows)} rows)")

    # ---- p over time, for the ablation figure -------------------------
    p_path = os.path.join(RESULTS_DIR, "p_trace.csv")
    cap = 400 if 400 in capacities else capacities[-1]
    regime = "quora-drift" if "quora-drift" in regimes else regimes[0]
    with open(p_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trace", "policy", "capacity", "seed", "step", "p"])
        for pol in ("ARC", "ARC-exact"):
            _, trace = run_one(regime, pol, cap, 0, n_queries, tau, keep_p=True)
            # thin to ~2000 points; the figure cannot resolve more
            step = max(1, len(trace) // 2000)
            for i in range(0, len(trace), step):
                w.writerow([regime, pol, cap, 0, i, trace[i]])
    print(f"wrote {p_path}")

    emit_latency_cdf(os.path.join(RESULTS_DIR, "lat_cdf.csv"),
                     regime, cap, n_queries, tau,
                     seeds=min(3, n_seeds))
    print(f"\ntotal {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
