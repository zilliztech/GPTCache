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

Time is measured twice, and only one of the two is quotable. Every sweep cell
carries ``lat_*``, ``throughput_qps`` and ``evict_us_per_req``, but the sweep
runs its cells through a process pool, so those columns absorb whatever CPU
contention ``--jobs`` created -- across runs of this harness the same cell has
varied roughly threefold on time while its hit rate held to four decimals.
``timing_pass`` therefore re-measures the cost-table cells one at a time in a
single process, into ``timing.csv``, and that is what the cost table and the
report quote. Hit rate is immune to all of this: it is deterministic.

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
    python benchmarks/run_bench.py --timing-only      # cost cells only

Writes ``benchmarks/results/bench.csv``, ``p_trace.csv``, ``timing.csv`` and
``lat_cdf.csv``.
"""

import argparse
import csv
import itertools
import os
import platform
import random
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traces  # noqa: E402
from simcache import SemanticCacheSim  # noqa: E402

# Overridable so a reduced run (smoke/quick) cannot overwrite the committed
# reference artifacts; run_pipeline.sh points those at a scratch subdirectory.
RESULTS_DIR = (os.environ.get("BENCH_RESULTS_DIR")
               or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))

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
# the wikianswers pair is built by prepare_data_ext.py rather than
# prepare_data.py, so it is dropped automatically when that has not been run
REGIMES = ["quora-stationary", "quora-drift", "wildchat",
           "wikianswers-stationary", "wikianswers-drift"]
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


def run_one(regime, policy_name, capacity, seed, n_queries, tau, keep_p=False,
            keep_lat=False):
    """Run a single ``(trace, policy, capacity, seed)`` cell.

    Returns ``(row, p_trace, lat)``; ``p_trace`` and ``lat`` are ``None`` unless
    the corresponding ``keep_*`` flag is set.
    """
    queries, cids = traces.build_trace(regime, n_queries, seed)
    policy, kwargs = POLICIES[policy_name]
    kwargs = dict(kwargs)   # SemanticCacheSim mutates it
    if policy == "RR":
        # cachetools.RRCache defaults to the unseeded global `random`, which is
        # the only source of run-to-run variation in this harness. Seeding it
        # per cell makes the whole sweep reproducible; RR stays random *within*
        # a run, which is the property being benchmarked.
        kwargs["choice"] = random.Random(seed).choice

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
    return row, p_trace, (lat if keep_lat else None)


def _worker(args):
    regime, policy, cap, seed, n_queries, tau = args
    row, _, _ = run_one(regime, policy, cap, seed, n_queries, tau)
    return row


# quantile grid dense enough to draw a latency CDF without storing 20k samples
CDF_Q = ([0.1, 0.5, 1, 2] + list(range(5, 100, 5)) + [98, 99, 99.5, 99.9])

# capacities the cost table reports. The sweep covers all six, but its timing
# columns are measured inside a worker pool and are therefore a function of
# --jobs; only this pass is uncontended, so it is kept small on purpose.
TIMING_CAPACITIES = [400, 1600]


def timing_pass(timing_path, cdf_path, regimes, capacities, n_queries, tau,
                seeds=3, cdf_regime=None, cdf_capacity=None):
    """Latency, throughput and eviction cost, measured one cell at a time.

    The sweep runs its cells through a process pool, so its ``lat_*``,
    ``throughput_qps`` and ``evict_us_per_req`` columns carry however much CPU
    contention ``--jobs`` created and are not a property of the policy. This
    pass re-measures the cost-table cells in a single process, and emits the
    CDF grid from the very same latency arrays so the two artifacts cannot
    drift apart. Hit rate is unaffected either way -- it is deterministic.
    """
    caps = [c for c in TIMING_CAPACITIES if c in capacities] or [capacities[-1]]
    if cdf_capacity not in caps:
        caps.append(cdf_capacity)
    cells = list(itertools.product(regimes, caps, POLICIES, range(seeds)))
    print(f"timing pass: {len(cells)} cells at capacities {caps}, "
          f"single process")

    cdf_rows = []
    t0 = time.perf_counter()
    with open(timing_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i, (regime, cap, pol, seed) in enumerate(cells, 1):
            row, _, lat = run_one(regime, pol, cap, seed, n_queries, tau,
                                  keep_lat=True)
            w.writerow(row)
            if regime == cdf_regime and cap == cdf_capacity:
                for q, v in zip(CDF_Q, np.percentile(lat, CDF_Q)):
                    cdf_rows.append([regime, pol, cap, seed, q, float(v)])
            if i % 20 == 0 or i == len(cells):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(cells)}  {el:6.1f}s elapsed, "
                      f"eta {el / i * (len(cells) - i):6.1f}s")
    print(f"wrote {timing_path}  ({len(cells)} rows)")

    with open(cdf_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trace", "policy", "capacity", "seed", "quantile",
                    "latency_us"])
        w.writerows(cdf_rows)
    print(f"wrote {cdf_path}  ({len(cdf_rows)} rows)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="tiny sweep for smoke-testing the harness")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes")
    ap.add_argument("--timing-only", action="store_true",
                    help="re-measure only the cost-table cells (timing.csv "
                         "and lat_cdf.csv), leaving bench.csv alone")
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

    # a clean clone runs prepare_data.py only, so the optional corpora may be
    # absent; skip loudly rather than crashing halfway through the sweep
    missing = [r for r in regimes if not traces.regime_available(r)]
    if missing:
        regimes = [r for r in regimes if r not in missing]
        print(f"skipping   : {missing}\n"
              f"             (run prepare_data_ext.py to include them)")

    cdf_cap = 400 if 400 in capacities else capacities[-1]
    cdf_regime = "quora-drift" if "quora-drift" in regimes else regimes[0]

    if args.timing_only:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        timing_pass(os.path.join(RESULTS_DIR, "timing.csv"),
                    os.path.join(RESULTS_DIR, "lat_cdf.csv"),
                    regimes, capacities, n_queries, tau,
                    seeds=min(3, n_seeds), cdf_regime=cdf_regime,
                    cdf_capacity=cdf_cap)
        return 0

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
    cap, regime = cdf_cap, cdf_regime
    with open(p_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trace", "policy", "capacity", "seed", "step", "p"])
        for pol in ("ARC", "ARC-exact"):
            _, trace, _ = run_one(regime, pol, cap, 0, n_queries, tau,
                                  keep_p=True)
            # thin to ~2000 points; the figure cannot resolve more
            step = max(1, len(trace) // 2000)
            for i in range(0, len(trace), step):
                w.writerow([regime, pol, cap, 0, i, trace[i]])
    print(f"wrote {p_path}")

    timing_pass(os.path.join(RESULTS_DIR, "timing.csv"),
                os.path.join(RESULTS_DIR, "lat_cdf.csv"),
                regimes, capacities, n_queries, tau,
                seeds=min(3, n_seeds), cdf_regime=regime, cdf_capacity=cap)
    print(f"\ntotal {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
