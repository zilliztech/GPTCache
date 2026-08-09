"""Locate the traffic conditions under which ARC actually beats LRU.

``run_bench.py`` answers "which policy wins on these three workloads". This
script answers the prior question: *what property of a workload decides it*.
Drift is treated as a continuous variable rather than a binary, so the
break-even contour can be read off directly instead of inferred from two
endpoints.

Three experiments, all driving the real GPTCache eviction classes through
:class:`simcache.SemanticCacheSim` exactly as ``run_bench.py`` does, with the
same ``clean_size=1`` occupancy control.

``drift-skew``
    ARC - LRU across a grid of drift rate x popularity skew, at fixed
    capacity. This is the headline: the zero contour is the crossover.
``drift-capacity``
    ARC - LRU across drift rate x capacity at fixed skew, which says how much
    of the answer is "your cache is too small" rather than "your traffic
    drifts".
``real``
    Every policy over the real, timestamp-ordered prompt streams at every
    capacity -- including ``wildchat-long``, which spans the full WildChat
    release window rather than two months, so whatever drift it contains is
    drift that actually happened.

Usage
-----
::

    python benchmarks/sweep_crossover.py                 # all three
    python benchmarks/sweep_crossover.py --exp drift-skew
    python benchmarks/sweep_crossover.py --quick

Writes ``benchmarks/results/crossover_{experiment}.csv``.
"""

import argparse
import csv
import itertools
import multiprocessing as mp
import os
import platform
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traces  # noqa: E402
import traces_param  # noqa: E402
from run_bench import POLICIES  # noqa: E402
from simcache import SemanticCacheSim  # noqa: E402

# Overridable so a reduced run (smoke/quick) cannot overwrite the committed
# reference artifacts; run_pipeline.sh points those at a scratch subdirectory.
RESULTS_DIR = (os.environ.get("BENCH_RESULTS_DIR")
               or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))

N_QUERIES = 20_000
N_SEEDS = 5
N_SEEDS_REAL = 10

DRIFTS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
SKEWS = [0.7, 0.9, 1.1, 1.3, 1.6]
CAPACITIES = [50, 100, 200, 400, 800, 1600]

SYNTH_POLICIES = ["LRU", "ARC", "LFU", "ARC-exact"]
STRIDES = [1, 2, 3, 4]
GRID_CAPACITY = 200
GRID_SKEW = 1.1
N_EPOCHS = 6

FIELDS = [
    "experiment", "corpus", "policy", "capacity", "seed", "n_queries",
    "drift_rate", "zipf_s", "n_epochs", "stride",
    "hit_rate", "false_hit_rate", "n_hits",
    "lat_mean_us", "evict_us_per_req",
    "policy_vectors", "resident_mean", "p_final", "p_mean", "tau",
]


def _simulate(queries, cids, policy_name, capacity, tau, seed):
    """One (trace, policy, capacity) cell. Returns a metrics dict."""
    policy, kwargs = POLICIES[policy_name]
    kwargs = dict(kwargs)
    if policy == "RR":
        kwargs["choice"] = random.Random(seed).choice

    sim = SemanticCacheSim(policy, capacity, tau, queries.shape[1], **kwargs)
    n = len(queries)
    lat = np.empty(n, dtype=np.float64)
    resident = np.empty(n, dtype=np.int32)
    hits = false_hits = 0
    p_sum = 0.0
    p_seen = False

    for i in range(n):
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

    return {
        "hit_rate": hits / n,
        "false_hit_rate": (false_hits / hits) if hits else 0.0,
        "n_hits": hits,
        "lat_mean_us": float(lat.mean()),
        "evict_us_per_req": sim.evict_seconds / n * 1e6,
        "policy_vectors": sim.n_vectors,
        "resident_mean": float(resident.mean()),
        "p_final": sim.p if p_seen else "",
        "p_mean": (p_sum / n) if p_seen else "",
    }


def _worker(job):
    exp, corpus, policy, cap, seed, drift, skew, n_epochs, n_q, tau, stride = job
    if exp in ("real", "density"):
        queries, cids = traces_param.build_real_trace(corpus, n_q, seed,
                                                      stride=stride)
    else:
        queries, cids = traces_param.build_param_trace(
            corpus, n_q, seed, zipf_s=skew, drift_rate=drift,
            n_epochs=n_epochs)
    row = {
        "experiment": exp, "corpus": corpus, "policy": policy,
        "capacity": cap, "seed": seed, "n_queries": len(queries),
        "drift_rate": drift, "zipf_s": skew, "n_epochs": n_epochs,
        "stride": stride, "tau": tau,
    }
    row.update(_simulate(queries, cids, policy, cap, tau, seed))
    return row


def _jobs_drift_skew(corpora, n_q, tau, seeds, caps):
    for corpus, drift, skew, pol, cap, seed in itertools.product(
            corpora, DRIFTS, SKEWS, SYNTH_POLICIES, caps, range(seeds)):
        yield ("drift-skew", corpus, pol, cap, seed, drift, skew,
               N_EPOCHS, n_q, tau, 1)


def _jobs_drift_capacity(corpora, n_q, tau, seeds, caps):
    for corpus, drift, pol, cap, seed in itertools.product(
            corpora, DRIFTS, SYNTH_POLICIES, caps, range(seeds)):
        yield ("drift-capacity", corpus, pol, cap, seed, drift, GRID_SKEW,
               N_EPOCHS, n_q, tau, 1)


def _jobs_real(corpora, n_q, tau, seeds, caps):
    for corpus, pol, cap, seed in itertools.product(
            corpora, POLICIES, caps, range(seeds)):
        yield ("real", corpus, pol, cap, seed, "", "", "", n_q, tau, 1)


def _jobs_density(n_q, tau, seeds, caps):
    """Control for ``wildchat-long``: thin the *short* stream and re-measure.

    ``wildchat-long`` spans a year but is subsampled to 150k, so it is both
    longer and sparser than ``wildchat``. If sparsity alone moves ARC, the
    extra span cannot be credited for the difference.
    """
    for pol, cap, seed, stride in itertools.product(
            ["LRU", "ARC", "LFU"], caps, range(seeds), STRIDES):
        yield ("density", "wildchat", pol, cap, seed, "", "", "", n_q, tau,
               stride)


# Rows are sorted before writing rather than streamed out in completion order.
# Every cell is individually deterministic, but imap_unordered returns them in
# whatever order the workers finish, which made the CSV's byte content depend on
# scheduling. Sorting costs nothing at these sizes (<= 3240 rows) and makes the
# file diffable against a previous run.
_SORT_KEY = ("experiment", "corpus", "policy", "capacity", "seed",
             "drift_rate", "zipf_s", "stride")


def _row_sort_key(row):
    # Numeric columns sort numerically (capacity 50 before 100, not "100"
    # before "50"); the tag keeps mixed types from being compared.
    key = []
    for name in _SORT_KEY:
        value = row.get(name, "")
        try:
            key.append((0, float(value), ""))
        except (TypeError, ValueError):
            key.append((1, 0.0, str(value)))
    return tuple(key)


def run(jobs, path, n_jobs):
    jobs = list(jobs)
    print(f"  {len(jobs)} cells -> {os.path.basename(path)}")
    t0 = time.perf_counter()
    rows = []
    if n_jobs > 1:
        with mp.Pool(n_jobs) as pool:
            for i, row in enumerate(
                    pool.imap_unordered(_worker, jobs, chunksize=4), 1):
                rows.append(row)
                if i % 200 == 0 or i == len(jobs):
                    el = time.perf_counter() - t0
                    print(f"    {i}/{len(jobs)}  {el:.0f}s "
                          f"(eta {el / i * (len(jobs) - i):.0f}s)",
                          flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            rows.append(_worker(job))
            if i % 100 == 0:
                print(f"    {i}/{len(jobs)}", flush=True)

    rows.sort(key=_row_sort_key)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  done in {time.perf_counter() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", choices=["drift-skew", "drift-capacity", "real",
                                      "density"], action="append")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--n-queries", type=int, default=N_QUERIES)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    traces.silence_spurious_fp_warnings()
    if not traces.data_available():
        sys.exit("no corpora: run benchmarks/prepare_data.py first")

    tau = traces.load_tau()
    n_q, seeds = args.n_queries, args.seeds
    caps = CAPACITIES
    cluster_corpora = [c for c in traces_param.CLUSTER_CORPORA
                       if os.path.exists(os.path.join(
                           traces.DATA_DIR, f"{c}_emb.npy"))]
    real_corpora = [c for c in traces_param.REAL_CORPORA
                    if os.path.exists(os.path.join(
                        traces.DATA_DIR, f"{c}_emb.npy"))]
    if args.quick:
        n_q, seeds, caps = 3000, 2, [100, 400]

    exps = args.exp or ["drift-skew", "drift-capacity", "real", "density"]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"tau        : {tau}")
    print(f"queries    : {n_q}   seeds: {seeds}   jobs: {args.jobs}")
    print(f"cluster    : {cluster_corpora}")
    print(f"real       : {real_corpora}")
    print(f"python     : {platform.python_version()}  numpy {np.__version__}")

    if "drift-skew" in exps:
        print("\n[drift-skew] ARC-LRU over drift x skew "
              f"at capacity {GRID_CAPACITY}")
        run(_jobs_drift_skew(cluster_corpora, n_q, tau, seeds,
                             [GRID_CAPACITY] if not args.quick else caps[:1]),
            os.path.join(RESULTS_DIR, "crossover_drift_skew.csv"), args.jobs)

    if "drift-capacity" in exps:
        print(f"\n[drift-capacity] ARC-LRU over drift x capacity "
              f"at skew {GRID_SKEW}")
        run(_jobs_drift_capacity(cluster_corpora, n_q, tau, seeds, caps),
            os.path.join(RESULTS_DIR, "crossover_drift_capacity.csv"),
            args.jobs)

    if "real" in exps:
        print("\n[real] all policies over real timestamp-ordered streams")
        run(_jobs_real(real_corpora, n_q, tau,
                       min(N_SEEDS_REAL, seeds * 2), caps),
            os.path.join(RESULTS_DIR, "crossover_real.csv"), args.jobs)

    if "density" in exps:
        print("\n[density] control: thin the 2-month stream in time")
        run(_jobs_density(n_q, tau, min(N_SEEDS_REAL, seeds * 2), caps),
            os.path.join(RESULTS_DIR, "crossover_density.csv"), args.jobs)


if __name__ == "__main__":
    main()
