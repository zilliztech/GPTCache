"""EmbeddingDispatcher concurrent-throughput benchmark.

Contribution #2 ("Concurrent embedding dispatcher") from directions.md:
EmbeddingDispatcher fans per-request embedding calls across multiprocessing
worker processes so concurrent callers are served in parallel instead of
queueing on a single process.

WHAT THIS MEASURES
Throughput (embeddings/s) and p50/p99 latency under N concurrent callers
(default 1/10/50/100), sequential (today's single-process GPTCache
behavior) vs EmbeddingDispatcher (multiprocess), plus per-run RSS memory
so the worker-duplication cost is measured, not assumed.

HONESTY TRAP (see directions.md)
Each worker duplicates the model in memory (~80MB for MiniLM x N workers) --
we measure and print RSS, not just claim a throughput win. RSS is summed
across the main process AND every live worker child process. Worker startup
(process spawn -- slow on Windows) is a one-time cost paid once when the
pool is created, so every run warms the pool up before timing, matching
real deployment where workers start once at boot.

Usage:
  python examples/benchmark/benchmark_embedding_dispatcher.py --dataset synthetic
  python examples/benchmark/benchmark_embedding_dispatcher.py --dataset ultrachat --n-prompts 200
"""

import argparse
import json
import os
import sys
import threading
import time
from typing import List

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from gptcache.embedding.base import BaseEmbedding
from gptcache.embedding.dispatcher import EmbeddingDispatcher


class _SyntheticEmbedding(BaseEmbedding):
    """No network, no model download -- for a structural smoke test."""

    def __init__(self, dim=32, latency_ms=20.0):
        self._dim = dim
        self._latency_s = latency_ms / 1000.0

    def to_embeddings(self, data, **_):
        import numpy as np
        time.sleep(self._latency_s)
        seed = abs(hash(data)) % (2 ** 32)
        return np.random.default_rng(seed).random(self._dim).astype("float32")

    @property
    def dimension(self):
        return self._dim


def make_synthetic_embedding():
    return _SyntheticEmbedding()


def make_sbert_embedding():
    from gptcache.embedding import SBERT
    return SBERT("all-MiniLM-L6-v2")


def load_prompts(args):
    if args.dataset == "synthetic":
        return [f"synthetic prompt {i}" for i in range(args.n_prompts)]
    from examples.benchmark.benchmark_lmsys import load_ultrachat
    entries = load_ultrachat(args.n_prompts, args.seed)
    if not entries:
        raise SystemExit("No entries loaded -- check dataset access or --n-prompts.")
    return [e.prompt for e in entries]


def get_rss_mb(include_children=False):
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        total = proc.memory_info().rss
        if include_children:
            for child in proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
        return total / (1024 * 1024)
    except ImportError:
        return None


def _split(items, n):
    n = max(1, n)
    k, m = divmod(len(items), n)
    return [items[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n)]


def pct(latencies_ms, p):
    if not latencies_ms:
        return 0.0
    s = sorted(latencies_ms)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def run_sequential(factory, prompts, n_concurrent):
    model = factory()
    latencies = []
    lock = threading.Lock()

    def worker(chunk):
        for p in chunk:
            t0 = time.perf_counter()
            model.to_embeddings(p)
            dt = (time.perf_counter() - t0) * 1000
            with lock:
                latencies.append(dt)

    chunks = _split(prompts, n_concurrent)
    threads = [threading.Thread(target=worker, args=(c,)) for c in chunks]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    rss = get_rss_mb(include_children=False)
    return elapsed, latencies, rss


def run_dispatcher(factory, prompts, n_concurrent, num_workers):
    d = EmbeddingDispatcher(factory, num_workers=num_workers)
    warm = [d.to_embeddings_async("warmup") for _ in range(d.num_workers)]
    for f in warm:
        f.result()

    latencies = []
    lock = threading.Lock()

    def worker(chunk):
        for p in chunk:
            t0 = time.perf_counter()
            d.to_embeddings(p)
            dt = (time.perf_counter() - t0) * 1000
            with lock:
                latencies.append(dt)

    chunks = _split(prompts, n_concurrent)
    threads = [threading.Thread(target=worker, args=(c,)) for c in chunks]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    rss = get_rss_mb(include_children=True)
    d.shutdown()
    return elapsed, latencies, rss, d.num_workers


def main():
    p = argparse.ArgumentParser(description="EmbeddingDispatcher concurrent-throughput benchmark")
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "ultrachat"])
    p.add_argument("--n-prompts", type=int, default=200)
    p.add_argument("--concurrency-levels", default="1,10,50,100")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="bench_embedding_dispatcher/results_ayala.json")
    args = p.parse_args()

    factory = make_synthetic_embedding if args.dataset == "synthetic" else make_sbert_embedding
    prompts_full = load_prompts(args)
    if not prompts_full:
        raise SystemExit("No prompts loaded.")

    levels = [int(x) for x in args.concurrency_levels.split(",")]
    print("=" * 88)
    print("EmbeddingDispatcher concurrent-throughput benchmark")
    print("=" * 88)
    print(f"  Dataset      : {args.dataset}")
    print(f"  Prompt pool  : {len(prompts_full)}")
    print(f"  Concurrency  : {levels}")
    print()

    results = []
    for n_concurrent in levels:
        if len(prompts_full) < n_concurrent:
            prompts = (prompts_full * (n_concurrent // len(prompts_full) + 1))[:n_concurrent]
        else:
            prompts = prompts_full[:max(len(prompts_full), n_concurrent)]

        seq_elapsed, seq_lat, seq_rss = run_sequential(factory, prompts, n_concurrent)
        seq_qps = len(prompts) / seq_elapsed if seq_elapsed > 0 else 0.0

        disp_elapsed, disp_lat, rss, num_workers = run_dispatcher(
            factory, prompts, n_concurrent, args.num_workers)
        disp_qps = len(prompts) / disp_elapsed if disp_elapsed > 0 else 0.0

        speedup = seq_elapsed / disp_elapsed if disp_elapsed > 0 else float("inf")
        row = {
            "concurrency": n_concurrent,
            "n_prompts": len(prompts),
            "num_workers": num_workers,
            "sequential": {
                "elapsed_s": seq_elapsed, "throughput_qps": seq_qps,
                "p50_ms": pct(seq_lat, 50), "p99_ms": pct(seq_lat, 99),
                "rss_mb": seq_rss,
            },
            "dispatcher": {
                "elapsed_s": disp_elapsed, "throughput_qps": disp_qps,
                "p50_ms": pct(disp_lat, 50), "p99_ms": pct(disp_lat, 99),
                "rss_mb": rss,
            },
            "speedup": speedup,
        }
        results.append(row)
        seq_rss_str = f"{seq_rss:.0f}MB" if seq_rss is not None else "n/a"
        rss_str = f"{rss:.0f}MB" if rss is not None else "n/a (psutil not installed)"
        print(f"concurrency={n_concurrent:4d}  "
              f"seq: {seq_qps:7.1f} q/s (p50={pct(seq_lat,50):.1f}ms p99={pct(seq_lat,99):.1f}ms rss={seq_rss_str})  "
              f"disp[{num_workers}w]: {disp_qps:7.1f} q/s (p50={pct(disp_lat,50):.1f}ms p99={pct(disp_lat,99):.1f}ms rss={rss_str})  "
              f"speedup={speedup:.2f}x")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nResults -> {args.out}")


if __name__ == "__main__":
    main()
