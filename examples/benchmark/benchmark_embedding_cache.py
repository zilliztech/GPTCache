"""Per-query embedding-cache benchmark.

Contribution #6 from directions.md: CachedEmbedding, an LRU wrapper around
any embedding backend that skips recomputation when the *exact same* text
is embedded twice.

WHY THIS IS A SEPARATE SCRIPT FROM benchmark_lmsys.py
-------------------------------------------------------
benchmark_lmsys.py computes all embeddings ONCE up front with a single
batched `model.encode(texts)` call, then looks them up by index during the
replay loop. That is a legitimate optimization of the *benchmark harness*,
but it means the harness never calls `to_embeddings()` per incoming query --
so it cannot exercise (or measure) a per-query embedding cache at all.

In real GPTCache usage, `embedding_func` IS called fresh for every incoming
request, including exact repeats of an earlier request. This script
reproduces that call pattern faithfully: it walks a query stream one prompt
at a time and calls `to_embeddings()` per query, once with a plain encoder
and once with the encoder wrapped in CachedEmbedding.

HONESTY TRAP (see directions.md)
---------------------------------
The benefit of this cache is capped by how often the *exact same string*
recurs in the stream -- GPTCache's own vector search already handles
near-duplicates, so this cache does nothing for paraphrases. We do not
assume a savings number: `--dataset synthetic` lets you set the exact-repeat
ratio directly for a controlled sanity check, and for real data we print the
measured exact-duplicate ratio of the stream before reporting any speedup,
so a low number here is expected on some corpora and should be reported as
such, not hidden.

Usage:
  # no network required, no model download -- structural smoke test
  python examples/benchmark/benchmark_embedding_cache.py --dataset synthetic

  # real data + real SBERT model (requires sentence-transformers, and HF
  # access if --dataset lmsys/wildchat)
  python examples/benchmark/benchmark_embedding_cache.py --dataset ultrachat --n-queries 2000
  python examples/benchmark/benchmark_embedding_cache.py --dataset ultrachat --drift
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Ensure repo root is importable when run directly
# ---------------------------------------------------------------------------
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from gptcache.embedding.base import BaseEmbedding
from gptcache.embedding.cached_embedding import CachedEmbedding


# ---------------------------------------------------------------------------
# Synthetic dataset -- no network, no model download. Lets you set the exact
# repeat ratio directly, and sanity-check the harness before spending time on
# a real HF download.
# ---------------------------------------------------------------------------
def make_synthetic_prompts(n: int, repeat_ratio: float, seed: int) -> List[str]:
    """Build n prompts where ~repeat_ratio of them are exact repeats of an
    earlier prompt in the pool (rest are unique)."""
    rng = np.random.default_rng(seed)
    n_unique = max(1, int(n * (1 - repeat_ratio)))
    pool = [f"synthetic user prompt number {i} about topic {i % 37}" for i in range(n_unique)]
    prompts = []
    for _ in range(n):
        if prompts and rng.random() < repeat_ratio:
            prompts.append(prompts[rng.integers(0, len(prompts))])
        else:
            prompts.append(pool[rng.integers(0, len(pool))])
    return prompts


class _DeterministicFakeEmbedding(BaseEmbedding):
    """Stand-in encoder used only with --dataset synthetic, so the whole
    harness can be smoke-tested with no model download and no network."""

    def __init__(self, dim: int = 32, latency_ms: float = 2.0):
        self._dim = dim
        self._latency_s = latency_ms / 1000.0

    def to_embeddings(self, data, **_):
        time.sleep(self._latency_s)  # simulate real encoder cost
        seed = abs(hash(data)) % (2 ** 32)
        rng = np.random.default_rng(seed)
        return rng.random(self._dim).astype("float32")

    @property
    def dimension(self) -> int:
        return self._dim


def load_prompts(args) -> List[str]:
    if args.dataset == "synthetic":
        return make_synthetic_prompts(args.n_queries, args.synthetic_repeat_ratio, args.seed)

    # Reuse the real loaders + drift stream from benchmark_lmsys.py verbatim,
    # so we exercise the same corpora/skew the eviction-policy paper used.
    from examples.benchmark.benchmark_lmsys import (
        load_lmsys, load_wildchat, load_ultrachat, drift_query_stream,
    )
    loader = {"lmsys": load_lmsys, "wildchat": load_wildchat}.get(
        args.dataset, load_ultrachat)
    entries = loader(args.n_queries, args.seed)
    if not entries:
        raise SystemExit("No entries loaded -- check dataset access or --n-queries.")

    if args.drift:
        order = drift_query_stream(
            n_items=len(entries), n_queries=args.drift_queries,
            alpha=args.drift_zipf, rotate=args.drift_rotate, seed=args.seed,
            shift_frac=args.drift_shift,
        )
        return [entries[i].prompt for i in order]
    return [e.prompt for e in entries]


def build_encoder(args) -> BaseEmbedding:
    if args.dataset == "synthetic":
        return _DeterministicFakeEmbedding()
    from gptcache.embedding import SBERT
    return SBERT(args.embed_model)


def time_encode_stream(encoder: BaseEmbedding, prompts: List[str]) -> dict:
    start = time.perf_counter()
    for p in prompts:
        encoder.to_embeddings(p)
    elapsed = time.perf_counter() - start
    return {
        "total_seconds": elapsed,
        "avg_ms_per_query": (elapsed / len(prompts)) * 1000,
    }


def main():
    p = argparse.ArgumentParser(description="Per-query embedding-cache benchmark")
    p.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "ultrachat", "lmsys", "wildchat"],
                    help="synthetic = no network/model needed, for a structural "
                         "smoke test; others reuse benchmark_lmsys.py loaders "
                         "(default: synthetic)")
    p.add_argument("--n-queries", type=int, default=2000)
    p.add_argument("--synthetic-repeat-ratio", type=float, default=0.30,
                    help="Only used with --dataset synthetic: fraction of "
                         "queries that are exact repeats (default: 0.30)")
    p.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    p.add_argument("--cache-size", type=int, default=10_000)
    p.add_argument("--drift", action="store_true",
                    help="Replay a Zipf-skewed, hot-set-rotating stream "
                         "(reuses benchmark_lmsys.drift_query_stream) instead "
                         "of touching each entry once. Only for real datasets.")
    p.add_argument("--drift-queries", type=int, default=10_000)
    p.add_argument("--drift-zipf", type=float, default=1.2)
    p.add_argument("--drift-rotate", type=int, default=3000)
    p.add_argument("--drift-shift", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workdir", default="bench_embedding_cache")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    print("=" * 72)
    print("Per-query embedding-cache benchmark")
    print("=" * 72)
    print(f"  Dataset          : {args.dataset}")
    print(f"  Queries          : {args.n_queries if args.dataset == 'synthetic' else '(see loaded count below)'}")
    print(f"  Cache size       : {args.cache_size}")

    prompts = load_prompts(args)
    n = len(prompts)
    dup_ratio = 1 - (len(set(prompts)) / n)
    print(f"\n  Loaded {n} queries")
    print(f"  Measured exact-duplicate ratio in stream: {dup_ratio*100:.1f}%")
    print("  (this is the hard ceiling on possible cache benefit -- a low "
          "number here means the corpus has few exact repeats, and that is "
          "a valid, reportable result, not a bug)")

    # ---- Baseline: no cache ----
    print(f"\nRunning baseline (no cache) ...")
    baseline_encoder = build_encoder(args)
    baseline = time_encode_stream(baseline_encoder, prompts)
    print(f"  total: {baseline['total_seconds']:.2f}s  "
          f"avg: {baseline['avg_ms_per_query']:.3f} ms/query")

    # ---- With CachedEmbedding ----
    print(f"\nRunning with CachedEmbedding(cache_size={args.cache_size}) ...")
    inner_encoder = build_encoder(args)
    cached_encoder = CachedEmbedding(inner_encoder, cache_size=args.cache_size)
    with_cache = time_encode_stream(cached_encoder, prompts)
    stats = cached_encoder.stats
    print(f"  total: {with_cache['total_seconds']:.2f}s  "
          f"avg: {with_cache['avg_ms_per_query']:.3f} ms/query")
    print(f"  cache hit_rate: {stats['hit_rate']*100:.1f}%  "
          f"(hits={stats['hits']} misses={stats['misses']})")

    speedup = (baseline["total_seconds"] / with_cache["total_seconds"]
               if with_cache["total_seconds"] > 0 else float("inf"))
    print(f"\n  Wall-clock speedup: {speedup:.2f}x")

    # ---- Save ----
    out_path = args.out or os.path.join(args.workdir, "results.json")
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args),
            "n_queries": n,
            "exact_duplicate_ratio": dup_ratio,
            "baseline": baseline,
            "with_cache": with_cache,
            "cache_stats": stats,
            "speedup": speedup,
        }, f, indent=2)
    print(f"\n  Results -> {out_path}")


if __name__ == "__main__":
    main()