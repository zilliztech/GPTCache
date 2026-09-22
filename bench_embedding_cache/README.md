# Embedding-Cache Benchmark — How to Run

This directory holds the benchmark for `CachedEmbedding`
(`gptcache/embedding/cached_embedding.py`), an LRU wrapper that skips
re-computing an embedding when the exact same text has already been
embedded once.

## What it measures

Whether wrapping an embedding backend in `CachedEmbedding` reduces
wall-clock time and model calls, on a query stream that faithfully
reproduces per-query calls the way live GPTCache traffic does (unlike
`examples/benchmark/benchmark_lmsys.py`, which pre-computes embeddings once
in a batch -- see `docs/writeup.md` section 10.3 for why that harness can't be
reused for this).

## Setup

pip install -r examples/benchmark/requirements.txt

This installs tiktoken, datasets, and sentence-transformers -- extra
dependencies needed only for the benchmark scripts, not the core library.

## Quick smoke test (no network, no model download, ~1 second)

python examples/benchmark/benchmark_embedding_cache.py --dataset synthetic

Validates the harness end-to-end with a synthetic prompt stream at a known
repeat ratio. Useful for confirming your environment is set up correctly
before spending time on a real download.

## Real-data run (single pass, no query repeats)

python examples/benchmark/benchmark_embedding_cache.py --dataset ultrachat --n-queries 2000

Downloads all-MiniLM-L6-v2 (SBERT) and a slice of UltraChat-200K (public,
no login) on first run. Each prompt is distinct, so the measured
exact-duplicate ratio is ~0% and the cache correctly shows no benefit -- a
valid negative result, not a bug (see "Honesty trap" below).

## Real-data run with realistic repeat traffic (the main result)

python examples/benchmark/benchmark_embedding_cache.py --dataset ultrachat --drift --drift-queries 5000 --seed 0 --out bench_embedding_cache/results_seed0.json

--drift replays a Zipf-skewed, hot-set-rotating query stream (the same
generator benchmark_lmsys.py uses for the eviction-policy experiments),
so popular prompts recur -- the traffic shape a live cache actually sees.

## Reproducing the paper's 7-seed result

Run this once per seed (0 through 6), changing --seed and --out each time:

python examples/benchmark/benchmark_embedding_cache.py --dataset ultrachat --drift --drift-queries 5000 --seed 0 --out bench_embedding_cache/results_seed0.json

Then:

python bench_embedding_cache/paired_stats.py "bench_embedding_cache/results_seed*.json"

paired_stats.py reports, per seed, the measured exact-duplicate ratio,
cache hit rate, and speedup, plus an exact Wilcoxon signed-rank p-value
across seeds (same statistical protocol as bench_lmsys/paired.py).

Note on wall-clock noise: absolute times are sensitive to what else is
running on the machine (background sync, other processes). Run on an
otherwise-idle machine if possible. The paired, within-seed comparison
(baseline and cached measured back-to-back in the same run) is what the
significance claim relies on -- not the absolute magnitude of any single
seed's numbers.

## Honesty trap

CachedEmbedding only ever helps when the exact same string recurs -- it
is not a semantic/near-duplicate cache (that's the vector search's job).
Every run prints the measured exact-duplicate ratio of its own query
stream before reporting a speedup, so a low number on a low-repeat corpus
is an expected, reportable result, not a failure.

## Output files

Each run writes a JSON log (--out, default
bench_embedding_cache/results.json) with the full config, measured
duplicate ratio, per-arm timing, cache stats, and speedup -- sample logs
from the seeds above are committed in this directory for reference.

## Unit tests

python -m pytest tests/unit_tests/embedding/test_cached_embedding.py -v -o addopts=""

8 tests covering cache/miss counting, LRU eviction, copy-on-read/write
safety, non-string bypass, and input validation.
