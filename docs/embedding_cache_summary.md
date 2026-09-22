# Embedding-Cache Co-Contribution: CachedEmbedding

Condensed for a shared report covering 4 contributions within an 8-12 page
budget (~1.5 pages here). The full version, with implementation detail and
extended discussion, lives in docs/writeup.md section 10 on the
feature/embedding-result-cache branch.

## Motivation

Every incoming GPTCache request -- hit or miss -- first pays for an embedding
forward pass. The default embedding_func recomputes this from scratch even
when the exact same text was embedded moments earlier. CachedEmbedding
(gptcache/embedding/cached_embedding.py) closes this gap: it wraps any
existing embedding backend with an in-process LRU cache keyed on the exact
input string, returning a cached vector without touching the model on a
repeat.

## Design

Composition, not inheritance: CachedEmbedding(SBERT(...)) wraps any
BaseEmbedding backend in one line, with no changes to core GPTCache files.
Returned arrays are copied on both read and write so a caller mutating a
result cannot corrupt the cache. Only hashable string inputs are cached;
batch/list calls and non-text backends pass through untouched.

## Honesty trap

This is an exact-match cache, not a semantic one -- GPTCache's vector
search already handles near-duplicates, so this cache only fires on
byte-for-byte repeats. We measure, rather than assume, the exact-duplicate
ratio of every query stream before reporting a speedup. On a single pass
through 2,000 distinct prompts (no repeats), the duplicate ratio is 0.0%
and the cache correctly shows no benefit (1.02x) -- a valid negative result.

## Experimental setup

A separate benchmark (benchmark_embedding_cache.py) walks a query stream
one prompt at a time, calling to_embeddings() per query -- unlike
benchmark_lmsys.py, which batches all embeddings once up front and so
cannot exercise a per-query cache at all. We replay 5,000 UltraChat queries
through the same Zipf-skewed, hot-set-rotating stream generator used for
the eviction experiments (sections 6-7), over 7 seeds, timing a plain SBERT
encoder against the same encoder wrapped in CachedEmbedding (cache size 10,000).

## Results

| Metric | Value |
|---|---|
| Exact-duplicate ratio (measured) | 75.6-76.5% across seeds |
| Cache hit rate | Tracks duplicate ratio almost exactly per seed |
| Seeds where cache was faster | 7/7 |
| Speedup range | 3.18x - 4.79x |
| Geometric mean speedup | 3.80x |
| Wilcoxon signed-rank p | 0.0156 (significant at alpha=0.05) |

Hit rate tracking the measured duplicate ratio in every seed is itself a
correctness check: the cache is neither under- nor over-firing relative to
the traffic it sees. Absolute wall-clock times varied across seeds on a
shared, non-isolated benchmark machine (baseline 176-210s for a fixed
5,000-query stream) -- the significance claim rests on the paired,
within-seed comparison (baseline/cached measured back-to-back), not on
absolute magnitude, following the same paired-seed protocol as the
eviction-policy results.

## Scope and limits

No claim about semantic near-duplicates (out of scope by design). Cache
size (10,000) was large enough that no observed entry was evicted before
re-request in this run, so hit rate here is an upper bound for this
workload size, not a general guarantee -- smaller caches or larger corpora
would show proportionally lower hit rates via graceful LRU degradation.

## Evidence

bench_embedding_cache/results_seed{0-6}.json,
bench_embedding_cache/paired_stats.json,
tests/unit_tests/embedding/test_cached_embedding.py (8/8 passing).
