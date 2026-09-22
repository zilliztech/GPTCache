# How to benchmark

Reproduction guide for the four contributions in *Cost-Aware W-TinyLFU for Semantic LLM Caches*
(`docs/report.tex`). Every command below is deterministic given a `--seed`; the paper's headline
numbers come from replaying seeds 0–6 and pairing per-seed.

| # | Contribution | Script | Result dir |
|---|---|---|---|
| 1 | `CA_W_TINYLFU` cost-aware eviction | `examples/benchmark/benchmark_lmsys.py` | `bench_lmsys/`, `bench_wildchat/` |
| 2 | SBERTMRL + HNSW/SQ8 storage | `examples/benchmark/benchmark_qqp.py` | `bench_real_100k/` |
| 3 | `EmbeddingDispatcher` fan-out | `examples/benchmark/benchmark_embedding_dispatcher.py` | `bench_embedding_dispatcher/` |
| 6 | `CachedEmbedding` exact-repeat cache | `examples/benchmark/benchmark_embedding_cache.py` | `bench_embedding_cache/` |

The reference JSON logs for every table in the report are already checked in under those `bench_*/`
directories — a grader can inspect the sample outputs without re-running anything.

---

## Setup

### Option A — Docker (recommended for reproducibility)

```bash
docker build -t gptcache-bench .
# writes results into ./out on the host
docker run --rm -v "$PWD/out:/app/out" \
  gptcache-bench examples/benchmark/benchmark_lmsys.py --dataset ultrachat --out out/results.json
```

`ultrachat` is public and needs no credentials. The gated datasets (`lmsys`, `wildchat`) need a
HuggingFace token:

```bash
docker run --rm -e HF_TOKEN=hf_xxxxx -v "$PWD/.hf_cache:/app/.hf_cache" -v "$PWD/out:/app/out" \
  gptcache-bench examples/benchmark/benchmark_lmsys.py --dataset lmsys --out out/lmsys.json
```

Mount `.hf_cache` so datasets/models download once and persist across runs.

### Option B — local virtualenv

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r examples/benchmark/requirements.txt
pip install faiss-cpu onnxruntime psutil
pip install -e .
```

Requires Python ≥ 3.8.1. For the gated datasets: `huggingface-cli login` (or `export HF_TOKEN=...`).

---

## Contribution 1 — cost-aware eviction (`CA_W_TINYLFU`)

Full-stack replay (SBERT encode → FAISS search → similarity gate → on-miss save with eviction) against
a fresh disk-backed SQLite+FAISS store. Primary metric: **cost-weighted hit rate**.

**Stationary regime** (fixed Zipf, frequency's regime — Table 2 / `tab:stationary`):

```bash
# flat skew z11 (drift-zipf 1.1), sharp skew z15 (drift-zipf 1.5); sweep seeds 0..6
for s in 0 1 2 3 4 5 6; do
  python examples/benchmark/benchmark_lmsys.py --dataset lmsys \
    --drift --drift-rotate 0 --drift-zipf 1.1 \
    --cache-sizes 25,50,100 --policies LRU,GDSF,WTINYLFU_FREQ,CA_W_TINYLFU \
    --repeats 1 --seed $s --out bench_lmsys/win_z11_seed$s.json
done
```

Swap `--dataset wildchat` for the external-validity replication (`bench_wildchat/`).

**Drift regime** (rotating hot set — Table 1 / `tab:drift`). Decay is the lever: `--virtual-clock-sec 0`
is decay-off (ADAPT loses), `120` is decay-on (ADAPT wins):

```bash
# decay-on, gentle drift, adaptive window, seeds 0..6
for s in 0 1 2 3 4 5 6; do
  python examples/benchmark/benchmark_lmsys.py --dataset lmsys \
    --drift --drift-rotate 3000 --drift-shift 0.02 --drift-zipf 1.2 \
    --virtual-clock-sec 120 --adaptive-window \
    --cache-sizes 100 --policies LRU,CA_W_TINYLFU --repeats 1 --seed $s \
    --out bench_lmsys/decaytuned_sh02_vc120_seed$s.json
done
```

**GDSF head-to-head** (Table 3 / `tab:gdsf`): include `GDSF` in `--policies` at `--drift-zipf 1.1`
(stationary flat) and `1.5` (sharp). **Cost-priority dial** (Section 6.6): add
`--cost-priority-sweep 0,0.25,0.5,0.75,1 --cache-sizes 200`.

**Aggregation** (paired stats → the deltas and Wilcoxon p-values in the report):

```bash
python bench_lmsys/paired.py       # paired mean delta, sign count, exact Wilcoxon, 95% CI
python bench_lmsys/aggregate.py    # per-cell summary tables
```

---

## Contribution 2 — storage (SBERTMRL + HNSW/SQ8)

100K-vector real-encoder run over QQP, 2000 true-positive + 2000 false-positive probes per cell.
Cells map to Table 5 (`tab:storage`): A = ONNX/768/Flat baseline, E = MRL/256/HNSW+PQ, G =
MRL/256/HNSW+SQ8, J = static-encoder variant.

```bash
python examples/benchmark/benchmark_qqp.py --scale 100000 --cells A,E,G,J \
  --threshold-sweep 0.86,0.88,0.90,0.92,0.94 --out bench_real_100k/results.json
```

The `--threshold-sweep` is what shows the false-positive gap is mostly a threshold artifact
(report Section 8, "Accuracy is mostly a threshold artifact"). Reference logs: `bench_real_100k/cell_*/`.

---

## Contribution 6 — exact-repeat embedding cache (`CachedEmbedding`)

Per-query replay (one `to_embeddings()` call at a time) timing a plain SBERT encoder against the same
encoder wrapped in `CachedEmbedding`. Reports the **measured** exact-duplicate ratio, hit rate, and
speedup — Table 6 (`tab:embcache`).

```bash
# structural smoke test — no network, no model download
python examples/benchmark/benchmark_embedding_cache.py --dataset synthetic

# paper run: real SBERT, UltraChat replay, seeds 0..6
for s in 0 1 2 3 4 5 6; do
  python examples/benchmark/benchmark_embedding_cache.py \
    --dataset ultrachat --n-queries 5000 --cache-size 10000 --seed $s \
    --out bench_embedding_cache/results_seed$s.json
done
python bench_embedding_cache/paired_stats.py   # 7/7 seeds, geo-mean 3.80x, Wilcoxon p=0.016
```

**Negative control** (the honesty check): a single pass over distinct prompts has a 0% duplicate ratio
and correctly yields ~1.02× — see `bench_embedding_cache/README.md`.

---

## Contribution 3 — dispatcher (`EmbeddingDispatcher`)

Embedding *throughput* under concurrent load: a single-process encoder (concurrent callers contend on
one shared model) vs. a dispatcher that fans across worker processes. Reports the crossover in Table 7 —
the dispatcher loses below ~15 concurrent callers (IPC overhead), crosses over in an unstable ~15–30
region, and wins above it, up to ~3.4× at 100, at a fixed ~5 GB RSS cost. The crossover point is
host-dependent; numbers are from one machine.

```bash
# offline plumbing check -- real worker processes, synthetic (no-download) encoder
python examples/benchmark/benchmark_embedding_dispatcher.py --dataset synthetic \
  --n-prompts 40 --concurrency-levels 1,4

# paper run: 3 runs of the 8-level crossover sweep (200 prompts, 8 workers)
for r in 1 2 3; do
  python examples/benchmark/benchmark_embedding_dispatcher.py --dataset ultrachat \
    --n-prompts 200 --concurrency-levels 1,10,20,30,40,50,75,100 \
    --out bench_embedding_dispatcher/results_fine_run$r.json
done

# aggregate the 3 runs into the per-concurrency mean +/- std of Table 7
python bench_embedding_dispatcher/finegrained_stats.py "bench_embedding_dispatcher/results_fine_run*.json"
```

Memory numbers (`rss_mb`, summed over the worker children) require `psutil` (`pip install psutil`);
without it the run still works and reports `0.0`. The dispatcher itself
(`gptcache/embedding/dispatcher.py`) is unit-tested (`tests/unit_tests/embedding/test_dispatcher.py`)
and usable directly:

```python
from gptcache.embedding.dispatcher import EmbeddingDispatcher

def make_sbert():                       # must be a picklable module-level fn (Windows spawn)
    from gptcache.embedding import SBERT
    return SBERT("all-MiniLM-L6-v2")

with EmbeddingDispatcher(make_sbert, num_workers=8) as d:
    vecs = d.to_embeddings_batch(list_of_texts)
```

---

## Metrics recorded

Each run emits JSON with, per configuration: `cost_weighted_hit_rate` (primary), `hit_rate`,
`token_saving_ratio`, latency `p50/p95/p99`, throughput (qps), and peak memory (`--track-mem`, or RSS
for the dispatcher). The paired helper scripts turn per-seed JSON into the report's deltas: mean paired
delta, sign-consistency count, exact two-sided Wilcoxon p, and a paired-t 95% CI.

## Running the unit tests

```bash
pytest tests/unit_tests/manager/eviction/test_ca_w_tinylfu.py \
       tests/unit_tests/embedding/test_cached_embedding.py \
       tests/unit_tests/embedding/test_dispatcher.py
```
