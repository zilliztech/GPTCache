# Cost-Aware W-TinyLFU for Semantic LLM Caches — submission

Eitan Dubinski, Ayala Egoz.

This repository is a fork of [GPTCache](https://github.com/zilliztech/GPTCache) with four
contributions to its eviction, storage, and embedding layers. The full write-up is
**`docs/report.tex`**; this README is the entry point for building, testing, and reproducing it.

## Contributions at a glance

| # | Contribution | Code | Report section | Unit test |
|---|---|---|---|---|
| 1 | `CA_W_TINYLFU` — cost-aware W-TinyLFU eviction | `gptcache/manager/eviction/ca_w_tinylfu.py` | *Design* / *Results* | `tests/unit_tests/manager/eviction/test_ca_w_tinylfu.py`, `..._routing.py` |
| 2 | `SBERTMRL` + FAISS HNSW/SQ8 storage compression | `gptcache/embedding/sbert_mrl.py`, `gptcache/manager/vector_data/faiss.py` | *Storage co-contribution* | `tests/unit_tests/manager/test_local_index.py` |
| 3 | `EmbeddingDispatcher` — multiprocess encode fan-out | `gptcache/embedding/dispatcher.py` | *Embedding-layer efficiency* | `tests/unit_tests/embedding/test_dispatcher.py` |
| 6 | `CachedEmbedding` — exact-repeat encode cache | `gptcache/embedding/cached_embedding.py` | *Embedding-layer efficiency* | `tests/unit_tests/embedding/test_cached_embedding.py` |

(Numbering follows the project directions; #4/#5 were not pursued.)

## Install

Requires Python ≥ 3.8.1.

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r examples/benchmark/requirements.txt
pip install faiss-cpu onnxruntime psutil
pip install -e .
```

A Docker path (recommended for a clean reproduction) is documented in
[`BENCHMARKING.md`](BENCHMARKING.md#setup). The gated datasets (`lmsys`, `wildchat`) need a
HuggingFace token (`huggingface-cli login` or `export HF_TOKEN=...`); `ultrachat` and `qqp` are public.

## Verify the build (no downloads)

```bash
# unit tests for all four contributions
pytest tests/unit_tests/manager/eviction/test_ca_w_tinylfu.py \
       tests/unit_tests/manager/eviction/test_ca_w_tinylfu_routing.py \
       tests/unit_tests/manager/test_local_index.py \
       tests/unit_tests/embedding/test_cached_embedding.py \
       tests/unit_tests/embedding/test_dispatcher.py

# offline benchmark smoke tests (real code paths, synthetic no-download encoders)
python examples/benchmark/benchmark_embedding_cache.py --dataset synthetic
python examples/benchmark/benchmark_embedding_dispatcher.py --dataset synthetic --n-prompts 40 --concurrency-levels 1,4
```

## Reproduce the paper

The reference JSON logs behind every table are already checked in under the `bench_*/` directories,
so a grader can inspect the numbers without re-running anything. To regenerate them:

| # | Benchmark script | Results dir |
|---|---|---|
| 1 | `examples/benchmark/benchmark_lmsys.py` | `bench_lmsys/`, `bench_wildchat/` |
| 2 | `examples/benchmark/benchmark_qqp.py` | `bench_real_100k/` |
| 3 | `examples/benchmark/benchmark_embedding_dispatcher.py` | `bench_embedding_dispatcher/` |
| 6 | `examples/benchmark/benchmark_embedding_cache.py` | `bench_embedding_cache/` |

**Full per-contribution commands, seeds, and aggregation scripts are in
[`BENCHMARKING.md`](BENCHMARKING.md).** Headline numbers come from replaying seeds 0–6 and pairing
per-seed (mean paired delta, sign count, exact Wilcoxon *p*, 95% CI).

> The dispatcher throughput crossover (#3) is host-dependent — the speedup magnitude and break-even
> point vary with core count and machine load. Reported numbers are from a single machine; the ~5 GB
> RSS floor is the stable, reproducible cost.

## Building the report

```bash
cd docs && pdflatex report.tex && pdflatex report.tex   # twice, for refs
```

`docs/report.tex` is self-contained (figures under `docs/figures/`, bibliography inline).
