# Storage front — walkthrough

> What we changed, why it compresses, why deletion needs tombstones, whether we
> could avoid them, and which of last year's wins transfer. Review doc, not prose
> for the thesis. Numbers marked **[TBD-JSON]** are not yet backed by a benchmark
> file and must not be quoted until they are.

---

## 1. What problem the storage front solves

GPTCache stores one float32 embedding per cached query in a Faiss index, plus the
question/answer in the scalar store. The embedding index is the memory hog: with the
default `IDMap,Flat` at 768 dims, every cached entry costs **768 × 4 = 3072 bytes** of
vector, and search is **O(n)** brute force over all of them. At cache sizes that make
semantic caching interesting (10⁵–10⁶ entries) that's the dominant cost and the
dominant latency.

We attack both with two independent, stackable levers.

---

## 2. What exactly we changed — two levers

### Lever A — MRL truncation (`gptcache/embedding/sbert_mrl.py`)

Matryoshka Representation Learning models (we use `nomic-embed-text-v1.5`) are trained
so the most important semantic information lives in the **first** coordinates of the
vector. So you can slice the 768-dim output down to `target_dim` (default 256) and
re-normalize, and keep most of the retrieval quality.

```
emb        = model.encode(text)        # (768,)
truncated  = emb[:, :target_dim]       # (256,)  ← just a slice
normalized = truncated / ||truncated|| # re-normalize, critical for cosine
```

That's the whole mechanism (`sbert_mrl.py:68-73`). 768→256 = **3× fewer dimensions**,
and crucially the truncation is free — it's an array slice, no extra model, no retrain.

### Lever B — HNSW + 8-bit scalar quantization (`gptcache/manager/vector_data/faiss.py`)

`index_type="hnsw_sq8"` swaps the flat index for `HNSW{M},SQ8` (`faiss.py:140-163`):

- **SQ8**: each float32 coordinate is quantized to a `uint8` (the index learns per-dim
  min/max). **4× smaller** vector payload, near-instant training (`faiss.py:218-222`).
- **HNSW**: a navigable-small-world graph → **O(log n)** search instead of O(n).

### The multiplicative claim — and what it measures out to

The *arithmetic* is dimension (3×) × quantization (4×) = ~12× on the **vector payload**
(3072 B → 256 B/vec). But that payload number is not total index memory: HNSW also stores
graph links (~`M·2·4` ≈ 256 B/vec at M=32), which roughly *doubles* the per-vector
footprint. So the headline must come from the benchmark, not the slide arithmetic — and it
now does (`bench_real_100k/results.json`, 100K real-encoder QQP, `faiss.serialize_index`
RAM vs the ONNX/768/Flat baseline A = 308 MB):

| Config | B/vec | **vs baseline** | TP% | FP% | precision |
|---|---:|---:|---:|---:|---:|
| A — ONNX/768/Flat (baseline) | 3080 | 1.0× | 89.25 | 18.50 | 82.8 |
| **G — MRL/256/HNSW+SQ8 (M=16)** | 408 | **7.5×** | 98.85 | 40.80 | 70.8 |
| **E — MRL/256/HNSW+PQ** | 315 | **9.8×** | 95.40 | 24.80 | 79.4 |

So the citable numbers are **~5.7× for the default SQ8 (M=32) cell, up to 7.5× at M=16, and
9.8× with PQ** — *not* the 12× payload arithmetic. The two Pareto-frontier cells are **E**
(max compression, 9.8×) and **G** (recall ceiling, 7.5×).

> **Honesty flag — compression is not free, and the cost is precision.** Every MRL cell
> *raises* TP hit rate (the matryoshka 256-d nomic encoder is simply a stronger retriever
> than the 768-d ONNX baseline) but also raises the **false-positive** rate (18.5% → 24.8–40.9%),
> so the baseline keeps the best precision (82.8%). Among compressed cells **E is the most
> precise (79.4%)** because PQ suppresses spurious near-matches. The accuracy tradeoff lives
> on the FP/precision axis, and it must be reported next to the ×-number. Caveat: this is a
> *fresh-index, no-eviction* measurement (see §5).
>
> **Update (threshold sweep + isolation cell I, §6):** this FP/precision gap is measured at a single
> shared 0.90 threshold and is *largely a threshold artifact* — at a per-cell threshold matched to the
> baseline's precision, the MRL cells hold TP ≥ baseline at 7.5–9.8× less RAM. The genuine residual
> costs are **e2e latency** (slower encoder, §6) and **tombstone steady-state** (§5), not accuracy.

---

## 3. Why tombstones?

The single hard fact: **Faiss HNSW indexes do not implement `remove_ids`.** A graph
index links nodes by internal id; ripping a node out would sever the graph's
connectivity, so Faiss simply doesn't support structural deletion on HNSW (flat IVF do).

But the eviction layer — our *other* contribution — is built entirely on deletion. When
`CA_W_TINYLFU` evicts a key, `SSDataManager._clear` → `eviction_manager` →
`vector.delete(ids)`. With flat that's a native `remove_ids` and the slot is freed
immediately. With HNSW, `delete()` has nothing to call.

So `delete()` branches (`faiss.py:333-351`):

- **flat** → `remove_ids`, physical removal, done.
- **hnsw** → add the id to a `_tombstones` set (logical "mark deleted").

and `search()` compensates (`faiss.py:288-301`): if tombstones exist, **over-fetch**
`top_k + len(tombstones)` candidates and skip any tombstoned id before returning top_k.
Tombstones persist next to the index as a `.tombstones.npy` sidecar so deleted ids stay
dead across reload (`faiss.py:359-365`).

**On "monkey patches":** there is no monkey-patching of the Faiss library here. It's a
thin wrapper *inside our `Faiss` class* — a set, an over-fetch-and-filter in search, and a
sidecar file. The only thing that feels hacky is over-fetching to mask deletions, and
that's the standard trick (below).

---

## 4. Could we avoid tombstones? Yes — but each alternative is a different Pareto point

First, the reframe: **tombstones are the canonical solution, not our hack.** Lucene/
Elasticsearch HNSW, Milvus, and Weaviate all do deletion in graph indexes exactly this
way — soft-delete + filter-at-search + periodic segment compaction. We're on the
well-trodden path. That said, the real alternatives that *eliminate* tombstones:

| Alternative | Native deletion? | What you give up |
|---|---|---|
| **IVF + SQ8** (`IVF{nlist},SQ8`) | ✅ `remove_ids` works (inverted lists are mutable) | IVF needs real training on a representative sample (SQ8-on-HNSW trains instantly); recall depends on `nprobe`; weaker than HNSW at small n. |
| **hnswlib backend** (not Faiss) | ✅ native `mark_deleted` / `allow_replace_deleted` | hnswlib stores **float32** — no scalar quantization, so you keep HNSW's search win but lose Lever B's 4× compression. |
| **Periodic re-embed rebuild** (keep HNSW+SQ8) | Simulated — rebuild a fresh index from live entries | Must re-embed live text from the scalar store (compute cost), or keep float32 originals (defeats the compression). |

The honest takeaway for the thesis: **we chose HNSW+SQ8 for instant training + best
small-n recall + 4× quantization, and the price of that specific choice is that deletion
becomes logical (tombstones) instead of physical.** IVF+SQ8 is the one option that gives
both compression *and* native deletion; it's the natural "future work / ablation" lever
if tombstone accumulation (§5) ever bites.

---

## 5. The real fragility tombstones introduce — and it's coupled to eviction

Look at `rebuild()` (`faiss.py:310-331`): for HNSW it is a **deliberate no-op**. The
quantized graph can't be decoded back to vectors, so we can't compact in place. Which
means:

- A tombstone removes an id from *results*, but the dead vector **still sits in the index
  consuming its SQ8 bytes + graph links**, forever (until the whole index is rebuilt from
  scratch elsewhere).
- Tombstones only grow (bounded by lifetime evictions). The set itself is cheap (~8 MB per
  1M evictions, `faiss.py:321-323`), but the *dead vectors it points at are not reclaimed*,
  and search over-fetch grows with the tombstone count.

So combined with our eviction contribution the headline erodes: **the 5.7–9.8× compression is
measured on a fresh index; under sustained eviction the index keeps its dead bodies, so
steady-state memory is worse than the fresh-index number, and search slowly pays more
over-fetch.** This is exactly the kind of in-line tradeoff to state honestly (it's the
storage analogue of last year's "+14% memory for the better model"). It's also the cleanest
argument for the IVF+SQ8 future-work lever, which *does* reclaim on delete.

---

## 6. Data source (gap now closed)

The compression ratio + accuracy cost are measured in **`bench_real_100k/results.json`**
(summarised in `bench_real_100k/frontier.md`): 100K real QQP, FAISS 1.13.2, 3 repeats,
threshold 0.90, cells A,D,E,F,G,H. Headlines: **5.7× (D, SQ8 M=32), 7.5× (G, SQ8 M=16),
9.8× (E, PQ)** vs the ONNX/768/Flat baseline; precision 70.7–79.4% vs baseline 82.8%.

Both open follow-ups are now **run** (cells A,E,G,I, 3 repeats, threshold sweep 0.84–0.96;
`frontier.md` §"Isolation cell I" and §"threshold sweep"):

- **Threshold sweep — the FP gap is largely a threshold artifact.** The global 0.90 threshold was
  tuned for the 768-d ONNX baseline and sits at a different point on each cell's PR curve. At a
  per-cell threshold matched to A's precision (~0.83), the MRL cells hold TP **≥** baseline at
  7.5–9.8× less RAM (E @0.92: prec 0.864, TP 0.882; G @0.94: prec 0.838, TP 0.900 > A's 0.892). So
  on the accuracy axis the MRL cells are Pareto-competitive once the threshold is tuned per cell — the
  "baseline keeps best precision" line only holds at a single shared threshold.
- **Isolation cell I (MRL/768/Flat) — the encoder, not truncation, drives the FP rise.** At the
  shared 0.90 threshold: A→I (encoder swap, full 768) is **+14.4pp FP** (18.5→32.9); I→G (truncate
  256 + SQ8) adds only +7.9pp; I→E (truncate 256 + PQ) *lowers* FP by 8.0pp (32.9→24.9). MRL
  truncation is not the precision culprit — the looser MRL encoder is, and PQ claws it back.
- **New honesty cost — e2e latency reverses the search win.** Search-only confirms ~100× (19.8→0.2 ms
  p95), but **e2e p95 rises** (A 59 → E/G 97–98 → I 117 ms) because the MRL encoder is slower than
  ONNX and embedding dominates e2e. Compression + search-CPU are the bankable wins; latency is not.

---

## 7. Which of last year's improvements also apply here?

Last year (ModelCache) shipped: bulk insertion, DB query optimization, embedding-model
swap (Data2VecAudio→all-mpnet, L2→cosine), in-memory LRU cache, REST→WebSocket,
multiprocessing embedding, async offloading.

| Last-year win | Applies to us? | Notes |
|---|---|---|
| **Bulk / batched insertion** | ✅ already have it | `mul_add` inserts batches (`faiss.py:203`); `SBERTMRL.to_embeddings` already batch-encodes (`sbert_mrl.py:64-66`). Could push batch size higher; low effort. |
| **Embedding-model swap** | ✅ that's our Lever A | They swapped for *quality*; we swap to an MRL model for *compressibility*. Same move, different objective. |
| **L2 → cosine** | ⚠️ already equivalent for us | We use `METRIC_L2` but L2-normalize every vector (embedder normalizes, `SSDataManager` normalizes). On unit vectors L2 distance is monotonic with cosine, so the switch is a no-op for us. Worth one sentence to preempt the reviewer question. |
| **In-memory LRU cache** | ✅ this *is* our eviction contribution | They added LRU; we generalize the whole eviction layer (CA_W_TINYLFU). Cite as the gap we extend. |
| **Multiprocessing / async embedding** | ➖ transferable but out of scope | GPTCache already has `aadapt`; parallel embedding is a generic throughput win unrelated to either contribution. Mention as future work, don't claim. |
| **DB query optimization** | ➖ scalar-store specific | Orthogonal to both our fronts; not our scope. |
| **REST → WebSocket** | ❌ N/A | We're a library, not a served system; no transport layer to optimize. |

The genuinely transferable, in-scope ones are **batched insertion** (have it) and the
**embedding swap / L2-vs-cosine clarification** (free reviewer-proofing). The rest are
either already our contribution, or service-architecture work that doesn't apply to a
library.
