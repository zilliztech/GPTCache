# Semantic-ARC for GPTCache

A complete account of one change to GPTCache: a new in-memory eviction policy,
`ARC`, whose ghost lists are matched by **embedding similarity** instead of by
exact key. This document covers the hypothesis, why it was screened before it
was built, how it is implemented, how it is tested, what it measures, and where
the evidence stops.

Branch: `feature/semantic-arc`. Everything below regenerates from a clean clone;
see [Reproducing](#12-reproducing).

---

## 1. Summary

| | |
|---|---|
| **Claim** | Cache policies that keep history about *non-resident* items key that history by exact identity. In a semantic cache the returning item is a **paraphrase** — a different string with a different row id — so the history never fires and the adaptive machinery is silently dead. Keying it by embedding similarity revives it. |
| **Change** | `gptcache/manager/eviction/arc.py` (467 lines) + a backward-compatible `embeddings=` kwarg on `EvictionBase.put()`. No existing policy path modified. |
| **Headline** | At capacity 100: ARC 58.08% / 54.21% / 29.02% on stationary, drifting and real-prompt traffic, against LRU's 48.87% / 48.05% / 28.90% and LFU's 56.53% / 28.68% / 11.96%. Best worst case, no per-workload tuning. |
| **The ablation** | Flipping one constructor flag to exact-key ghosts drops drifting-traffic hit rate from 54.21% to 36.76% and pins ARC's adaptive parameter at exactly `p = 0.000` for every trace. |
| **Honest limit** | On the only real, timestamp-ordered traffic tested, ARC beats LRU by **0.06–1.31 pp**. The large margins are on synthetic arrival orders. |
| **Tests** | 27 new tests, all passing. Pre-existing failures unchanged at 29 (all external services / optional deps). |

---

## 2. The hypothesis

### 2.1 What a semantic cache is

GPTCache answers a query by finding a *semantically similar* query it has
already answered, and returning that answer instead of calling the LLM. A hit
is `cos(q, entry) >= tau` against the resident set.

The cache is bounded, so every miss forces an eviction. GPTCache shipped four
policies — LRU (default), LFU, FIFO, RR — all delegating to `cachetools`.

### 2.2 The gap

Modern replacement policies do better than LRU by keeping **history about items
that are no longer resident**:

- **ARC** (Megiddo & Modha, FAST 2003) keeps two *ghost lists* `B1`/`B2`: the
  identities of recently evicted entries, holding no cached response. A query
  that returns and hits a ghost is proof the eviction was a mistake, and ARC
  moves its recency/frequency boundary `p` accordingly.
- **TinyLFU** keeps a frequency sketch of items it has seen, resident or not.

Both key that history **by exact item identity**. That is correct for a block
cache, where the same block number returns. It is not correct for a semantic
cache, where the thing that returns is a *paraphrase*: a different string, and —
because GPTCache assigns a fresh row id on every miss — a different id.

### 2.3 The thesis, stated so it can fail

> In a semantic cache, history keyed by exact identity can never fire, so the
> adaptive machinery that depends on it is inert. Keying that history by
> embedding similarity restores it.

This makes two falsifiable predictions:

1. **Classic ARC's `p` never moves.** If ghosts can only match ids that by
   construction never return, `p` stays at its initial value for an entire
   trace, and ARC silently degrades to something close to plain LRU.
2. **An exact-key frequency sketch carries no signal.** Its estimates should be
   uncorrelated with true query popularity.

Prediction 2 is deliberately stated as a property of *keying*, not of any
policy implementation — it is a claim about exact-key sketches on semantic
workloads, and is independent of TinyLFU's merits as an algorithm.

### 2.4 Scope

Rejected before implementation, with reasons recorded in
`resources/RESULTS.md`: cost-aware eviction (GDSF) and W-TinyLFU + cost
admission were already open PRs by classmates; approximate matching *is* the
library; coverage/redundancy-based eviction was tested in simulation and lost
to LRU by 5.9 pp. An LSH-keyed frequency sketch is measured here as evidence
for prediction 2 but deliberately **not** implemented — it is future work.

---

## 3. Screening before building

Both predictions were tested on real embeddings **before** any policy code was
written, as a pass/fail gate (`benchmarks/gate_screening.py`). The point was to
make the project falsifiable at a stage where abandoning it was still cheap.

**Check 1 — exact-key sketches are inert.** Correlation between sketch estimate
and true cluster popularity:

| keying | mean estimate | max | corr with true popularity |
|---|---|---|---|
| exact key | 0.20 | 2 | **−0.037** |
| LSH (12 bits × 16) | 2.15 | 7 | **+0.931** |

Pass condition `|corr_exact| < 0.15` → passed. An exact-key counter almost never
sees the same key twice, so it measures nothing.

**Check 2 — semantic ghosts beat exact ghosts on drifting traffic.**

| trace | cap | LRU | LFU | ARC exact | ARC semantic | margin | final `p` (exact / semantic) |
|---|---|---|---|---|---|---|---|
| quora-drift | 100 | 47.90% | 25.42% | 37.05% | **54.31%** | +17.26 pp | 0.0 / 1.0 |
| quora-drift | 400 | 62.00% | 47.21% | 47.05% | **62.20%** | +15.15 pp | 0.0 / 233.8 |
| wildchat | 100 | 28.29% | 12.34% | 16.46% | **28.44%** | +11.98 pp | 0.0 / 56.1 |
| wildchat | 400 | 30.22% | 18.80% | 19.30% | **30.48%** | +11.18 pp | 0.0 / 103.4 |

Pass condition "every margin ≥ 3 pp" → passed, worst margin +11.18 pp. Note the
final-`p` column: **exact ghosts hold `p` at 0.0 on every trace**, which is
prediction 1 confirmed directly.

---

## 4. Implementation

**File:** `gptcache/manager/eviction/arc.py`. Self-contained — standard library,
numpy, and `EvictionBase`.

### 4.1 State

```
c              capacity (maximum resident entries)
p              float in [0, c], adaptive target size for T1
T1, T2         resident:  seen-once / seen-more-than-once, LRU-ordered
B1, B2         ghost lists: evicted from T1 / T2, embedding but no response
tau            cosine threshold for a ghost match
```

Invariants, asserted in tests and by a `check_invariants()` method:

```
|T1| + |T2| <= c            |T1| + |B1| <= c
|T2| + |B2| <= 2c           |T1|+|T2|+|B1|+|B2| <= 2c
0 <= p <= c                 |T1|+|T2| < c  =>  B1 and B2 are empty
```

### 4.2 The contribution, in one method

Cases I–IV follow Figure 4 of the paper line for line. The **only** algorithmic
change is the ghost membership test:

```python
def _match(self, ghosts: _VecList, query, key):
    """Ghost membership test -- the semantic modification lives here."""
    if not self._semantic:
        return key if key in ghosts else None      # classic ARC
    if query is None:
        return None
    return ghosts.nearest(query, self._tau)        # nearest ghost, if >= tau
```

Classic ARC asks *"is this id in B1?"*. Semantic ARC asks *"is there a ghost
whose embedding is within `tau` of this query?"*. Everything downstream — the
`p` update, `REPLACE`, the four cases — is unchanged.

This is gated behind a constructor flag, `ghost_matching="semantic"|"exact"`,
and **that one flag is the ablation**: `"exact"` reproduces classic ARC and its
failure mode exactly, which is why it exists.

Resident lookup is deliberately *not* done here. GPTCache's vector store already
finds the nearest resident entry and calls `get(id)` on a hit; only the ghosts
need their own search.

### 4.3 `_VecList` — why not a dict of vectors

The obvious implementation keeps `dict[id] -> np.ndarray` and stacks on demand.
That allocates a fresh `(n, d)` matrix on every miss. Instead each of the four
ARC lists is a `_VecList`: one contiguous `float32` matrix plus an
insertion-ordered dict for LRU order.

- **Search** is a single BLAS `matvec` over a contiguous block — no Python loop,
  no per-call allocation beyond the score vector. Since inputs are unit-norm,
  cosine similarity *is* the dot product.
- **Removal** is O(1): swap the last row into the vacated slot and fix two index
  entries.
- **Growth** doubles the backing matrix.

`nearest()` is tested against a naive scan for agreement
(`test_nearest_matches_naive_scan`).

### 4.4 Integration

Four small, backward-compatible edits:

| File | Change |
|---|---|
| `eviction/base.py` | `put(objs)` → `put(objs, embeddings=None)`. Optional and defaulted, so every existing implementation and caller is unaffected. |
| `eviction/memory_cache.py` | One `elif self._policy == "ARC"` branch. ARC is **not** wrapped in `popitem_wrapper` — that wrapper exists to adapt cachetools, and ARC performs its own eviction and reports through `on_evict` directly. |
| `manager/factory.py` | Threads an `eviction_params` dict through, so `tau` / `ghost_matching` reach the policy. |
| `manager/data_manager.py` | Passes the already-normalised `embedding_datas` to `put()`. Every non-ARC policy ignores the kwarg. |

The LRU / LFU / FIFO / RR code paths are untouched.

### 4.5 Failure modes handled deliberately

- **Entries with no embedding.** `SSDataManager` calls `put()` without vectors at
  start-up, to register rows already in the database. These become the zero
  vector, whose similarity to anything is 0 and therefore below any sensible
  `tau`. They participate fully in eviction but can never produce a ghost hit —
  degrading to classic ARC for those ids alone. Intended degradation, not a
  silent failure.
- **`clean_size` is ignored.** cachetools policies drop `0.2 * maxsize` entries in
  a batch; ARC releases one at a time and sits at exactly 100% occupancy. This
  matters for benchmarking and is controlled for — see §6.3.
- **Spurious FP warnings on macOS.** numpy 2.x `float32` matmul through Apple's
  Accelerate raises divide/overflow/invalid flags on perfectly finite unit-norm
  input. Verified harmless against a float64 `einsum` of the same data
  (agreement to 8e-8) and suppressed locally, because otherwise a cache emits
  bogus warnings on every miss.

---

## 5. Tests

`tests/unit_tests/eviction/test_arc_cache.py` — **26 tests**, plus one
parametrised case added to `tests/unit_tests/manager/test_eviction.py`. 27 new,
all passing.

| Group | What it pins down |
|---|---|
| **Invariants (5)** | Every invariant in §4.1 re-asserted after every operation, on random traffic, on paraphrase traffic, and under adversarial sequences chosen to push `p` to its bounds. Also that vector storage cannot grow without bound. |
| **Eviction protocol (3)** | A cold cache evicts nothing; `on_evict` fires exactly once per id — at eviction, never again when that id's ghost is later dropped; ghost demotion is announced when it happens. |
| **Ablation (3)** | `test_p_moves_under_semantic_matching`, `test_p_stays_zero_under_exact_matching`, `test_semantic_matching_wins_on_drifting_traffic`. The failure mode is a **test**, not just a chart: if someone "optimises away" semantic ghost matching, these fail. |
| **Degenerate (9)** | capacity 1; get on an empty cache; get a missing id; a duplicate id in `put` is an access not an admission; `put` without embeddings; a `None` embedding never ghost-matches; bad constructor arguments rejected; non-unit embeddings normalised; policy name. |
| **`_VecList` (3)** | `nearest()` agrees with a naive scan; respects `tau`; LRU order and `touch` behave. |
| **Through GPTCache (3)** | The public `EvictionBase` factory; a real `get_data_manager(CacheBase("sqlite"), VectorBase("faiss"), eviction="ARC")` saving 30 entries and holding exactly `max_size` rows; `tau` / `ghost_matching` actually reaching the policy object. |

### 5.1 Regression baseline

Recorded **before** any source change and re-run after
(`benchmarks/results/baseline_tests.txt`):

```
before:  29 failed, 25 passed
after:   29 failed, 52 passed
```

The identical 29 failures in the identical 12 files, every one requiring an
external service (redis, DynamoDB, Milvus, pgvector, Mongo, Qdrant) or a missing
optional dependency (chromadb, usearch, duckdb, hnswlib). None touch eviction.
The success criterion was that this set does not grow. It did not.

### 5.2 Determinism

Every correctness metric is bit-for-bit reproducible: the full sweep was run
twice and diffed, and max |delta| over `hit_rate`, `false_hit_rate`, `n_hits`,
`p_final`, `p_max` is exactly `0.0`. Two fixes were needed to get there:

- `cachetools.RRCache` picks its victim with the *unseeded global* `random`, so
  RR moved by up to 0.78 pp between runs. It now gets a per-cell seeded
  `choice` — random *within* a run, identical across runs.
- The exact-key sketch in the gate used builtin `hash()`, which Python salts
  per process for `str`. That silently affected only the exact-key arm, which is
  precisely what the gate measures. Now a stable blake2b hash.

Latency, throughput and heap are hardware-dependent and will not match exactly.
Hit rates will.

---

## 6. Benchmark methodology

### 6.1 What is real and what is not

This is the most important caveat in the document.

| Component | Real GPTCache? |
|---|---|
| ARC / LRU / LFU / FIFO / RR policy logic | **Yes** — the shipped classes, via the public factory |
| Vector store | No — a numpy brute-force top-1 scan replaces FAISS |
| Scalar store | No — a dict replaces SQLite |
| Request-time embedding | No — pre-computed offline |
| `Cache`, adapters, similarity evaluation, LLM call | Not exercised |

`benchmarks/simcache.py` supplies only the half of the cache the vector store
would supply, and delegates **every eviction decision** to a real
`MemoryCacheEviction`. So:

- **Hit rates are trustworthy.** Hit rate depends only on which entries are
  resident, and residency is decided entirely by the real policy objects. A
  brute-force top-1 cosine scan is the exact version of what FAISS approximates,
  and it is identical for every policy.
- **Latency is policy-level, not end-to-end.** The eviction-decision cost is
  measured on real policy objects and is valid. The per-request figures are
  simulator-level; real GPTCache adds ONNX embedding + FAISS + SQLite and is
  measured in milliseconds, not microseconds.
- **No LLM was called.** Savings are inferred from hit rate, not observed.

Isolating the variable under test is a defensible design, but "we benchmarked
GPTCache" would overstate it. "We benchmarked GPTCache's eviction policies" is
accurate.

### 6.2 Corpora

| Corpus | Size | Ground truth | Arrival order |
|---|---|---|---|
| `quora` | 53,250 questions, 12,235 paraphrase clusters | **Yes** — human `is_duplicate` labels, union-found | Synthetic (Zipf over clusters) |
| `stackexchange` | 225,540 titles, 35,878 clusters | **Yes** — duplicate-question links | Synthetic |
| `wildchat` | 71,637 prompts | No | **Real**, timestamped, Apr–Jun 2023 |
| `wildchat-long` | 150,000 prompts | No | **Real**, timestamped, Apr 2023 – Apr 2024 |

All embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-d, L2-normalised
so cosine is a dot product). Every input file's sha256 is recorded in
`data/*_meta.json`.

Ground-truth clusters are what make **false hit rate** measurable: a hit whose
stored entry belongs to a different cluster is a wrong answer served to a user.
Few caching papers report it. It is not measurable on the WildChat traces.

LMSYS-Chat-1M was specified originally and is still gated to this account —
`list_repo_files` succeeds, downloads raise `GatedRepoError`. WildChat is the
ungated equivalent and additionally exposes real timestamps.

### 6.3 Controls

- **Occupancy.** GPTCache's default drops `0.2 * maxsize` per eviction, so a
  cachetools cache lives between 80% and 100% full while ARC sits at 100%.
  Uncontrolled, that gap would show up as an ARC win having nothing to do with
  the replacement decision. Every cachetools policy is pinned to
  `clean_size=1`, `LRU-batch` carries the untouched default for comparison, and
  `resident_mean` is recorded on every row so a reader can check the control
  held.
- **`tau` selection.** The brief specified "the value maximising F1 between
  'same cluster' and 'sim ≥ tau'". Taken literally over random pairs that is
  unusable: the F1 curve is flat to ~1e-4 across tau ∈ [0.40, 0.50], and a cache
  never classifies a random pair — it takes the **maximum** over all resident
  entries, so the per-pair false-positive rate compounds with capacity. The
  pairwise argmax (0.43) makes a 1600-entry cache find a wrong-cluster neighbour
  above threshold for ~85% of queries. So the objective was kept and the
  decision rule corrected: `tau = 0.8`, maximising F1 of the *actual serving
  decision*. F1 0.975, precision 0.991, recall 0.960, false hit rate 0.88%. Both
  numbers are printed and stored; only the second is used.
- **Statistics.** Every comparison is a **paired bootstrap**: policies see the
  identical arrival sequence per seed, differences are taken per seed, and the
  mean difference is bootstrapped over seeds (10,000 resamples, 95% CI).

---

## 7. Results — the three fixed workloads

Capacity 100, mean of 10 seeds:

| Policy | Stationary | Drifting | Real prompts | **Worst case** |
|---|---|---|---|---|
| LRU (default) | 48.87% | 48.05% | 28.90% | 28.90% |
| LFU | 56.53% | 28.68% | 11.96% | **11.96%** |
| FIFO | 43.56% | 42.94% | 28.50% | 28.50% |
| RR | 43.73% | 42.68% | 27.21% | 27.21% |
| ARC-exact | 58.13% | 36.76% | 16.36% | 16.36% |
| **ARC** | **58.08%** | **54.21%** | **29.02%** | **29.02%** |

Read honestly: **LFU beats plain LRU by 7.7 points on stationary traffic**, and
ARC's margin over LFU there is only ~1.5 points. The case for ARC is the last
column. LFU's worst case is 12%; ARC's is 29%, and it is best in every column
but never by a landslide.

**The ablation is the sharpest single result.** `ARC-exact` is the same code
with one flag flipped. On stationary traffic it is statistically
indistinguishable from semantic ARC (58.13% vs 58.08%) — because with a fixed
popularity ranking, ghosts rarely need to fire. On drifting traffic it drops to
36.76% against 54.21%, and on real prompts to 16.36% against 29.02%. Its `p`
stays at exactly 0.000 throughout. Classic ARC on a semantic workload is not
merely suboptimal; it is inert.

---

## 8. Results — when does ARC actually beat LRU?

§7 establishes the worst-case claim but cannot say *what property of a workload*
decides the winner, because its two synthetic regimes are the endpoints of an
axis with nothing sampled in between. `benchmarks/sweep_crossover.py` makes the
axis continuous: `drift_rate` δ is the fraction of the popularity ranking
re-permuted at each of 6 epochs, so δ=0 reproduces the stationary regime and
δ=1 the drifting one.

**Validation of the parameterisation:** at capacity 100 the endpoints land at
LRU 49.16% / ARC 58.40% (δ=0) and LRU 47.73% / ARC 53.94% (δ=1), against the
committed 48.87/58.08 and 48.05/54.21. The axis is continuous *and* its ends
agree with the existing benchmark.

5,520 cells: 2 cluster corpora × 9 drift rates × 5 skews × 6 capacities ×
4 policies × 5 seeds, plus the real traces and a control.

### 8.1 The crossover surface

ARC − LRU on Quora at Zipf s=1.1, percentage points (bold = 95% paired
bootstrap CI excludes zero):

| capacity | δ=0 | δ=0.2 | δ=0.5 | δ=1.0 |
|---|---|---|---|---|
| 50 | **+10.93** | **+10.25** | **+9.58** | **+9.05** |
| 100 | **+9.23** | **+8.75** | **+7.51** | **+6.21** |
| 200 | **+7.46** | **+6.40** | **+4.84** | **+2.49** |
| 400 | **+5.38** | **+3.87** | **+2.15** | +0.10 |
| 800 | **+3.08** | **+1.57** | **+0.35** | **−1.16** |
| 1600 | **+0.88** | **+0.36** | **−0.25** | **−0.34** |

Two readings, and the second is the surprise:

1. The margin falls monotonically as **capacity** grows. This is the dominant
   effect.
2. The margin *also* falls as **drift** rises. ARC's advantage over LRU is
   **largest on stationary traffic**, not on drifting traffic — the opposite of
   the intuition that ARC is "for" drift.

ARC loses only in the corner where capacity is large *and* drift is heavy. The
same surface with the same signs reproduces on StackExchange (+13.07 at capacity
50, δ=0 → −1.67 at capacity 1600, δ=1), so this is a property of traffic, not of
Quora.

Skew matters too, non-monotonically: the advantage peaks around s ≈ 0.9–1.1 and
collapses at s=1.6, where a handful of clusters carry everything and any policy
keeps them.

### 8.2 The rule of thumb

The best single predictor is capacity relative to the **working set** — the
number of clusters covering 90% of traffic, measurable from a request log.

| capacity / working set | mean ARC − LRU | share of cells where ARC is significantly ahead |
|---|---|---|
| < 0.02 | +8.77 pp | 100% |
| 0.02 – 0.05 | +7.57 pp | 100% |
| 0.05 – 0.15 | +4.47 pp | 93% |
| 0.15 – 0.40 | +1.78 pp | 74% |

Correlation with ARC − LRU: **r = −0.71** against log10(capacity / working set),
versus **−0.36** against drift rate. Below ~15% of the working set ARC earns its
overhead; past ~40% the sign is no longer reliable. This is a *partial* collapse
— at fixed ratio the spread across skews is still a few points — but it orders
the results better than anything else measured.

### 8.3 Why the drift result is not a contradiction

ARC's own state explains it. `p` is the target size of the recency list `T1`:
low `p` means ARC is leaning on frequency, high `p` means it is behaving like
LRU. As a fraction of capacity, on Quora:

| | δ=0 | δ=0.5 | δ=1.0 |
|---|---|---|---|
| capacity 200 | 4.5% | 14.0% | 35.9% |
| capacity 400 | 5.7% | 25.6% | 44.9% |

At zero drift ARC runs almost pure frequency, and **that frequency component is
where its win over LRU comes from**. As drift rises the ghosts correctly report
that the frequency bet is failing, `p` climbs, and ARC converges *toward* LRU —
so its margin over LRU converges toward zero at the same time.

The adaptation is **insurance against collapse, not a source of gain**. That is
the right shape for a default, but it means the honest pitch is "never much
worse, often much better", not "handles drift better than LRU". What the
adaptation buys is visible against the policies that lack it, at capacity 100 on
the drifting trace: LFU 28.68%, exact-ghost ARC 36.76%, LRU 48.05%, semantic
ARC 54.21%.

Across all cells of both new corpora, `ARC-exact` holds `p_mean = 0.000`. The
ablation replicates on a corpus it was never tuned against.

### 8.4 Real traffic

| capacity | WildChat, 2 months | WildChat, 12.5 months |
|---|---|---|
| 50 | **+0.06** | **+1.31** |
| 400 | **+0.34** | **+0.53** |
| 1600 | −0.01 | **+0.23** |

Statistically significant, practically about a point. The longer trace is
friendlier to ARC — but **that comparison is confounded and a control was run**:
`wildchat-long` is subsampled to 150,000 rows, making it ~2.2× sparser in time
as well as longer. Thinning the two-month stream by the same factor recovers
+0.20 to +0.76 pp on its own. Some of the difference is span, some is sparsity,
and this data cannot separate them.

Meanwhile LFU is 10–18 points behind on the same traces.

### 8.5 Costs

Measured, not hidden. Capacity 400, drifting trace:

| | LRU | ARC |
|---|---|---|
| eviction decision | 3.37 µs/req | 11.51 µs/req |
| mean request (simulator) | 12.29 µs | 20.16 µs |
| p99 request (simulator) | 37.68 µs | 65.65 µs |
| peak heap | 841 KB | 3,310 KB |
| policy vectors retained | 0 | 800 |

At capacity 1600 the heap gap is 3.3 MB vs 15.6 MB. Ghosts hold an embedding but
no response, so the policy retains up to `2c` vectors — at `maxsize=1000` and
384-d float32, about 3 MB.

Time is the cheap axis: these are microseconds against an LLM call measured in
hundreds of milliseconds, so converting one miss into a hit repays the added
scan thousands of times over. **Memory is the real constraint** — budget for
`2c` vectors, not `c`.

---

## 9. Threats to validity

Stated plainly, because several of these materially limit the conclusions.

1. **No full GPTCache request path was benchmarked.** §6.1. Hit rates are sound;
   end-to-end latency claims are not available from this data.
2. **The large margins come from synthetic arrival orders.** Quora and
   StackExchange have no natural arrival order, so Zipf popularity and the drift
   model are invented. A total reshuffle every ~3,300 queries is aggressive and
   deliberately adversarial toward LFU. On the traces with genuine arrival order
   the margin is ~1 pp.
3. **Real-order traces have no paraphrase ground truth.** False hit rate — the
   safety-relevant metric — is only measurable on synthetic-order traffic.
4. **The long/short real comparison is confounded** with temporal density
   (§8.4), and the control shows the confound is material.
5. **One embedding model.** Every result is conditional on MiniLM-L6-v2's
   similarity geometry, and ghost matching is *entirely* a function of that
   geometry.
6. **One `tau`.** Chosen well and documented, but not swept as a robustness axis
   in the crossover study.
7. **The working-set rule is a partial collapse, not a law** (§8.2).

### What would close them

- Replay ~5k WildChat prompts through a real `Cache` + `SSDataManager` + FAISS +
  ONNX under LRU and ARC; report real hit rate and real p99. The test suite
  already builds this stack, so it is mostly wiring.
- Rebuild `wildchat-long` at full density to break the span/sparsity confound.
- Repeat the crossover grid under a second embedding model.
- Sweep `tau` as a third axis.

---

## 10. Conclusions

1. **The hypothesis holds.** Exact-key history is inert on semantic workloads —
   demonstrated twice, once for frequency sketches (r = −0.037) and once for ARC
   ghosts (`p ≡ 0.000` across every trace and both corpora). Semantic keying
   revives it.
2. **The mechanism is understood, not just observed.** ARC's advantage over LRU
   is its frequency component; the semantic ghosts exist to detect when
   frequency stops paying and retreat toward LRU. `p` traces this directly.
3. **The practical claim is robustness.** Best worst case across regimes, no
   per-workload knob. Not the best policy in any single regime.
4. **The regime where it pays is capacity-constrained**, roughly below 15% of
   the working set — not, as intuition suggests, drifting traffic.
5. **On real prompt traffic the gain is about a point.** Worth having for free;
   not worth 2× cache memory unless the cache is genuinely small relative to the
   working set. The stronger argument on real traffic is what ARC *avoids*: LFU
   is 10–18 points behind.

---

## 11. File map

**Policy**
- `gptcache/manager/eviction/arc.py` — `ARCCache` and `_VecList`
- `gptcache/manager/eviction/base.py`, `memory_cache.py`, `factory.py`,
  `manager/data_manager.py` — integration

**Tests**
- `tests/unit_tests/eviction/test_arc_cache.py` — 26 tests
- `tests/unit_tests/manager/test_eviction.py` — +1 ARC case

**Data & benchmarks**
- `benchmarks/prepare_data.py` — Quora, WildChat, `tau` selection
- `benchmarks/prepare_data_ext.py` — StackExchange, WildChat-long
- `benchmarks/traces.py` — the three fixed regimes
- `benchmarks/traces_param.py` — parameterised drift/skew
- `benchmarks/simcache.py` — semantic-cache front-end over real policies
- `benchmarks/gate_screening.py` — the pre-implementation gate
- `benchmarks/run_bench.py` / `analyze.py` — main sweep, figs 1–6
- `benchmarks/sweep_crossover.py` / `analyze_crossover.py` — crossover, figs 7–11

**Results** — `benchmarks/results/`: `summary.md`, `crossover.md`,
`gate_screening.txt`, `baseline_tests.txt`, `bench.csv`, `crossover_*.csv`,
`fig1`–`fig11`

**Docs** — `docs/eviction_policies.md` (user-facing selection guide),
`benchmarks/README.md` (reproduction), this report

---

## 12. Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . && pip install -r benchmarks/requirements.txt

# corpora (~30 min, ~4 GB download, once)
python benchmarks/prepare_data.py
python benchmarks/prepare_data_ext.py

python benchmarks/gate_screening.py     # the pre-implementation gate
python benchmarks/run_bench.py          # main sweep      (~90 s)
python benchmarks/analyze.py            # figs 1-6, summary.md
python benchmarks/sweep_crossover.py    # 5,520 cells     (~2.5 min, 12 cores)
python benchmarks/analyze_crossover.py  # figs 7-11, crossover.md

python -m pytest tests/unit_tests/eviction/ tests/unit_tests/manager/ -q
```

A pinned `Dockerfile` is in `benchmarks/`. Reference hardware for the committed
numbers: MacBook Pro, Apple M-series, 24 GB, macOS 26.4, Python 3.10.11,
numpy 2.2.6.

---

## 13. Reference

Nimrod Megiddo and Dharmendra S. Modha. *ARC: A Self-Tuning, Low Overhead
Replacement Cache.* USENIX FAST 2003.

Gil Einziger and Roy Friedman. *TinyLFU: A Highly Efficient Cache Admission
Policy.* PDP 2014. — Einziger, Friedman & Manes, ACM TOS 2017.

Datasets: Quora Question Pairs (via `sentence-transformers/quora-duplicates`);
`sentence-transformers/stackexchange-duplicates`; `allenai/WildChat-1M`.
Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
