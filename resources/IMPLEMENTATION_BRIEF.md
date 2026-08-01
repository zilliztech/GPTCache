# Semantic-ARC for GPTCache — implementation brief

Hand this to Claude Code inside a clone of `zilliztech/GPTCache`.
Work on branch `feature/semantic-arc`. Do steps in order. Do not skip Step 3.

---

## 0. Context and guardrails

**Goal.** Add a new in-memory eviction policy, `ARC`, to GPTCache, whose ghost
lists are matched by *embedding similarity* rather than exact key.

**The thesis.** Modern cache policies keep history about non-resident items
(ARC's ghost lists, TinyLFU's frequency sketch). All key that history by exact
item identity. A semantic cache matches resident entries by embedding
similarity but its history by exact string — and paraphrases never repeat
verbatim, so the history never fires and the adaptive machinery is silently
dead. Making the history semantic revives it.

**Guardrails — these are not optional:**

- **Never claim W-TinyLFU or GDSF "underperform."** The course professor,
  Gil Einziger, invented TinyLFU (Einziger & Friedman, PDP 2014; Einziger,
  Friedman & Manes, ACM TOS 2017). Claim only the measurable thing: *an
  exact-key frequency sketch carries no signal on semantic workloads.* That
  claim is independent of any policy implementation.
- **Do not implement an LSH-keyed sketch.** Out of scope. It is Future Work.
- **Do not modify** the existing LRU / LFU / FIFO / RR paths. All existing
  tests must pass unmodified.
- Existing open PRs #689 (GDSF) and #680 (W-TinyLFU) are classmates' work.
  Do not duplicate, do not depend on, do not disparage.

**Grading weights** (drive every tradeoff): correctness 40%, reproducibility
30%, performance gain 15%, clarity 15%. When in doubt, choose the boring
correct thing over the clever thing.

---

## 1. Setup and baseline verification

```bash
git clone https://github.com/zilliztech/GPTCache.git && cd GPTCache
git checkout -b feature/semantic-arc
python3 -m venv .venv && source .venv/bin/activate
pip install -e . && pip install pytest numpy sentence-transformers datasets matplotlib
python -m pytest tests/unit_tests/eviction/ tests/unit_tests/manager/ -q
```

**Acceptance:** existing eviction tests pass before you change anything.
Record the output — it is the "no regression" baseline for the report.

Read these files before writing code:

| File | Why |
|---|---|
| `gptcache/manager/eviction/memory_cache.py` | `MemoryCacheEviction` — the `if/elif` policy chain you extend |
| `gptcache/manager/eviction/base.py` | `EvictionBase` ABC — `put()` signature lives here |
| `gptcache/manager/eviction/manager.py` | policy registry / factory |
| `gptcache/manager/data_manager.py` | `SSDataManager`; `import_data()` calls `eviction_base.put(ids)`; `_clear()` is the `on_evict` callback |
| `gptcache/manager/eviction_manager.py` | `soft_evict` / `check_evict` / `delete` |

---

## 2. Data preparation

Build two real traces. Cache embeddings to `.npy` so benchmarks are
deterministic and rerunnable offline — this is what buys the 30%
reproducibility score.

**Script:** `benchmarks/prepare_data.py`

1. **Quora Question Pairs** (`quora` on HF datasets). Take pairs labelled
   `is_duplicate=True` to build paraphrase clusters via union-find over
   duplicate pairs. Ground-truth cluster IDs let you measure **false hit
   rate**, not just hit rate.
2. **LMSYS-Chat-1M** — real LLM prompts in real arrival order. Gives natural
   drift instead of a synthetic drift model. Take the first ~50k first-turn
   user prompts.

Embed with `sentence-transformers/all-MiniLM-L6-v2` (384-d, CPU, fast on M-series).
Normalise to unit length so cosine similarity is a dot product.

Emit to `benchmarks/data/`:
- `{name}_emb.npy` — float32 `[n, 384]`, unit-norm
- `{name}_cid.npy` — int32 `[n]` cluster ids (`-1` where unknown)
- `{name}_meta.json` — model name, dims, count, sha256 of inputs

**Acceptance:** print intra-cluster and inter-cluster cosine similarity
percentiles. Pick `tau` as the value maximising F1 between "same cluster" and
"sim >= tau" on Quora. Record it; every later experiment uses that `tau`.

---

## 3. GATE — replicate the screening on real embeddings

**Do not write policy code until this passes.**

Take `policy_sim.py` (attached separately). Replace `make_trace` with a loader
that reads the `.npy` files from Step 2. Rerun the two screening results:

| Check | Synthetic result to reproduce | Pass condition |
|---|---|---|
| Exact-key sketch is inert | mean est 0.21, corr **−0.059** | corr with true cluster popularity ≈ 0 (\|corr\| < 0.15) |
| Semantic ghosts beat exact ghosts | 68.63% vs 52.63% on drift | semantic-ghost ARC > exact-ghost ARC by a clear margin |

**If either fails, STOP and report the numbers.** Do not proceed to
implementation. The whole project rests on these two measurements.

---

## 4. Implement `ARCCache`

**New file:** `gptcache/manager/eviction/arc.py`

Self-contained, stdlib + numpy only. No GPTCache imports beyond `EvictionBase`.

### State

```
c                  capacity (max resident entries)
p                  float in [0, c], adaptive target size for T1
T1, T2             resident: OrderedDict id -> None  (LRU order, left = LRU)
B1, B2             ghost:    OrderedDict id -> None
vec                dict id -> np.ndarray (unit-norm), for T1/T2/B1/B2 members
tau                similarity threshold for ghost matching
on_evict           callback(list_of_ids) — fires ONLY on real eviction
```

Ghosts hold an embedding but no cached response. Memory cost is `2c` vectors;
at `c=1000`, 384-d float32 that is ~3 MB. Document this in the docstring.

### Invariants (assert these in tests)

```
|T1| + |T2| <= c
|T1| + |B1| <= c
|T2| + |B2| <= 2c
|T1| + |T2| + |B1| + |B2| <= 2c
0 <= p <= c
if |T1| + |T2| < c:  B1 and B2 are empty
```

### `REPLACE(in_b2, p)`

```
if T1 non-empty and (|T1| > p or (in_b2 and |T1| == p)):
    k = pop LRU of T1;  push k to MRU of B1
else:
    k = pop LRU of T2;  push k to MRU of B2
on_evict([k])          # entry leaves the cache; embedding stays for the ghost
```

### The semantic modification — this is the contribution

Classic ARC tests `x in B1` by exact key. Replace **every ghost membership
test** with nearest-neighbour search:

```
def _match(self, d, q):
    """Return the id in d whose embedding is nearest q, if sim >= tau."""
    if not d: return None
    ids = list(d)
    sims = np.stack([self.vec[i] for i in ids]) @ q
    j = int(sims.argmax())
    return ids[j] if sims[j] >= self.tau else None
```

Resident lookup (T1/T2) is **not** done here — GPTCache's vector store already
handles that and calls `get(id)` on a hit. Only the ghosts need this.

Gate it behind a constructor flag for the ablation:

```
ghost_matching: str = "semantic"   # {"semantic", "exact"}
```

`"exact"` uses `id in d` instead of `_match`. This one flag *is* Ablation 1.

### Operations

`get(id)` — resident hit. Case I: if `id in T1`, move to MRU of T2. Elif
`id in T2`, move to MRU of T2. This is the promotion path.

`put(ids, embeddings)` — called on a miss, after the LLM responded.
For each `(id, q)`:

```
gb1 = self._match(B1, q)          # ghost hit in B1?
gb2 = self._match(B2, q)          # ghost hit in B2?

if gb1 is not None:               # Case II — recency was right, grow T1
    p = min(c, p + max(1, |B2| / max(|B1|, 1)))
    REPLACE(in_b2=False, p)
    remove gb1 from B1
    insert id at MRU of T2

elif gb2 is not None:             # Case III — frequency was right, shrink T1
    p = max(0, p - max(1, |B1| / max(|B2|, 1)))
    REPLACE(in_b2=True, p)
    remove gb2 from B2
    insert id at MRU of T2

else:                             # Case IV — true miss
    if |T1| + |B1| == c:
        if |T1| < c:  drop LRU of B1;  REPLACE(in_b2=False, p)
        else:         k = pop LRU of T1;  on_evict([k]);  del vec[k]
    elif |T1| + |B1| < c and (|T1|+|T2|+|B1|+|B2|) >= c:
        if (|T1|+|T2|+|B1|+|B2|) == 2*c:  drop LRU of B2
        REPLACE(in_b2=False, p)
    insert id at MRU of T1
```

Whenever a ghost id is dropped from B1/B2 entirely, `del self.vec[id]`.

### Reference

Megiddo & Modha, *ARC: A Self-Tuning, Low Overhead Replacement Cache*,
FAST 2003 — Figure 4 has the canonical pseudocode. Diff your Case I–IV against
it line by line. Cite it in the docstring and the report.

---

## 5. Wire into GPTCache

Follow the shape of PR #689 (GDSF), which solved the same plumbing problem.

**`gptcache/manager/eviction/base.py`** — extend the ABC:

```python
def put(self, objs: List[Any], embeddings: Optional[List[Any]] = None):
```

Optional and defaulted, so every existing implementation is unaffected.

**`gptcache/manager/eviction/memory_cache.py`** — add one branch to the chain:

```python
elif self._policy == "ARC":
    self._cache = ARCCache(maxsize=maxsize, tau=tau,
                           ghost_matching=ghost_matching, on_evict=on_evict)
```

`ARCCache` manages its own eviction, so **do not** wrap it in
`popitem_wrapper` — that helper exists for the `cachetools` policies.
Keep `put`/`get` signatures identical to the other policies.

**`gptcache/manager/data_manager.py`** — in `import_data`, pass the embeddings
that are already in scope through to `put`:

```python
self.eviction_base.put(ids, embeddings=embedding_datas)
```

**`distributed_cache.py`, `redis_eviction.py`** — accept and ignore the new
kwarg, exactly as #689 did. Three lines each.

**Acceptance:** `python -m pytest tests/unit_tests/ -q` passes with zero
changes to existing test files. Default behaviour (`policy="LRU"`) is
byte-identical to `main`.

---

## 6. Tests

**New file:** `tests/unit_tests/eviction/test_arc_cache.py`. Target ~15 tests.

*Correctness is 40% of the grade — this file is the single highest-value
artifact in the project.*

1. **Invariant fuzz test.** 5000 random ops against `c=50`; assert all six
   invariants after *every* operation. This is the test that proves ARC is
   implemented correctly.
2. Capacity is never exceeded.
3. `on_evict` fires exactly once per evicted id, never for a ghost demotion
   that was already counted.
4. `p` stays in `[0, c]` under adversarial sequences.
5. `p` **moves** under `ghost_matching="semantic"` on a paraphrase trace.
6. `p` **stays at 0** under `ghost_matching="exact"` on the same trace.
   *(This is the ablation, encoded as a regression test.)*
7. Ghost embeddings are freed when a ghost is dropped — no unbounded growth in
   `vec` over 10k ops.
8. Degenerate cases: `c=1`; empty cache `get`; duplicate id in `put`.
9. Cold start: first `c` inserts evict nothing.
10. End-to-end through `get_data_manager(..., policy="ARC")`.

Add a parametrised case to `tests/unit_tests/manager/test_eviction.py` so
`"ARC"` is covered by the existing policy matrix.

---

## 7. Benchmark harness

**New dir:** `benchmarks/`. Keep it out of the package.

```
benchmarks/
  prepare_data.py     # Step 2
  run_bench.py        # sweeps, emits CSV
  analyze.py          # CSV -> figures + summary table
  data/               # .npy, gitignored; regenerate via prepare_data.py
  results/            # CSV + PNG, committed
  README.md           # "how to benchmark" — required deliverable
```

`run_bench.py` sweeps:

- policies: `LRU, LFU, FIFO, RR, ARC, ARC-exact-ghosts`
- capacities: `50, 100, 200, 400, 800, 1600`
- traces: `quora`, `lmsys`
- seeds: **10** (match #689's rigour)

Metrics per run — collect all of these, the brief asks for them explicitly:

| Metric | Note |
|---|---|
| hit rate | primary |
| **false hit rate** | hits where cluster id differs — semantic caches trade correctness for speed, and almost nobody measures this |
| per-request latency mean / p95 / p99 | |
| eviction-decision latency | ARC's ghost scan costs more than LRU's `popitem` — measure it, don't hide it |
| peak RSS | ghost vectors cost memory; quantify |
| throughput (queries/s) | |

Emit tidy CSV: one row per `(trace, policy, capacity, seed)`.

`analyze.py` computes **paired bootstrap 95% CIs** across seeds (same seeds,
same trace, policies paired) and emits:

- hit rate vs capacity, one line per policy, CI bands
- latency CDF at fixed capacity
- ablation bar chart: ARC vs ARC-exact-ghosts
- a summary table of relative improvement vs LRU at each capacity

Axes labelled, legends present, fonts >= 10pt, consistent figure sizes.

---

## 8. Ablation

Two runs, one flag. Report:

| variant | hit rate | final `p` |
|---|---|---|
| LRU (GPTCache default) | | — |
| ARC, `ghost_matching="exact"` | | expect **0** — never adapts |
| ARC, `ghost_matching="semantic"` | | expect nonzero — adapts |

Plot `p` over time for both. The exact-ghost line pinned flat at zero is the
single most persuasive figure in the report — it shows the adaptation
machinery is *inert*, not merely suboptimal.

Add the sketch-inertness measurement from Step 3 as a **one-figure** section
showing the same failure mode in a second policy family. Keep it to one
figure and one paragraph. Do not implement anything for it.

---

## 9. Documentation

Required, and it feeds the 15% clarity score.

**Docstrings** — every public method on `ARCCache`: purpose, args, returns,
complexity. Class docstring covers: what ARC is, the Megiddo & Modha citation,
what "semantic ghosts" changes and why, memory cost (`2c` vectors), and the
`ghost_matching` flag.

**`docs/` page** or a `README.md` section:

- when to use ARC (mixed / drifting traffic) vs LFU (stable) vs LRU (default)
- every tunable, with its effect on performance:

| Param | Default | Effect |
|---|---|---|
| `maxsize` | — | resident capacity `c` |
| `tau` | inherit from cache | ghost match threshold; higher = stricter, fewer ghost hits, slower adaptation |
| `ghost_matching` | `"semantic"` | `"exact"` reproduces classic ARC (ablation only) |

- a worked example mirroring the existing `EvictionBase` docstring style
- honest statement of costs: ghost scan is O(2c) per miss; memory is `2c`
  vectors

**`benchmarks/README.md`** — "how to benchmark", a required deliverable:
exact commands from clean clone to figures, expected runtime, expected output
files. Someone must be able to clone and reproduce with no other instructions.

**`Dockerfile`** or `environment.yml` pinning Python and all versions.
Test it from a clean clone — reproducibility is 30%.

---

## 10. Report (8–12 pages, PDF)

Follow the brief's mandated structure:

1. **Introduction** — semantic caching, GPTCache architecture, its
   LRU/LFU/FIFO/RR policies; related work on ARC and TinyLFU.
2. **Extension design** — the exact-identity-history thesis; why ghosts
   never fire on paraphrases; the semantic ghost fix.
3. **Experimental setup** — traces, embedding model, `tau` selection, metrics,
   seeds, hardware (M5 MacBook Pro, 24 GB).
4. **Results** — hit rate vs capacity; false hit rate; latency; overhead.
5. **Discussion** — where ARC loses. **Say plainly that LFU beats ARC on
   stationary traffic.** Frame the contribution as *robustness*: best
   worst-case across regimes, no tuning knob. Reporting the loss is what makes
   the win credible.
6. **Conclusion & future work** — LSH-keyed sketches as the natural next step
   for the TinyLFU family.

Appendix: link every script and data artifact.

---

## 11. Pull request

Model it on #689 — that PR is the quality bar: 271 lines, 12 tests, 10 seeds,
paired bootstrap CIs, separate benchmark repo linked.

PR body: what it adds, why (the inert-ghosts diagnosis with numbers),
backward compatibility (existing policies untouched, `put()` kwarg optional,
all existing tests unmodified), and a link to your benchmark repo.

Sign commits (`git commit -s`) — GPTCache requires DCO, and several open PRs
are stuck on `needs-dco`.

**Expect it not to merge.** GPTCache has merged ~2 PRs since mid-2024 and
states it is in maintenance mode. Submit for the record; the grade must not
depend on it.

---

## Order of work (~3 person-days across 2 people)

| Day | Work | Gate |
|---|---|---|
| 1 AM | Steps 1–2 | baseline tests pass; traces built |
| 1 PM | **Step 3** | **both screening results replicate, or STOP** |
| 2 AM | Step 4 | invariant fuzz test green |
| 2 PM | Steps 5–6 | full suite green, no regressions |
| 3 AM | Steps 7–8 | CSVs + figures |
| 3 PM | Steps 9–11 | docs, PDF, PR |

Parallelise: one person on Steps 4+6 (policy and tests), the other on
Steps 2+7 (data and harness). They meet at Step 5.
