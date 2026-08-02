# Choosing an eviction policy

A cache holds a bounded number of entries, so on every miss something has to
leave. The eviction policy is the rule that picks it. GPTCache ships five
in-memory policies:

| Policy | Keeps | Use it when |
|---|---|---|
| `LRU` (default) | the most recently used entries | you have no information about the workload. A sound default. |
| `LFU` | the most frequently used entries | traffic is **stationary**: today's popular questions are tomorrow's. |
| `FIFO` | the most recently inserted entries | insertion order is meaningful and access order is not. |
| `RR` | an arbitrary subset | you want the cheapest possible decision and do not care which entry goes. |
| `ARC` | balances recency and frequency, and **retunes that balance as traffic changes** | your cache is small relative to your working set, or you do not know which regime you are in and do not want to tune. See [when ARC beats LRU](#when-arc-beats-lru-and-by-how-much). |

```python
from gptcache.manager import get_data_manager, CacheBase, VectorBase

data_manager = get_data_manager(
    CacheBase("sqlite"),
    VectorBase("faiss", dimension=384),
    max_size=1000,
    eviction="ARC",
    eviction_params={"tau": 0.8},
)
```

## The tradeoff in one paragraph

LRU and LFU each bet on one signal. LFU wins when popularity is stable and
loses badly when it is not — a question that was hot last week occupies the
cache long after anyone stops asking it. LRU never collapses that way but never
exploits stable popularity either. ARC keeps both a recency list and a
frequency list and continuously moves the boundary between them based on which
one has been making better predictions lately. It is not the best policy in any
single regime; it is the policy with the best *worst* case, and it has no knob
to set per workload.

Measured on this repository's benchmarks (`benchmarks/`, Quora paraphrase
clusters and WildChat prompts, mean of 10 seeds, capacity 100):

| Policy | Stationary | Drifting | Real prompt stream | **Worst case** |
|---|---|---|---|---|
| LRU (default) | 48.87% | 48.05% | 28.90% | 28.90% |
| LFU | 56.53% | 28.68% | 11.96% | **11.96%** |
| FIFO | 43.56% | 42.94% | 28.50% | 28.50% |
| RR | 43.73% | 42.68% | 27.21% | 27.21% |
| **ARC** | 58.08% | 54.21% | 29.02% | **29.02%** |

Read that table honestly: **LFU beats plain LRU by 7.7 points on stationary
traffic**, and on that regime ARC's margin over LFU is only about 1.5 points.
The case for ARC is the last column. LFU's worst case is 12%; ARC's is 29%, and
it is the best in every column but never by a landslide.

At large capacities the policies converge — at 1600 entries on the drifting
trace every policy lands within a point of the others, because the cache is
large enough that the replacement decision stops mattering. The gains are in
the capacity-constrained regime, which is the regime you are in if eviction is
happening at all. The next section makes that precise.

## When ARC beats LRU, and by how much

The three workloads above establish the worst-case claim but cannot say *what
property of a workload* decides the winner, because the two synthetic regimes
are the endpoints of an axis with nothing sampled between them.
`benchmarks/sweep_crossover.py` makes the axis continuous: `drift_rate` is the
fraction of the popularity ranking re-permuted each epoch, so 0.0 is the
stationary regime, 1.0 is the drifting one, and everything between is measured.
Numbers below are the mean of 5 seeds with 95% paired bootstrap CIs.

**The answer is mostly about capacity, not drift.** ARC − LRU on Quora at
Zipf s=1.1, in percentage points of hit rate (bold = 95% paired bootstrap CI
excludes zero):

| capacity | drift 0 | drift 0.2 | drift 0.5 | drift 1.0 |
|---|---|---|---|---|
| 50 | **+10.93** | **+10.25** | **+9.58** | **+9.05** |
| 100 | **+9.23** | **+8.75** | **+7.51** | **+6.21** |
| 200 | **+7.46** | **+6.40** | **+4.84** | **+2.49** |
| 400 | **+5.38** | **+3.87** | **+2.15** | +0.10 |
| 800 | **+3.08** | **+1.57** | **+0.35** | **−1.16** |
| 1600 | **+0.88** | **+0.36** | **−0.25** | **−0.34** |

Two things to read off it. First, the margin falls monotonically as capacity
grows — that is the dominant effect. Second, and against intuition, **the
margin also falls as drift rises**: ARC's advantage over LRU is *largest* on
stationary traffic, not on drifting traffic. ARC loses to LRU only in the
corner where capacity is large *and* drift is heavy. The same surface, with the
same signs, reproduces on a second ground-truth corpus (StackExchange
duplicate titles, 35,878 clusters): +13.07 at capacity 50 falling to −1.67 at
capacity 1600 under total reshuffle.

### The rule of thumb

The single best predictor is capacity relative to the **working set** — the
number of distinct semantic clusters covering 90% of your traffic, which you
can measure from a request log. Correlation with ARC − LRU is r = **−0.71**
against log10(capacity / working set), against −0.36 for drift rate.

| capacity / working set | mean ARC − LRU | share of cells where ARC is significantly ahead |
|---|---|---|
| < 0.02 | +8.77 pp | 100% |
| 0.02 – 0.05 | +7.57 pp | 100% |
| 0.05 – 0.15 | +4.47 pp | 93% |
| 0.15 – 0.40 | +1.78 pp | 74% |

Below roughly 15% of the working set, ARC is worth its overhead. Above it the
margin is small and the sign is no longer reliable. This is a partial collapse,
not a law — at a fixed ratio the spread across popularity skews is still a few
points — but it orders the results better than anything else measured.

### On real traffic the margin is about a point

Both synthetic regimes above invent an arrival order. Two traces do not:
`wildchat` is two months of real timestamped prompts, `wildchat-long` is
150,000 prompts spanning April 2023 to April 2024. ARC − LRU:

| capacity | WildChat, 2 months | WildChat, 12.5 months |
|---|---|---|
| 50 | **+0.06** | **+1.31** |
| 400 | **+0.34** | **+0.53** |
| 1600 | −0.01 | **+0.23** |

Significant, small. The longer trace is friendlier to ARC, but that comparison
is confounded and the benchmark says so: `wildchat-long` is subsampled to
150,000 rows, making it ~2.2x sparser in time as well as longer, and a control
that thins the two-month stream by the same factor recovers +0.20 to +0.76 pp
on its own. Some of the difference is span, some is sparsity, and this data
cannot separate them.

The practical reading: on a real prompt stream, expect ARC to match LRU or beat
it by around a point — and to save you from LFU, which is 10 to 18 points
behind on the same traces.

### Why the drift result is not a contradiction

ARC's own adaptive state explains it. `p` is the target size of the recency
list `T1`; a low `p` means ARC is leaning on frequency, a high `p` means it is
leaning on recency and therefore behaving like LRU. Measured as a fraction of
capacity on Quora:

| | drift 0 | drift 0.5 | drift 1.0 |
|---|---|---|---|
| capacity 200 | 4.5% | 14.0% | 35.9% |
| capacity 400 | 5.7% | 25.6% | 44.9% |

At zero drift ARC runs almost pure frequency, and that frequency component is
where its win over LRU comes from. As drift rises, the ghosts correctly report
that the frequency bet is failing, `p` climbs, and ARC converges *toward* LRU —
so its margin over LRU converges toward zero at the same time. The adaptation
is insurance against the LFU-style collapse, not a source of gain. That is
exactly the shape you want from a default, but it means the honest pitch is
"never much worse, often much better", not "handles drift better than LRU".

What the adaptation buys is visible by comparing against the policies that
cannot do it, at capacity 100 on the drifting trace: LFU 28.68%, classic
exact-ghost ARC 36.76%, LRU 48.05%, **semantic-ghost ARC 54.21%**.

## What is different about ARC here

Standard ARC (Megiddo & Modha, *ARC: A Self-Tuning, Low Overhead Replacement
Cache*, USENIX FAST 2003) does its self-tuning with two **ghost lists**: `B1`
and `B2` remember the identities of recently evicted entries, holding no cached
response. A returning query that hits a ghost is a signal that the eviction was
a mistake, and the recency/frequency boundary `p` moves accordingly.

In a semantic cache that mechanism does not work as written. GPTCache assigns a
fresh row id to every miss, and the queries that ought to recognise each other
are *paraphrases* — different strings, different ids. An identity-keyed ghost
can only fire on an id that by construction never returns. So the ghosts fill
up, never match, `p` stays at exactly 0 for the entire trace, and ARC's adaptive
machinery is not merely suboptimal — it is inert, silently degrading to
something close to plain LRU.

GPTCache's ARC therefore matches ghosts by **embedding similarity** instead of
by id: a ghost fires when its cosine similarity to the incoming query is at
least `tau`. This is the only change to the algorithm; Cases I–IV are otherwise
Figure 4 of the paper. It is what makes the adaptation work, and the effect is
large — on drifting traffic at capacity 100, semantic ghosts reach 54.21%
against exact-key ghosts' 36.76%.

## Tunables

| Param | Default | Effect |
|---|---|---|
| `maxsize` | `1000` | Resident capacity `c`: how many entries hold a cached response. |
| `tau` | `0.8` | Ghost match threshold. **Set this to the same threshold your similarity evaluation serves at.** Higher is stricter: fewer ghost hits, slower adaptation, and in the limit ARC stops adapting. Lower means ghosts fire on loosely related queries and `p` moves on noise. |
| `ghost_matching` | `"semantic"` | `"exact"` reproduces classic ARC. It exists for the ablation and to document the failure mode; there is no reason to use it in production. |
| `clean_size` | ignored | ARC decides and performs its own eviction, releasing one entry at a time. The cachetools-backed policies instead drop `clean_size` entries in a batch (default `0.2 * maxsize`), so they run at 80–100% occupancy where ARC runs at 100%. |

## Costs, stated plainly

ARC is not free, and the benchmark measures the cost rather than hiding it.

- **Memory.** Ghosts retain an embedding but no response, so the policy holds up
  to `2 * maxsize` vectors. At `maxsize=1000` and 384-d float32 that is about
  3 MB. Measured peak heap at capacity 1600: 15.6 MB for ARC against 3.3 MB for
  LRU.
- **Time.** A miss scans both ghost lists — O(`2 * maxsize`) similarity
  computations — against LRU's O(1) `popitem`. Measured at capacity 400 on the
  drifting trace: **11.51 µs per request spent on the eviction decision against
  LRU's 3.37 µs**. Mean whole-request time in the benchmark harness is 20.16 µs
  against 12.29 µs — but note that is a *simulator* request, not a GPTCache
  one: the harness replaces the vector and scalar stores with numpy and a dict
  (see `benchmarks/README.md` §0). Real GPTCache adds ONNX embedding, FAISS and
  SQLite on top, which are milliseconds. The eviction-decision figure is the one
  measured on shipped code, and it is the one that matters here.

Whether that is worth paying depends entirely on what a miss costs you. These
are microseconds against an LLM call measured in hundreds of milliseconds, so
converting a single miss into a hit repays the added scan overhead many
thousands of times over. The memory is the real constraint: budget for `2c`
vectors, not `c`.

## When not to use ARC

- **Stationary traffic where you know it is stationary.** Use `LFU`. It is
  simpler, cheaper, has no vector overhead, and on that regime ARC's advantage
  over it is about 1.5 points (58.08% against 56.53% at capacity 100). Note
  the conditional: LFU is the right call only if you are confident popularity
  will not move, because the same table shows it at 28.68% when it does.
- **Caches that are large relative to the working set.** Once capacity passes
  roughly 15% of the clusters covering 90% of your traffic the margin drops
  under two points, and past 40% its sign is no longer reliable — if traffic
  also drifts hard, ARC is measurably *behind* LRU (−1.16 pp at capacity 800
  under total reshuffle). Meanwhile the ghost scan grows linearly with
  capacity. Two bad trends at once: use `LRU`.
- **You cannot supply embeddings.** `put()` accepts entries without them —
  `SSDataManager` does this at start-up for rows already in the database — but
  those entries can never produce a ghost hit, so ARC degrades toward classic
  ARC for them.

## Reproducing these numbers

See `benchmarks/README.md`. The three-workload comparison regenerates from a
clean clone with two commands; the crossover study in this document is
section 7 of that README and adds two more corpora and a 5,520-cell sweep.

## Reference

Nimrod Megiddo and Dharmendra S. Modha. *ARC: A Self-Tuning, Low Overhead
Replacement Cache.* USENIX FAST 2003.
