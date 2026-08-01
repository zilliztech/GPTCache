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
| `ARC` | balances recency and frequency, and **retunes that balance as traffic changes** | traffic is mixed or drifts, or you do not know which regime you are in and do not want to tune. |

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
happening at all.

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
  drifting trace: 12.2 µs per request spent on the eviction decision against
  LRU's 4.1 µs; mean end-to-end request 21.2 µs against 14.0 µs.

Whether that is worth paying depends entirely on what a miss costs you. These
are microseconds against an LLM call measured in hundreds of milliseconds, so
converting a single miss into a hit repays the added scan overhead many
thousands of times over. The memory is the real constraint: budget for `2c`
vectors, not `c`.

## When not to use ARC

- **Stationary traffic where you know it is stationary.** Use `LFU`. It is
  simpler, cheaper, has no vector overhead, and on that regime ARC's advantage
  is small.
- **Very large caches.** Above roughly 1600 entries on these traces the
  policies converge, and ARC's ghost scan grows linearly with capacity while its
  advantage shrinks.
- **You cannot supply embeddings.** `put()` accepts entries without them —
  `SSDataManager` does this at start-up for rows already in the database — but
  those entries can never produce a ghost hit, so ARC degrades toward classic
  ARC for them.

## Reproducing these numbers

See `benchmarks/README.md`. Every figure and table above regenerates from a
clean clone with two commands.

## Reference

Nimrod Megiddo and Dharmendra S. Modha. *ARC: A Self-Tuning, Low Overhead
Replacement Cache.* USENIX FAST 2003.
