# How to benchmark GPTCache eviction policies

Everything below runs from a clean clone with no other instructions. Total
runtime is about 20 minutes on an M-series MacBook Pro, most of it spent
embedding text once.

## 0. What this measures

Seven policies on three traces at six capacities with ten seeds — 1260 runs.
Every run drives the **real GPTCache eviction classes** (`MemoryCacheEviction`,
`ARCCache`, `cachetools`) through a thin semantic-cache front end that supplies
only the part a production deployment gets from its vector store: resident
nearest-neighbour lookup. Nothing here re-implements a policy.

| Policy | What it is |
|---|---|
| `LRU` `LFU` `FIFO` `RR` | GPTCache's existing policies, `clean_size=1` |
| `ARC` | the new policy, semantic ghost lists |
| `ARC-exact` | the same code with `ghost_matching="exact"` — the ablation |
| `LRU-batch` | GPTCache's *untouched* default, `clean_size = 0.2 * maxsize` |

## 1. Install

```bash
git clone https://github.com/zilliztech/GPTCache.git && cd GPTCache
git checkout feature/semantic-arc

python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r benchmarks/requirements.txt
```

Python 3.10 or newer. Everything runs on CPU; the embedding step uses Apple MPS
or CUDA automatically if present.

## 2. Build the traces  (~10 min, ~700 MB download, once)

```bash
python benchmarks/prepare_data.py
```

Downloads two public datasets, builds paraphrase clusters, embeds every text
with `sentence-transformers/all-MiniLM-L6-v2`, and writes unit-norm float32
arrays to `benchmarks/data/` (gitignored — this script is how you regenerate
them).

| Corpus | Source | What it provides |
|---|---|---|
| `quora` | Quora Question Pairs | 53,250 questions in 12,235 paraphrase clusters, built by union-find over `is_duplicate=True` edges. The cluster ids are **ground truth**, which is what makes false hit rate measurable. |
| `wildchat` | [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | 71,637 first-turn English prompts in **real timestamp order**, spanning 2023-04-09 to 2023-06-07. Whatever drift is present actually happened. |

> The project brief specified LMSYS-Chat-1M for the second trace. That dataset
> is gated and was not available to this account; WildChat-1M is the ungated
> equivalent and additionally carries timestamps, which LMSYS encodes only
> implicitly as row order.

The script also selects `tau` and writes it to `data/tau.json`; every later
step reads it from there. Expected output ends with:

```
  [B] serving-decision F1 on a simulated cache (cap=400, 10000 queries)  <- USED
      best F1 = 0.9751 at tau = 0.800 (precision 0.9912, recall 0.9595, false-hit rate 0.0088)
  wrote data/tau.json -- every later experiment uses tau=0.8
```

<details>
<summary>Why tau is not chosen by pairwise F1</summary>

The brief specifies "the value maximising F1 between 'same cluster' and
'sim >= tau'". Taken literally over random question *pairs*, that criterion is
degenerate on this corpus, and `prepare_data.py` prints the evidence rather than
quietly working around it: the F1 curve is flat to ~1e-4 across tau in
[0.40, 0.50], so its argmax is arbitrary; and a cache never classifies a random
pair — it takes the **maximum** similarity over every resident entry, so the
per-pair false-positive rate compounds with capacity. The pairwise argmax
(tau=0.43) makes a 1600-entry cache find a wrong-cluster neighbour above
threshold for ~85% of queries.

The same F1 objective, evaluated on the decision the cache actually makes,
gives tau=0.80 at a 0.88% false-hit rate. Both numbers are printed and both are
recorded in `tau.json`; only the second is used.
</details>

## 3. Run the gate  (~2 min)

```bash
python benchmarks/gate_screening.py
```

Reruns, on real embeddings, the two screening measurements the whole project
rests on. **It exits non-zero if either fails.**

| Check | Pass condition | Actual |
|---|---|---|
| An exact-key frequency sketch carries no signal | \|corr\| < 0.15 | **+0.005** |
| Semantic ghosts beat exact ghosts on drift | ≥ 3pp, every capacity | **+11.2 to +18.2pp** |

Writes `results/gate_screening.{json,txt}`.

## 4. Sweep  (~90 s)

```bash
python benchmarks/run_bench.py --jobs 6        # --quick for a 30 s smoke test
```

Writes three CSVs to `results/`: `bench.csv` (one row per
trace×policy×capacity×seed), `p_trace.csv` (ARC's adaptation state over time),
`lat_cdf.csv` (latency quantile grids).

Metrics per row: hit rate; false hit rate; latency mean/p50/p95/p99; eviction
decision latency; throughput; peak heap; vectors retained by the policy;
mean residency; ARC's `p` final/max/mean.

<details>
<summary>Two measurement decisions worth knowing about</summary>

**`clean_size` is controlled.** GPTCache's default releases `0.2 * maxsize`
entries per eviction, so a cachetools policy spends its life between 80% and
100% full while ARC, releasing one at a time, sits at 100%. Left alone, that
occupancy gap would show up as an ARC win having nothing to do with the
replacement decision. So the headline comparison pins every cachetools policy
to `clean_size=1`, `LRU-batch` carries the untouched default for reference, and
`resident_mean` is on every row so you can check the control held.

**Peak heap is `tracemalloc`, not RSS.** RSS on this workload is dominated by
the shared corpus and by allocator behaviour, is not attributable to the policy,
and is not reproducible across parallel workers. `tracemalloc` peak is all
three. The analytic cost is reported alongside as `policy_vectors`.
</details>

## 5. Figures and tables  (~10 s)

```bash
python benchmarks/analyze.py
```

Reads only the CSVs, so figures always regenerate from committed data without
rerunning the sweep. Writes to `results/`:

| File | Content |
|---|---|
| `fig1_hitrate_vs_capacity.png` | hit rate vs capacity per trace, 95% CI bands |
| `fig2_latency_cdf.png` | per-request latency CDF |
| `fig3_ablation.png` | ARC vs ARC-exact |
| `fig4_p_over_time.png` | `p` trajectory — the flat line is the whole argument |
| `fig5_sketch_inertness.png` | the same failure mode in a second policy family |
| `fig6_false_hit_rate.png` | wrong-cluster hits |
| `summary.md`, `summary.csv` | improvement vs LRU with paired bootstrap CIs |

Policies are compared with a **paired bootstrap**: every policy sees the same
ten seeds on the same trace, and a seed fixes the arrival sequence, so runs pair
seed-by-seed. Each of 10,000 resamples averages the per-seed *difference*
against LRU, cancelling between-seed variance. Intervals are 95% percentile
intervals.

## 6. Expected headline result

At capacity 100, mean of 10 seeds:

| Policy | Stationary | Drifting | Real prompts | Worst case |
|---|---|---|---|---|
| LRU | 48.87% | 48.05% | 28.90% | 28.90% |
| LFU | 56.53% | 28.68% | 11.96% | 11.96% |
| ARC-exact | 58.13% | 36.76% | 16.36% | 16.36% |
| **ARC** | **58.08%** | **54.21%** | **29.02%** | **29.02%** |

The claim is the last column: best worst case, no per-workload tuning. LFU beats
LRU by 7.7 points on stationary traffic and collapses to 12% on real prompts;
that loss is what makes the robustness claim worth anything.

## Determinism

Every correctness metric is bit-for-bit reproducible. Verified by running the
full 1260-cell sweep twice and diffing: max |delta| over `hit_rate`,
`false_hit_rate`, `n_hits`, `p_final` and `p_max` is exactly `0.0`.

That took two fixes worth knowing about, because both are easy to get wrong:

- **`RR` was the only nondeterministic policy.** `cachetools.RRCache` picks its
  victim with the unseeded global `random`, so RR moved by up to 0.78pp between
  runs while every other policy was already exact. `run_bench.py` now passes a
  per-cell seeded `choice`, so RR is random *within* a run — the property being
  benchmarked — but identical across runs.
- **The exact-key sketch in `gate_screening.py` used the builtin `hash()`**,
  which Python salts per process for `str`. That silently affected only the
  exact-key arm, which is precisely what the gate measures. It now uses a
  blake2b-based stable hash.

Embedding is deterministic given the same batch size: re-encoding the same texts
in a fresh process yields bitwise-identical vectors on both CPU and MPS
(verified). Changing `--batch-size` perturbs them at the 1e-7 level, which can
flip a handful of near-threshold similarity decisions, so regenerate with the
default 256 to match the committed figures. `data/*_meta.json` records the
sha256 of every input file, the model name and the dimensions.

Latency, throughput and heap figures are hardware-dependent by nature and will
not match exactly; hit rates will.

Reference hardware for the committed numbers: MacBook Pro, Apple M-series,
24 GB, macOS 26.4, Python 3.10.11, numpy 2.2.6.
