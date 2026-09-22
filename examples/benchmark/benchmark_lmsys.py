"""Full-stack benchmark: LLM conversation data, all eviction policies.

Differences vs benchmark_eviction.py:
  - Runs the full GPTCache stack: sentence-transformers → FAISS → eviction
  - Uses real LLM conversation data (LMSYS-Chat-1M or UltraChat-200K)
  - "Hit" means semantic similarity >= threshold, not exact key match
  - Measures latency, memory, and throughput in addition to hit rate

Additional (our thesis metric):
  - cost_weighted_hit_rate = Σ(LLMCost on hits) / Σ(LLMCost on all queries)
    where LLMCost = latency_ms × model_tier × (1 + tokens/1000)

Datasets:
  --dataset ultrachat  (default, public, no login required)
  --dataset lmsys      (requires HF access: huggingface.co/datasets/lmsys/lmsys-chat-1m)

Usage:
  python examples/benchmark/benchmark_lmsys.py
  python examples/benchmark/benchmark_lmsys.py --dataset lmsys --n-queries 3000
  python examples/benchmark/benchmark_lmsys.py --policies LRU,LFU,CA_W_TINYLFU
  python examples/benchmark/benchmark_lmsys.py --cache-sizes 50,100,200 --repeats 3
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ensure repo root is importable when run directly
# ---------------------------------------------------------------------------
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.manager.eviction.ca_w_tinylfu import LLMCost


# ---------------------------------------------------------------------------
# Similarity threshold helpers
# ---------------------------------------------------------------------------
# GPTCache uses SearchDistanceEvaluation whose range() = (0, 4), where
# score = 4 - squared_L2.  A hit is: score >= 4 * similarity_threshold
# → squared_L2 <= 4 * (1 - similarity_threshold)
#
# similarity_threshold=0.85 → sq_dist <= 0.60  (cosine sim ≥ 0.70)
# similarity_threshold=0.80 → sq_dist <= 0.80  (cosine sim ≥ 0.60)

def sq_dist_threshold(similarity_threshold: float) -> float:
    return 4.0 * (1.0 - similarity_threshold)


# ---------------------------------------------------------------------------
# Model-tier helper (for LLMCost and cost_weighted_hit_rate)
# ---------------------------------------------------------------------------
_CLAUDE_KEYWORDS = ("claude-2", "claude-3", "claude-opus", "claude-sonnet")
_LLAMA_KEYWORDS  = ("llama", "vicuna", "alpaca", "mistral", "falcon")

def _model_tier(model_name: str) -> float:
    m = (model_name or "").lower()
    if "gpt-4" in m:
        return 20.0
    if any(k in m for k in _CLAUDE_KEYWORDS):
        return 15.0
    if any(k in m for k in _LLAMA_KEYWORDS):
        return 0.5
    return 1.0  # gpt-3.5 and other hosted models

def _latency_estimate_ms(model_tier: float, token_count: int) -> float:
    """Approximate generation latency from tier + tokens (used in LLMCost)."""
    base = {20.0: 10_000, 15.0: 6_000, 1.0: 800, 0.5: 2_000}.get(model_tier, 1_000)
    return float(base + token_count * 4)  # rough ms/token scaling


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ConvEntry:
    prompt:      str
    response:    str
    n_tokens:    int    # tiktoken count of response
    model_name:  str
    model_tier:  float
    llm_cost:    float     # LLMCost.cost scalar
    llm_cost_obj: LLMCost  # full cost object, reused on cache miss (CA only)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def _count_tokens(text: str, enc) -> int:
    return len(enc.encode(text, disallowed_special=()))


def _stream_entries(dataset, split, msgs_key, model_fn, tier_fn, n) -> List[ConvEntry]:
    """Stream first `n` valid first-turn (user, assistant) exchanges.

    `msgs_key` is the row field holding the message list ("conversation" for
    LMSYS, "messages" for UltraChat); `model_fn(row)` names the model and
    `tier_fn(r_tok, model)` assigns the pricing tier.
    """
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("cl100k_base")
    ds = load_dataset(dataset, split=split, streaming=True)
    entries: List[ConvEntry] = []
    for row in ds:
        if len(entries) >= n:
            break
        msgs = row.get(msgs_key, [])
        if len(msgs) < 2:
            continue
        if msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
            continue
        prompt   = (msgs[0].get("content") or "").strip()
        response = (msgs[1].get("content") or "").strip()
        if not prompt or not response:
            continue
        p_tok = _count_tokens(prompt, enc)
        r_tok = _count_tokens(response, enc)
        if p_tok > 512 or r_tok < 5:
            continue
        model = model_fn(row)
        tier  = tier_fn(r_tok, model)
        cost_obj = LLMCost(
            generation_latency_ms=_latency_estimate_ms(tier, r_tok),
            token_count=r_tok, model_tier=tier,
        )
        entries.append(ConvEntry(prompt, response, r_tok, model, tier,
                                 cost_obj.cost, cost_obj))
    return entries


def load_lmsys(n: int, seed: int) -> List[ConvEntry]:
    """Stream first `n` valid first-turn exchanges from LMSYS-Chat-1M."""
    return _stream_entries(
        "lmsys/lmsys-chat-1m", "train", "conversation",
        model_fn=lambda row: row.get("model", ""),
        tier_fn=lambda r_tok, model: _model_tier(model),
        n=n,
    )


def load_wildchat(n: int, seed: int) -> List[ConvEntry]:
    """Stream first `n` valid first-turn exchanges from WildChat-1M.

    Same shape as LMSYS (per-row `model` field, `conversation` message list),
    so it reuses _model_tier for cost tiers. Gated: needs HF access to
    allenai/WildChat-1M.
    """
    return _stream_entries(
        "allenai/WildChat-1M", "train", "conversation",
        model_fn=lambda row: row.get("model", ""),
        tier_fn=lambda r_tok, model: _model_tier(model),
        n=n,
    )


def load_ultrachat(n: int, seed: int) -> List[ConvEntry]:
    """Stream first `n` valid first-turn exchanges from UltraChat-200K."""
    # UltraChat has no model field — assign tier by response length.
    def _tier(r_tok, _model):
        if r_tok >= 500:
            return 20.0
        if r_tok >= 200:
            return 3.0
        return 1.0
    return _stream_entries(
        "HuggingFaceH4/ultrachat_200k", "train_sft", "messages",
        model_fn=lambda row: "ultrachat",
        tier_fn=_tier,
        n=n,
    )


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM   = 384

def batch_encode(texts: List[str], batch_size: int = 256) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    vecs = model.encode(texts, batch_size=batch_size,
                        show_progress_bar=True, normalize_embeddings=True)
    return vecs.astype("float32")


# ---------------------------------------------------------------------------
# Virtual clock — lets us exercise CA_W_TINYLFU's EWMA time-decay, which is
# otherwise dormant because the whole replay finishes in a few wall-clock
# seconds (dt~=0 => decay~=1.0 on every access). With a virtual clock we
# advance time by a fixed interval per query, so a once-hot item that is not
# re-accessed organically loses frequency over the run.
# ---------------------------------------------------------------------------
class VirtualClock:
    def __init__(self, step_seconds: float):
        self._now = 0.0
        self._step = step_seconds

    def now(self) -> float:
        return self._now

    def tick(self) -> None:
        self._now += self._step


# ---------------------------------------------------------------------------
# Popularity-drift query stream
# ---------------------------------------------------------------------------
# The default replay touches each entry exactly once, so "frequency" is purely
# a function of semantic near-duplicates in the corpus and popularity never
# drifts. That is LFU's best case and gives decay nothing to do.
#
# This generator builds a Zipf-skewed access stream over the entry pool whose
# *hot set rotates over time*: every `rotate` queries the rank->item mapping is
# cyclically shifted, so formerly-hot items cool off and new ones heat up. This
# is the regime our policy is designed for — decay forgets the stale hot set,
# and cost-awareness preferentially retains expensive items within the hot set.
def drift_query_stream(
    n_items: int, n_queries: int, alpha: float, rotate: int, seed: int,
    shift_frac: float = 0.10,
) -> List[int]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_items)          # rank position -> item index
    # Fraction of the pool that rotates per cycle. Small values (e.g. 0.02)
    # keep the hot set stable for many cycles, so frequency still accumulates
    # and the workload is no longer pure-recency (LRU's sweet spot). Large
    # values churn the hot set fast and reward recency.
    shift = max(1, int(n_items * shift_frac))
    samples = rng.zipf(alpha, size=n_queries * 2)
    stream: List[int] = []
    si = 0
    for q in range(n_queries):
        if rotate > 0 and q > 0 and q % rotate == 0:
            order = np.roll(order, shift)     # drift the hot set
        if si >= len(samples):                # refill if Zipf overshot
            samples = rng.zipf(alpha, size=n_queries)
            si = 0
        rank = (samples[si] - 1) % n_items
        si += 1
        stream.append(int(order[rank]))
    return stream


# ---------------------------------------------------------------------------
# Single-policy simulation
# ---------------------------------------------------------------------------
def run_one(
    policy:     str,
    cache_size: int,
    entries:    List[ConvEntry],
    embeddings: np.ndarray,
    sim_threshold: float,  # GPTCache similarity_threshold (0–1)
    clean_size: int,
    eviction_params: Optional[dict] = None,
    policy_label:    Optional[str] = None,
    virtual_clock_sec: float = 0.0,
    query_order:     Optional[List[int]] = None,
    track_mem:       bool = False,
) -> dict:
    """
    Replay a query stream against a fresh SSDataManager + eviction policy.
    Returns per-run metrics dict.

    `policy` is the actual eviction policy string passed to GPTCache;
    `policy_label` is the name reported in the output (lets us run the same
    CA_W_TINYLFU policy under several configs, e.g. a frequency-only variant).
    `eviction_params` are forwarded to the policy constructor (CA only).
    `query_order` is an optional list of indices into `entries` defining the
    replay stream. Default (None) replays each entry once in order; a drift
    stream repeats indices so re-access — and thus eviction — actually happens.
    """
    sq_thr = sq_dist_threshold(sim_threshold)
    is_ca  = policy == "CA_W_TINYLFU"
    label  = policy_label or policy

    order = query_order if query_order is not None else list(range(len(entries)))

    # Build the per-run eviction params (copy so we can inject a fresh clock).
    evp = dict(eviction_params) if eviction_params else {}
    clock = None
    if is_ca and virtual_clock_sec > 0:
        clock = VirtualClock(virtual_clock_sec)
        evp["time_fn"] = clock.now

    hits = 0
    tok_saved = 0
    tok_total = sum(entries[qi].n_tokens for qi in order)
    cost_saved = 0.0
    cost_total = sum(entries[qi].llm_cost for qi in order)
    latencies_ms: List[float] = []

    if track_mem:
        tracemalloc.start()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path  = os.path.join(tmpdir, "cache.db")
        idx_path = os.path.join(tmpdir, "cache.index")

        data_manager = get_data_manager(
            CacheBase("sqlite", sql_url=f"sqlite:///{db_path}"),
            VectorBase("faiss", dimension=EMBED_DIM,
                       index_file_path=idx_path, top_k=10),
            max_size=cache_size,
            clean_size=clean_size,
            eviction=policy,
            eviction_params=evp or None,
        )

        for qi in order:
            entry = entries[qi]
            if clock is not None:
                clock.tick()
            t0  = time.perf_counter()
            emb = embeddings[qi]

            # search returns [(sq_l2_distance, id), ...]
            results = data_manager.search(emb)

            hit_found = False
            for sq_dist, cache_id in (results or []):
                if sq_dist > sq_thr:
                    break  # FAISS returns sorted by distance
                scalar = data_manager.get_scalar_data((sq_dist, cache_id))
                if scalar is None:
                    continue  # soft-deleted
                # confirmed hit
                data_manager.hit_cache_callback((sq_dist, cache_id))
                hit_found = True
                break

            elapsed_ms = (time.perf_counter() - t0) * 1_000
            latencies_ms.append(elapsed_ms)

            if hit_found:
                hits += 1
                tok_saved  += entry.n_tokens
                cost_saved += entry.llm_cost
            else:
                data_manager.save(entry.prompt, entry.response, emb,
                                  **({"llm_cost": entry.llm_cost_obj} if is_ca else {}))

        data_manager.close()

    if track_mem:
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    else:
        peak_mem = 0

    n = len(order)
    latencies_ms.sort()

    def pct(p):
        idx = int(len(latencies_ms) * p / 100)
        return latencies_ms[min(idx, len(latencies_ms) - 1)]

    return {
        "policy":                  label,
        "eviction_policy":         policy,
        "eviction_params":         {k: v for k, v in evp.items() if k != "time_fn"},
        "cache_size":              cache_size,
        "similarity_threshold":    sim_threshold,
        "n_queries":               n,
        "hits":                    hits,
        "hit_rate":                hits / max(n, 1),
        "token_saving_ratio":      tok_saved  / max(tok_total, 1),
        "cost_weighted_hit_rate":  cost_saved / max(cost_total, 1e-9),
        "latency_p50_ms":          pct(50),
        "latency_p95_ms":          pct(95),
        "latency_p99_ms":          pct(99),
        "latency_mean_ms":         statistics.mean(latencies_ms),
        "peak_memory_mb":          peak_mem / 1_048_576,
        "throughput_qps":          n / max(sum(latencies_ms) / 1_000, 1e-9),
    }


def _avg_results(runs: List[dict]) -> dict:
    """Average repeated runs; keep scalar fields from first run."""
    if len(runs) == 1:
        return runs[0]
    scalar_keys = {"policy", "eviction_policy", "eviction_params",
                   "cache_size", "similarity_threshold",
                   "n_queries", "hits"}
    out = dict(runs[0])
    float_keys = [k for k in runs[0] if k not in scalar_keys]
    for k in float_keys:
        vals = [r[k] for r in runs]
        out[k]         = statistics.mean(vals)
        out[k + "_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Full-stack LLM conversation benchmark")
    p.add_argument("--dataset",      default="ultrachat",
                   choices=["ultrachat", "lmsys", "wildchat"],
                   help="Dataset to use (default: ultrachat; lmsys/wildchat require HF gated access)")
    p.add_argument("--n-queries",    type=int, default=3000,
                   help="Number of conversation entries to load (default: 3000)")
    p.add_argument("--cache-sizes",  default="50,100,200",
                   help="Comma-separated cache sizes (default: 50,100,200)")
    p.add_argument("--policies",     default="LRU,LFU,FIFO,RR,WTINYLFU_FREQ,CA_W_TINYLFU",
                   help="Comma-separated eviction policies. WTINYLFU_FREQ is the "
                        "frequency-only (prior-art) baseline: CA_W_TINYLFU with "
                        "cost_aware=False.")
    p.add_argument("--threshold",    type=float, default=0.80,
                   help="GPTCache similarity_threshold (default: 0.80)")
    p.add_argument("--window-ratio", type=float, default=0.01,
                   help="CA_W_TINYLFU window segment ratio (default: 0.01)")
    p.add_argument("--freq-weight",  type=float, default=16.0,
                   help="CA_W_TINYLFU frequency weight in the admission score "
                        "(default: 16.0 = lexicographic; lower blends in cost)")
    p.add_argument("--sweep",        action="store_true",
                   help="Sweep window_ratio x freq_weight for CA_W_TINYLFU "
                        "(ignores --window-ratio/--freq-weight for that policy)")
    p.add_argument("--cost-priority-sweep", default=None,
                   help="Comma-separated cost_priority values in [0,1] to sweep "
                        "for CA_W_TINYLFU (e.g. '0,0.25,0.5,0.75,1'). 0=maximize "
                        "hit rate, 1=maximize money saved. Plots the tradeoff "
                        "curve; takes precedence over --sweep for that policy.")
    p.add_argument("--adaptive-window", action="store_true",
                   help="Also run a CA_W_TINYLFU_ADAPT variant whose window<->main "
                        "boundary self-tunes via a Caffeine-style hill-climb on the "
                        "cost-weighted objective (no per-workload window tuning)")
    p.add_argument("--virtual-clock-sec", type=float, default=0.0,
                   help="Advance a virtual clock by N seconds per query so "
                        "CA_W_TINYLFU's EWMA time-decay is exercised (default: 0 = off)")
    p.add_argument("--drift", action="store_true",
                   help="Replay a Zipf-skewed stream with a rotating hot set over "
                        "the entry pool (exercises re-access, eviction, and decay) "
                        "instead of touching each entry once")
    p.add_argument("--drift-queries", type=int, default=30000,
                   help="Length of the drift query stream (default: 30000)")
    p.add_argument("--drift-zipf", type=float, default=1.2,
                   help="Zipf skew for the drift stream, must be > 1 (default: 1.2)")
    p.add_argument("--drift-rotate", type=int, default=3000,
                   help="Queries between hot-set rotations; 0 disables drift "
                        "(static Zipf) (default: 3000)")
    p.add_argument("--drift-shift", type=float, default=0.10,
                   help="Fraction of the pool that rotates per cycle (default: "
                        "0.10). Lower (e.g. 0.02) = gentler drift: hot set stays "
                        "stable longer so frequency/cost matter, not just recency.")
    p.add_argument("--repeats",      type=int, default=3,
                   help="Number of repeats per config (default: 3)")
    p.add_argument("--clean-pct",    type=float, default=0.20,
                   help="Clean size as fraction of cache size (default: 0.20)")
    p.add_argument("--track-mem",    action="store_true",
                   help="measure peak memory via tracemalloc (~3x slower; "
                        "off by default — peak_memory_mb reports 0 when off)")
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--workdir",      default="bench_lmsys")
    p.add_argument("--out",          default=None,
                   help="Output JSON path (default: <workdir>/results.json)")
    args = p.parse_args()

    cache_sizes = [int(x) for x in args.cache_sizes.split(",") if x.strip()]
    policies    = [x.strip() for x in args.policies.split(",") if x.strip()]

    # ---- Build run specs: (label, eviction_policy, eviction_params) ----
    # Standard cachetools policies take no extra params (eviction_params=None).
    # CA_W_TINYLFU and the frequency-only baseline share the CostAwareWTinyLFU
    # implementation and differ only via eviction_params.
    SWEEP_WINDOW = [0.01, 0.05, 0.10, 0.20]
    SWEEP_FREQW  = [16.0, 4.0, 1.0]
    specs: List[Tuple[str, str, Optional[dict]]] = []
    for pol in policies:
        if pol == "CA_W_TINYLFU":
            if args.cost_priority_sweep:
                for cp in [float(x) for x in args.cost_priority_sweep.split(",") if x.strip()]:
                    specs.append((
                        f"CA_cp{cp:g}", "CA_W_TINYLFU",
                        {"window_ratio": args.window_ratio,
                         "cost_priority": cp, "cost_aware": True},
                    ))
            elif args.sweep:
                for wr in SWEEP_WINDOW:
                    for fw in SWEEP_FREQW:
                        specs.append((
                            f"CA_w{wr:g}_f{fw:g}", "CA_W_TINYLFU",
                            {"window_ratio": wr, "freq_weight": fw, "cost_aware": True},
                        ))
            else:
                specs.append((
                    "CA_W_TINYLFU", "CA_W_TINYLFU",
                    {"window_ratio": args.window_ratio,
                     "freq_weight": args.freq_weight, "cost_aware": True},
                ))
            if args.adaptive_window:
                # Same policy with the window<->main boundary self-tuned online.
                specs.append((
                    "CA_W_TINYLFU_ADAPT", "CA_W_TINYLFU",
                    {"window_ratio": args.window_ratio,
                     "freq_weight": args.freq_weight, "cost_aware": True,
                     "adaptive_window": True},
                ))
        elif pol == "WTINYLFU_FREQ":
            specs.append((
                "WTINYLFU_FREQ", "CA_W_TINYLFU",
                {"window_ratio": args.window_ratio,
                 "freq_weight": args.freq_weight, "cost_aware": False},
            ))
        else:
            specs.append((pol, pol, None))

    print("=" * 72)
    print("Full-stack LLM conversation benchmark")
    print("=" * 72)
    print(f"  Dataset          : {args.dataset}")
    print(f"  Queries          : {args.n_queries}")
    print(f"  Cache sizes      : {cache_sizes}")
    print(f"  Policies         : {[s[0] for s in specs]}")
    print(f"  Sim threshold    : {args.threshold}")
    if args.virtual_clock_sec > 0:
        print(f"  Virtual clock    : +{args.virtual_clock_sec}s per query (decay active)")
    if args.drift:
        mode = (f"rotate {100*args.drift_shift:g}% every {args.drift_rotate}q"
                if args.drift_rotate > 0 else "static Zipf")
        print(f"  Drift stream     : {args.drift_queries} queries, "
              f"Zipf a={args.drift_zipf}, {mode}")
    print(f"  Repeats          : {args.repeats}")
    print(f"  Embedding model  : {EMBED_MODEL} ({EMBED_DIM}d)")

    # ---- Load data ----
    print(f"\nLoading {args.dataset} data ...")
    loader = {"lmsys": load_lmsys, "wildchat": load_wildchat}.get(
        args.dataset, load_ultrachat)
    entries = loader(args.n_queries, args.seed)
    print(f"  Loaded {len(entries)} entries")

    if not entries:
        raise SystemExit("No entries loaded — check dataset access or --n-queries.")

    # ---- Compute embeddings (once) ----
    print(f"\nEncoding {len(entries)} prompts with {EMBED_MODEL} ...")
    embeddings = batch_encode([e.prompt for e in entries])
    print(f"  Embeddings shape: {embeddings.shape}")

    # ---- Workload stats ----
    costs = sorted(e.llm_cost for e in entries)
    n_expensive = sum(1 for e in entries if e.model_tier >= 15)
    print(f"\n  Workload stats:")
    print(f"    cost min/med/p99/max : "
          f"{costs[0]:.0f} / {costs[len(costs)//2]:.0f} / "
          f"{costs[int(len(costs)*0.99)]:.0f} / {costs[-1]:.0f}")
    print(f"    expensive (tier>=15) : {n_expensive} / {len(entries)} "
          f"({100*n_expensive/len(entries):.1f}%)")
    tiers = sorted(set(e.model_tier for e in entries))
    print(f"    model tiers seen     : {tiers}")

    # ---- Build query stream (drift mode) ----
    query_order = None
    if args.drift:
        query_order = drift_query_stream(
            n_items=len(entries), n_queries=args.drift_queries,
            alpha=args.drift_zipf, rotate=args.drift_rotate, seed=args.seed,
            shift_frac=args.drift_shift,
        )
        from collections import Counter
        c = Counter(query_order)
        print(f"    drift stream         : {len(query_order)} queries, "
              f"{len(c)}/{len(entries)} items touched, "
              f"top-10 share {100*sum(n for _, n in c.most_common(10))/len(query_order):.1f}%")
    print()

    # ---- Run benchmark ----
    all_results = []
    header = (f"  {'policy':<15} {'cs':>4}  "
              f"{'hit%':>7} {'tok_save%':>10} {'cost_wt%':>10} "
              f"{'p50ms':>7} {'p95ms':>7} {'mem_mb':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cache_size in cache_sizes:
        clean_size = max(1, int(cache_size * args.clean_pct))
        for label, policy, evp in specs:
            runs = []
            for rep in range(args.repeats):
                try:
                    r = run_one(policy, cache_size, entries, embeddings,
                                args.threshold, clean_size,
                                eviction_params=evp, policy_label=label,
                                virtual_clock_sec=args.virtual_clock_sec,
                                query_order=query_order,
                                track_mem=args.track_mem)
                    runs.append(r)
                except Exception as exc:
                    print(f"  [skip] {label} cs={cache_size} rep={rep}: {exc}")
                    break
            if not runs:
                continue
            avg = _avg_results(runs)
            avg["repeats"] = len(runs)
            all_results.append(avg)
            print(f"  {label:<15} {cache_size:>4}  "
                  f"{avg['hit_rate']*100:>6.2f}% "
                  f"{avg['token_saving_ratio']*100:>9.2f}% "
                  f"{avg['cost_weighted_hit_rate']*100:>9.2f}% "
                  f"{avg['latency_p50_ms']:>7.1f} "
                  f"{avg['latency_p95_ms']:>7.1f} "
                  f"{avg['peak_memory_mb']:>7.1f}")

    # ---- Save ----
    out_path = args.out or os.path.join(args.workdir, "results.json")
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "args":     vars(args),
            "dataset":  args.dataset,
            "n_loaded": len(entries),
            "embed_model": EMBED_MODEL,
            "embed_dim":   EMBED_DIM,
            "results":  all_results,
        }, f, indent=2)
    print(f"\n  Results -> {out_path}")


if __name__ == "__main__":
    main()
