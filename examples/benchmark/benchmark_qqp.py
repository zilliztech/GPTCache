"""GPTCache QQP Benchmark (4-cell isolation matrix)

Benchmark points (BP1-BP6):

  BP1 - 4-cell encoder x index matrix:
        A: ONNX 768d + Flat        (true baseline - what users have today)
        B: ONNX 768d + HNSW+SQ8    (isolates the index change)
        C: MRL  256d + Flat        (isolates the encoder change)
        D: MRL  256d + HNSW+SQ8    (combined - current optimised config)
  BP2 - Pure search latency reported separately from end-to-end.
  BP3 - Index-only RAM via faiss.serialize_index; on-disk via os.path.getsize.
        These are reported as two distinct columns - different problems.
  BP4 - Configurable scale (--scale 10000 / 100000 / 1000000).
  BP5 - Warmup, repeats (median + IQR), explicit thread pinning,
        p50/p90/p95/p99 distributions.
  BP6 - Old OpenAI-driven scripts moved to examples/smoke/ (see that folder).

This harness is encoder-agnostic. If a real encoder (ONNX, sentence-transformers)
is not installed, that cell is SKIPPED with a clear message; the other cells
continue. A `--encoder synthetic` mode is also provided for fast harness
verification without any model downloads.

Usage
-----
  # Quick sanity run, synthetic encoder, 10K vectors:
  python benchmark_qqp.py --scale 10000 --encoder synthetic --repeats 1

  # Real run (requires sentence-transformers, transformers, onnxruntime,
  # datasets):
  python benchmark_qqp.py --scale 10000 --encoder auto --repeats 3
  python benchmark_qqp.py --scale 100000 --encoder auto --repeats 3
  python benchmark_qqp.py --scale 1000000 --encoder auto --repeats 3

Output
------
The harness prints a results table for the four cells and writes a JSON
artifact next to it (one row per cell, including memory + percentiles)
that downstream tooling (or docs/steps-1-5-results.md) can diff against.
"""

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager

import numpy as np

# ---------------------------------------------------------------------------
# Thread pinning - MUST happen before numpy/torch heavy imports for OMP_NUM_THREADS
# to take effect. We re-pin via faiss.omp_set_num_threads() later as well.
# ---------------------------------------------------------------------------
def _set_thread_env_early(n):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(n))


_DEFAULT_THREADS = int(os.environ.get("GPTCACHE_FAISS_THREADS", "1"))
_set_thread_env_early(_DEFAULT_THREADS)

import faiss  # noqa: E402  - import after env pin

from gptcache import cache, Config  # noqa: E402
from gptcache.manager import CacheBase, VectorBase, get_data_manager  # noqa: E402
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation  # noqa: E402


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class _BaseEncoder:
    dimension = 0
    label = "?"

    def to_embeddings(self, data, **_):
        raise NotImplementedError


class SyntheticEncoder(_BaseEncoder):
    """Deterministic hash-based pseudo-embedding for harness verification.

    Same string -> same vector. To make the harness produce realistic
    non-zero TP / FP rates without any model download, the encoder
    recognises the synthetic dataset's naming convention:

      "topic_<N>"               -> seed N (base vector)
      "topic_<N>_paraphrase"    -> seed N + 0.05 * noise (TP variant - near)
      "<anything else>"         -> full-text hash (FP variant - far)

    For real encoders this lookup path is dead code.
    """

    _PARAPHRASE_SUFFIX = "_paraphrase"

    def __init__(self, dim, paraphrase_noise=0.05):
        self.dimension = dim
        self.label = f"synthetic-{dim}d"
        self._paraphrase_noise = paraphrase_noise

    def _topic_seed(self, text):
        # Try to extract "topic_<N>" prefix; return (seed, is_paraphrase)
        if text.startswith("topic_"):
            stripped = text
            is_para = False
            if stripped.endswith(self._PARAPHRASE_SUFFIX):
                stripped = stripped[: -len(self._PARAPHRASE_SUFFIX)]
                is_para = True
            tail = stripped[len("topic_"):]
            try:
                return int(tail), is_para
            except ValueError:
                return None, False
        return None, False

    def _vec(self, text):
        seed, is_para = self._topic_seed(text)
        if seed is None:
            seed = int.from_bytes(
                hashlib.blake2s(text.encode("utf-8"), digest_size=8).digest(), "little"
            )
            is_para = False
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dimension).astype(np.float32)
        if is_para:
            noise_rng = np.random.default_rng(seed ^ 0x9E3779B97F4A7C15)
            v = v + self._paraphrase_noise * noise_rng.standard_normal(self.dimension).astype(np.float32)
        v /= max(np.linalg.norm(v), 1e-9)
        return v

    def to_embeddings(self, data, **_):
        if isinstance(data, list):
            return np.stack([self._vec(d) for d in data])
        return self._vec(data)


def _try_import_onnx_encoder():
    try:
        from gptcache.embedding import Onnx
        return Onnx
    except ImportError:
        return None


def _try_import_mrl_encoder():
    try:
        from gptcache.embedding import SBERTMRL
        return SBERTMRL
    except ImportError:
        return None


class _SBERT768Encoder(_BaseEncoder):
    """PyTorch SentenceTransformer wrapper producing 768-d embeddings.

    Uses the same underlying model as the ONNX encoder
    (paraphrase-albert-small-v2) but runs in native PyTorch, which
    supports true batching.  This is ~30x faster than the static-batch-1
    ONNX model for ingest, at the cost of not using ONNX Runtime.
    """

    def __init__(self):
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        self._model = SentenceTransformer("paraphrase-albert-small-v2")
        self._model.eval()
        self.dimension = self._model.get_sentence_embedding_dimension()
        self.label = f"sbert768-{self.dimension}d"

    def to_embeddings(self, data, **_):
        import numpy as np  # noqa: PLC0415
        if isinstance(data, str):
            data = [data]
        emb = self._model.encode(data, batch_size=64, show_progress_bar=False)
        result = np.array(emb).astype("float32")
        return result.squeeze(0) if result.shape[0] == 1 else result


def make_encoder(kind, *, dim=None, onnx_fallback=True, model=None):
    """Build the encoder for a cell. Returns None if its deps aren't available.

    kind in {"onnx", "mrl", "synthetic"}. `model` overrides the MRL model name
    (e.g. static-retrieval-mrl-en-v1 for a zero-transformer, ~0ms encoder).
    onnx_fallback: if True, fall back to the PyTorch SBERT-768 encoder when
                   the ONNX model is unavailable (e.g. no dynamic-batch export).
    """
    if kind == "synthetic":
        assert dim is not None
        return SyntheticEncoder(dim)
    if kind == "onnx":
        Onnx = _try_import_onnx_encoder()
        if Onnx is None:
            return None
        try:
            enc = Onnx()
            # Check if the loaded ONNX model supports dynamic batching.
            # The original GPTCache model was exported with static batch_size=1,
            # making it ~30-100x slower than a batched PyTorch encoder.
            # Fall back to SBERT-768 unless a dynamic-batch model was provided.
            if onnx_fallback and not getattr(enc, "_dynamic_batch", True):
                print("  [fallback] ONNX model has static batch_size=1 — "
                      "falling back to PyTorch SentenceTransformer 768d (true batching).")
                print("  [tip] Run scripts/export_onnx_dynamic.py and set "
                      "GPTCACHE_ONNX_MODEL_DIR to use a fast dynamic-batch ONNX model.")
                try:
                    return _SBERT768Encoder()
                except Exception as fe:  # noqa: BLE001
                    print(f"  [skip] sbert768 fallback also failed: {fe}")
                    return None
            enc.label = f"onnx-{enc.dimension}d"
            return enc
        except Exception as e:  # noqa: BLE001
            if onnx_fallback:
                print(f"  [fallback] ONNX unavailable ({e})")
                print("  [fallback] Using PyTorch SentenceTransformer 768d instead (supports batching).")
                try:
                    return _SBERT768Encoder()
                except Exception as fe:  # noqa: BLE001
                    print(f"  [skip] sbert768 fallback also failed: {fe}")
                    return None
            print(f"  [skip] onnx encoder unavailable: {e}")
            return None
    if kind == "mrl":
        SBERTMRL = _try_import_mrl_encoder()
        if SBERTMRL is None:
            return None
        try:
            enc = (SBERTMRL(model, target_dim=dim or 256) if model
                   else SBERTMRL(target_dim=dim or 256))
            enc.label = f"mrl-{enc.dimension}d"
            return enc
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] MRL encoder unavailable: {e}")
            return None
    raise ValueError(f"unknown encoder kind: {kind}")


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

def _try_load_qqp(n_ingest, n_tp, n_fp):
    """Return (db_questions, tp_queries, fp_queries) or None if datasets is missing."""
    try:
        from datasets import load_dataset
    except ImportError:
        return None
    ds = load_dataset("glue", "qqp", split="train")
    dup = ds.filter(lambda x: x["label"] == 1)
    non = ds.filter(lambda x: x["label"] == 0)
    # If the caller asks for more than the dataset can supply, fall back to
    # what's available rather than crashing.
    n_ingest = min(n_ingest, len(dup))
    n_tp = min(n_tp, n_ingest)
    n_fp = min(n_fp, len(non))
    dup_pairs = list(dup.select(range(n_ingest)))
    non_pairs = list(non.select(range(n_fp)))
    db_questions = [p["question1"] for p in dup_pairs]
    tp_queries = [p["question2"] for p in dup_pairs[:n_tp]]
    fp_queries = [p["question2"] for p in non_pairs]
    return db_questions, tp_queries, fp_queries


def _synthetic_dataset(n_ingest, n_tp, n_fp, seed=0):
    """Deterministic question-pair generator for harness sanity runs.

    Uses the "topic_<N>" naming convention that SyntheticEncoder
    recognises, so TP queries land near their ingested originals in
    vector space and FP queries land far away.
    """
    db_questions = [f"topic_{i}" for i in range(n_ingest)]
    tp_queries = [f"topic_{i}{SyntheticEncoder._PARAPHRASE_SUFFIX}"
                  for i in range(min(n_tp, n_ingest))]
    # Use seeds well outside the ingested range so FP vectors are uncorrelated.
    fp_queries = [f"topic_{seed + n_ingest + 10_000_000 + i}" for i in range(n_fp)]
    return db_questions, tp_queries, fp_queries


def load_dataset_or_synthesise(prefer_qqp, n_ingest, n_tp, n_fp):
    if prefer_qqp:
        loaded = _try_load_qqp(n_ingest, n_tp, n_fp)
        if loaded is not None:
            return loaded, "qqp"
        print("  [data] HuggingFace `datasets` not installed - falling back to synthetic pairs")
    return _synthetic_dataset(n_ingest, n_tp, n_fp), "synthetic"


# ---------------------------------------------------------------------------
# Cache setup per cell
# ---------------------------------------------------------------------------

def setup_cell(encoder, index_kind, work_dir, similarity_threshold, max_size,
               hnsw_m=32, m_pq=32, k_factor=4):
    """Initialise a fresh GPTCache for a cell. Returns (data_manager, faiss_path, sqlite_path).

    ``index_kind`` selects the FAISS index: ``flat``, ``hnsw_sq8``, ``hnsw_pq``
    (Product Quantization) or ``hnsw_pq_refine`` (PQ + full-precision re-rank).
    ``hnsw_m``/``m_pq``/``k_factor`` tune the graph degree, PQ code length and
    refine over-fetch respectively.
    """
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)
    sqlite_path = os.path.join(work_dir, "sqlite.db")
    faiss_path = os.path.join(work_dir, "faiss.index")

    cache_base = CacheBase("sqlite", sql_url=f"sqlite:///{sqlite_path}")
    if index_kind == "flat":
        vector_base = VectorBase("faiss", dimension=encoder.dimension, index_path=faiss_path)
    elif index_kind == "hnsw_sq8":
        vector_base = VectorBase(
            "faiss", dimension=encoder.dimension, index_path=faiss_path,
            index_type="hnsw_sq8", hnsw_m=hnsw_m,
        )
    elif index_kind in ("hnsw_pq", "hnsw_pq_refine"):
        vector_base = VectorBase(
            "faiss", dimension=encoder.dimension, index_path=faiss_path,
            index_type=index_kind, hnsw_m=hnsw_m, m_pq=m_pq, k_factor=k_factor,
        )
    else:
        raise ValueError(f"unknown index_kind: {index_kind}")

    data_manager = get_data_manager(cache_base, vector_base, max_size=max_size)
    cache.init(
        embedding_func=encoder.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
        config=Config(similarity_threshold=similarity_threshold),
    )
    return data_manager, faiss_path, sqlite_path


def encode_corpus(encoder, questions, batch_size):
    """Encode the whole corpus once, batched. Returns an (n, dim) float32 array."""
    embs = []
    for start in range(0, len(questions), batch_size):
        e = encoder.to_embeddings(questions[start:start + batch_size])
        embs.append(np.asarray(e, dtype=np.float32))
    return np.vstack(embs)


def ingest(encoder, db_questions, data_manager, batch_size=64, precomputed=None):
    """Time the ingest path. Returns total seconds.

    If ``precomputed`` (an (n, dim) corpus-embedding array) is given, insert it
    directly via data_manager.import_data instead of re-encoding. This lets
    cells that share an encoder (all MRL cells) skip redundant encode passes.
    import_data applies the same normalize() as cache.import_data, so stored
    vectors are identical. Query-time encoding in measure_cell is untouched, so
    latency/recall stay honest.
    """
    dummy_answers = [f"a_{i}" for i in range(len(db_questions))]
    t0 = time.perf_counter()
    if precomputed is None:
        for start in range(0, len(db_questions), batch_size):
            bq = db_questions[start:start + batch_size]
            ba = dummy_answers[start:start + batch_size]
            cache.import_data(questions=bq, answers=ba, batch_size=batch_size)
    else:
        for start in range(0, len(db_questions), batch_size):
            sl = slice(start, start + batch_size)
            bq = db_questions[sl]
            data_manager.import_data(
                questions=bq,
                answers=dummy_answers[sl],
                embedding_datas=list(precomputed[sl]),
                session_ids=[None] * len(bq),
            )
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _sweep_thresholds(tp_scores, fp_scores, min_r, max_r, thresholds):
    """Recompute TP/FP/precision at each candidate threshold from cached query
    scores. The index is searched once; only the accept cutoff moves, so this is
    free - no re-ingest, no re-search. Answers "is MRL's high FP just a
    mis-tuned global threshold?" by giving a precision/recall curve per cell."""
    tp = np.asarray(tp_scores, dtype=np.float64)
    fp = np.asarray(fp_scores, dtype=np.float64)
    rows = []
    for t in thresholds:
        cut = (max_r - min_r) * t
        tp_hits = int((tp >= cut).sum())
        fp_hits = int((fp >= cut).sum())
        denom = tp_hits + fp_hits
        rows.append({
            "threshold": t,
            "tp_hit_rate": tp_hits / max(tp.size, 1),
            "fp_hit_rate": fp_hits / max(fp.size, 1),
            "precision": (tp_hits / denom) if denom else 0.0,
        })
    return rows


def measure_cell(encoder, data_manager, tp_queries, fp_queries,
                 similarity_threshold, warmup, repeats, top_k=1,
                 threshold_sweep=None):
    """Run TP + FP query sets with warmup and repeats.

    Returns dict with:
      - tp_hit_rate, fp_hit_rate
      - search_latency_ms: dict with p50, p90, p95, p99 (pooled across repeats)
      - e2e_latency_ms:    dict with p50, p90, p95, p99
      - per_repeat: list of dicts with the same percentile keys
    """
    evaluator = cache.similarity_evaluation
    min_r, max_r = evaluator.range()
    rank_threshold = (max_r - min_r) * similarity_threshold

    all_queries = list(tp_queries) + list(fp_queries)
    is_tp = [True] * len(tp_queries) + [False] * len(fp_queries)

    # --- Warmup (results discarded) ---
    for q in all_queries[:warmup]:
        e = encoder.to_embeddings(q)
        _ = data_manager.search(e)

    pooled_search = []
    pooled_e2e = []
    per_repeat = []
    tp_scores = []  # raw match scores, captured once (rep 0) for the threshold sweep
    fp_scores = []

    for rep in range(repeats):
        search_ms = []
        e2e_ms = []
        tp_hits = 0
        fp_hits = 0
        for q, tp in zip(all_queries, is_tp):
            # End-to-end timer: embed + search + evaluate
            t0 = time.perf_counter()
            emb = encoder.to_embeddings(q)
            # BP2 - pure search measurement: time search() in isolation
            ts0 = time.perf_counter()
            res = data_manager.search(emb)
            ts1 = time.perf_counter()
            search_ms.append((ts1 - ts0) * 1000.0)

            hit = False
            score = None
            if res:
                distance, cid = res[0]
                score = evaluator.evaluation({}, {"search_result": (distance, cid)})
                if score >= rank_threshold:
                    hit = True
            t1 = time.perf_counter()
            e2e_ms.append((t1 - t0) * 1000.0)
            if tp and hit:
                tp_hits += 1
            elif (not tp) and hit:
                fp_hits += 1
            if rep == 0:
                # No result => never a hit at any threshold.
                s = score if score is not None else float("-inf")
                (tp_scores if tp else fp_scores).append(s)

        pooled_search.extend(search_ms)
        pooled_e2e.extend(e2e_ms)
        per_repeat.append({
            "search_ms": _percentiles(search_ms),
            "e2e_ms": _percentiles(e2e_ms),
            "tp_hits": tp_hits,
            "fp_hits": fp_hits,
        })

    sweep = None
    if threshold_sweep:
        sweep = _sweep_thresholds(tp_scores, fp_scores, min_r, max_r, threshold_sweep)

    return {
        "tp_hit_rate": per_repeat[-1]["tp_hits"] / max(len(tp_queries), 1),
        "fp_hit_rate": per_repeat[-1]["fp_hits"] / max(len(fp_queries), 1),
        "search_latency_ms": _percentiles(pooled_search),
        "e2e_latency_ms": _percentiles(pooled_e2e),
        "per_repeat": per_repeat,
        "threshold_sweep": sweep,
    }


def _percentiles(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0,
                "iqr": 0.0, "n": 0}
    p25, p50, p75, p90, p95, p99 = np.percentile(arr, [25, 50, 75, 90, 95, 99])
    return {
        "p50": float(p50),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
        "mean": float(arr.mean()),
        "iqr": float(p75 - p25),
        "n": int(arr.size),
    }


def measure_exact_match_shortcut(encoder, data_manager, repeat_queries, warmup, repeats):
    """Compare time-to-answer for an exact-repeat query through:
       (a) embed + search (the path without the shortcut)
       (b) exact_match_cache.get (the exact-match shortcut)

    The exact-match cache is populated up front with the same queries so
    every lookup is a guaranteed hit.

    Returns dict with shortcut_ms (shortcut path) and baseline_ms (without).
    """
    from gptcache import cache as global_cache  # local import to avoid name collision
    emc = getattr(global_cache, "exact_match_cache", None)
    if emc is None:
        return None
    # Populate the shortcut cache
    emc.clear()
    for q in repeat_queries:
        emc.put(q, f"cached_answer_for::{q}")

    # Warmup
    for q in repeat_queries[:warmup]:
        _ = encoder.to_embeddings(q)
        _ = data_manager.search(_)
        _ = emc.get(q)

    baseline_ms = []
    shortcut_ms = []
    for _ in range(repeats):
        for q in repeat_queries:
            t0 = time.perf_counter()
            e = encoder.to_embeddings(q)
            _ = data_manager.search(e)
            t1 = time.perf_counter()
            baseline_ms.append((t1 - t0) * 1000.0)

            t0 = time.perf_counter()
            _ = emc.get(q)
            t1 = time.perf_counter()
            shortcut_ms.append((t1 - t0) * 1000.0)

    return {
        "baseline_ms": _percentiles(baseline_ms),
        "shortcut_ms": _percentiles(shortcut_ms),
        "exact_match_hits": emc.hits,
        "exact_match_misses": emc.misses,
    }


def measure_memory(data_manager, faiss_path, sqlite_path):
    """BP3 - index-only RAM via faiss.serialize_index + on-disk sizes."""
    # The underlying vector store inside SSDataManager is exposed as .v
    vector_store = data_manager.v
    faiss_index = getattr(vector_store, "_index", None)
    ram_bytes = 0
    if faiss_index is not None:
        try:
            ram_bytes = int(len(faiss.serialize_index(faiss.downcast_index(faiss_index))))
        except Exception:  # noqa: BLE001
            # downcast_index may raise on IndexIDMap wrappers; serialize the wrapper instead
            try:
                ram_bytes = int(len(faiss.serialize_index(faiss_index)))
            except Exception:  # noqa: BLE001
                ram_bytes = 0
    faiss_disk = os.path.getsize(faiss_path) if os.path.isfile(faiss_path) else 0
    sqlite_disk = os.path.getsize(sqlite_path) if os.path.isfile(sqlite_path) else 0
    return {
        "faiss_ram_bytes": ram_bytes,
        "faiss_disk_bytes": int(faiss_disk),
        "sqlite_disk_bytes": int(sqlite_disk),
    }


# ---------------------------------------------------------------------------
# Cell driver
# ---------------------------------------------------------------------------

CELLS = [
    {"name": "A", "label": "ONNX/768/Flat",     "encoder": "onnx",      "index": "flat"},
    {"name": "B", "label": "ONNX/768/HNSW+SQ8", "encoder": "onnx",      "index": "hnsw_sq8"},
    {"name": "C", "label": "MRL/256/Flat",      "encoder": "mrl",       "index": "flat"},
    {"name": "D", "label": "MRL/256/HNSW+SQ8", "encoder": "mrl",       "index": "hnsw_sq8"},
    # Compression frontier on the production MRL/256 encoder (S1/S2/S3).
    # E/F trade graph quality for code size; G/H sweep the HNSW degree M.
    {"name": "E", "label": "MRL/256/HNSW+PQ",        "encoder": "mrl", "index": "hnsw_pq"},
    {"name": "F", "label": "MRL/256/HNSW+PQ+Refine", "encoder": "mrl", "index": "hnsw_pq_refine"},
    {"name": "G", "label": "MRL/256/HNSW+SQ8 (M=16)", "encoder": "mrl", "index": "hnsw_sq8", "hnsw_m": 16},
    {"name": "H", "label": "MRL/256/HNSW+SQ8 (M=24)", "encoder": "mrl", "index": "hnsw_sq8", "hnsw_m": 24},
    # Isolation cell: nomic at FULL 768d + flat. Separates the precision cost of
    # MRL *truncation* (256d) from the *encoder + threshold* (vs ONNX baseline A).
    {"name": "I", "label": "MRL/768/Flat", "encoder": "mrl", "index": "flat", "mrl_dim": 768},
    # Static encoder: token-embedding lookup, no transformer forward → ~0ms encode,
    # killing the e2e-latency cost that the nomic MRL encoder pays (frontier.md note 5).
    # Same MRL/256/HNSW+PQ index as cell E for an apples-to-apples compression compare.
    {"name": "J", "label": "StaticMRL/256/HNSW+PQ", "encoder": "mrl", "index": "hnsw_pq",
     "mrl_model": "sentence-transformers/static-retrieval-mrl-en-v1"},
]


def run_cell(spec, data, args, threads, encoder_cache):
    name = spec["name"]
    print(f"\n=== Cell {name} - {spec['label']} ===")

    encoder_kind = spec["encoder"]
    mrl_dim = spec.get("mrl_dim", 256)
    # Cache key must include the MRL dim so the 256d and 768d cells don't share
    # an encoder (different output dimension, different index).
    mrl_model = spec.get("mrl_model")
    enc_key = (f"mrl{mrl_dim}:{mrl_model or 'nomic'}"
               if encoder_kind == "mrl" else encoder_kind)
    reuse = not args.no_reuse_embeddings
    cached = encoder_cache.get(enc_key) if reuse else None
    if cached is not None:
        encoder, corpus_emb = cached
    else:
        if args.encoder == "synthetic":
            # Force-synthetic regardless of cell - useful for harness sanity runs.
            # Pick the dim that matches the cell's logical encoder.
            encoder = SyntheticEncoder(768 if encoder_kind == "onnx" else mrl_dim)
        else:
            onnx_fallback = not getattr(args, "no_onnx_fallback", False)
            encoder = make_encoder(encoder_kind, dim=mrl_dim,
                                   onnx_fallback=onnx_fallback, model=mrl_model)
            if encoder is None:
                print(f"  [skip] cell {name}: encoder '{encoder_kind}' unavailable. "
                      f"Install its deps to run this cell.")
                return None
        corpus_emb = None
        if reuse:
            encoder_cache[enc_key] = (encoder, None)

    work_dir = os.path.join(args.workdir, f"cell_{name}")
    # Per-cell overrides (e.g. an M-sweep cell sets its own hnsw_m) fall back
    # to the global CLI defaults.
    data_manager, faiss_path, sqlite_path = setup_cell(
        encoder=encoder,
        index_kind=spec["index"],
        work_dir=work_dir,
        similarity_threshold=args.threshold,
        max_size=max(args.scale * 2, 100_000),
        hnsw_m=spec.get("hnsw_m", args.hnsw_m),
        m_pq=spec.get("m_pq", args.m_pq),
        k_factor=spec.get("k_factor", args.k_factor),
    )

    db_q, tp_q, fp_q = data
    print(f"  Encoder      : {encoder.label}")
    print(f"  Index        : {spec['index']}")
    print(f"  Threads      : {threads}")
    print(f"  Ingest size  : {len(db_q)}")
    print(f"  TP queries   : {len(tp_q)}")
    print(f"  FP queries   : {len(fp_q)}")

    encode_s = 0.0
    precomputed = None
    if reuse:
        if corpus_emb is None:
            te0 = time.perf_counter()
            corpus_emb = encode_corpus(encoder, db_q, args.ingest_batch)
            encode_s = time.perf_counter() - te0
            encoder_cache[enc_key] = (encoder, corpus_emb)
        precomputed = corpus_emb

    ingest_s = ingest(encoder, db_q, data_manager,
                      batch_size=args.ingest_batch, precomputed=precomputed)
    total_ingest_s = encode_s + ingest_s
    print(f"  Ingest time  : {total_ingest_s:.2f}s "
          f"({len(db_q)/max(total_ingest_s,1e-9):.0f} vec/s)"
          + ("" if encode_s or not reuse else "  [reused embeddings]"))

    sweep_list = ([float(x) for x in args.threshold_sweep.split(",")]
                  if getattr(args, "threshold_sweep", None) else None)
    metrics = measure_cell(
        encoder=encoder,
        data_manager=data_manager,
        tp_queries=tp_q,
        fp_queries=fp_q,
        similarity_threshold=args.threshold,
        warmup=args.warmup,
        repeats=args.repeats,
        threshold_sweep=sweep_list,
    )

    # Optional: exercise the exact-match shortcut on a slice of the
    # TP query set. Only meaningful on cell D (the production config),
    # but we measure on any cell - the lever is encoder/index agnostic.
    exact_match_metrics = None
    if args.exact_repeat_frac > 0.0:
        n_repeat = max(int(len(tp_q) * args.exact_repeat_frac), 1)
        repeat_queries = list(tp_q[:n_repeat])
        exact_match_metrics = measure_exact_match_shortcut(
            encoder=encoder,
            data_manager=data_manager,
            repeat_queries=repeat_queries,
            warmup=min(args.warmup, n_repeat),
            repeats=args.repeats,
        )

    # Flush the FAISS index to disk before measuring sizes; otherwise the
    # index file size is 0 until close() runs (auto_flush only triggers
    # every N saves and may not have fired yet).
    data_manager.flush()
    mem = measure_memory(data_manager, faiss_path, sqlite_path)
    data_manager.close()

    print(f"  TP hit rate  : {metrics['tp_hit_rate']*100:.2f}%")
    print(f"  FP hit rate  : {metrics['fp_hit_rate']*100:.2f}%")
    print(f"  Search lat   : p50 {metrics['search_latency_ms']['p50']:.3f} ms"
          f"  p95 {metrics['search_latency_ms']['p95']:.3f} ms"
          f"  p99 {metrics['search_latency_ms']['p99']:.3f} ms"
          f"  IQR {metrics['search_latency_ms']['iqr']:.3f}")
    print(f"  E2E    lat   : p50 {metrics['e2e_latency_ms']['p50']:.3f} ms"
          f"  p95 {metrics['e2e_latency_ms']['p95']:.3f} ms"
          f"  p99 {metrics['e2e_latency_ms']['p99']:.3f} ms")
    print(f"  FAISS RAM    : {mem['faiss_ram_bytes']/1e6:.2f} MB"
          f"  (serialize_index)")
    print(f"  FAISS disk   : {mem['faiss_disk_bytes']/1e6:.2f} MB")
    print(f"  SQLite disk  : {mem['sqlite_disk_bytes']/1e6:.2f} MB")

    if metrics.get("threshold_sweep"):
        print("  Thr sweep    : thr ->  TP%   FP%  prec%")
        for row in metrics["threshold_sweep"]:
            print(f"                 {row['threshold']:.2f} -> "
                  f"{row['tp_hit_rate']*100:5.1f} {row['fp_hit_rate']*100:5.1f} "
                  f"{row['precision']*100:5.1f}")

    if exact_match_metrics is not None:
        bm = exact_match_metrics["baseline_ms"]
        sm = exact_match_metrics["shortcut_ms"]
        print(f"  ExactMatch   : baseline p50 {bm['p50']:.3f} ms"
              f" -> shortcut p50 {sm['p50']:.6f} ms"
              f"  (speedup {bm['p50']/max(sm['p50'],1e-9):.0f}x)"
              f"  hits={exact_match_metrics['exact_match_hits']}")

    return {
        "name": name,
        "label": spec["label"],
        "encoder": encoder.label,
        "index": spec["index"],
        "ingest_seconds": total_ingest_s,
        "metrics": metrics,
        "memory": mem,
        "exact_match": exact_match_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="GPTCache QQP benchmark (4-cell matrix)")
    p.add_argument("--scale", type=int, default=10_000,
                   help="Number of vectors to ingest. Use 10000 / 100000 / 1000000.")
    p.add_argument("--n-tp", type=int, default=None,
                   help="Number of TP queries (default: min(scale/5, 2000))")
    p.add_argument("--n-fp", type=int, default=None,
                   help="Number of FP queries (default: same as TP)")
    p.add_argument("--encoder", choices=["auto", "synthetic"], default="auto",
                   help="auto = real encoders if installed, else skip; "
                        "synthetic = deterministic hash-based (no model downloads)")
    p.add_argument("--data", choices=["auto", "qqp", "synthetic"], default="auto",
                   help="qqp requires HuggingFace `datasets`; synthetic uses generated pairs")
    p.add_argument("--threshold", type=float, default=0.90,
                   help="similarity_threshold (default 0.90)")
    p.add_argument("--threshold-sweep", default=None, dest="threshold_sweep",
                   help="comma-separated thresholds to re-score each cell at, e.g. "
                        "'0.85,0.88,0.90,0.92,0.95'. Reuses the single search pass "
                        "(no re-ingest); emits a precision/recall curve per cell so "
                        "you can compare cells at matched precision, not one global cut.")
    p.add_argument("--ingest-batch", type=int, default=64)
    p.add_argument("--warmup", type=int, default=20,
                   help="queries to discard from latency stats (BP5)")
    p.add_argument("--repeats", type=int, default=3,
                   help="how many times to re-run the query set (BP5)")
    p.add_argument("--threads", type=int, default=_DEFAULT_THREADS,
                   help="faiss.omp_set_num_threads value (default $GPTCACHE_FAISS_THREADS or 1)")
    p.add_argument("--cells", default="A,B,C,D",
                   help="comma-separated cell names to run (E/F=PQ, G/H=M-sweep, "
                        "I=MRL/768/flat truncation-isolation cell)")
    p.add_argument("--hnsw-m", type=int, default=32, dest="hnsw_m",
                   help="HNSW graph degree M for hnsw_* cells (per-cell spec may override)")
    p.add_argument("--m-pq", type=int, default=32, dest="m_pq",
                   help="PQ sub-quantizer count = bytes/code for hnsw_pq[_refine] cells")
    p.add_argument("--k-factor", type=int, default=4, dest="k_factor",
                   help="IndexRefineFlat over-fetch factor for hnsw_pq_refine cells")
    p.add_argument("--no-onnx-fallback", action="store_true", default=False,
                   help="disable PyTorch SBERT-768 fallback for cells A/B when "
                        "the ONNX dynamic-batch model is unavailable")
    p.add_argument("--no-reuse-embeddings", action="store_true", default=False,
                   help="disable sharing ingest embeddings across cells that use "
                        "the same encoder (all MRL cells). Reuse skips redundant "
                        "encode passes; query-time encoding is always real, so "
                        "search latency and recall are unaffected.")
    p.add_argument("--workdir", default="bench_work",
                   help="directory for per-cell sqlite + faiss files")
    p.add_argument("--out", default=None,
                   help="path to write JSON results (default: <workdir>/results.json)")
    p.add_argument("--exact-repeat-frac", type=float, default=0.0,
                   help="Fraction of the TP query workload to clone as exact repeats "
                        "(0.0..1.0). With >0 the harness runs an additional measurement "
                        "with the gptcache.adapter pipeline so the exact-match "
                        "shortcut can be exercised end-to-end. Default 0.0.")
    args = p.parse_args()

    if args.n_tp is None:
        args.n_tp = max(min(args.scale // 5, 2000), 100)
    if args.n_fp is None:
        args.n_fp = args.n_tp

    # BP5 - thread pinning
    faiss.omp_set_num_threads(args.threads)

    # Resolve dataset
    prefer_qqp = args.data in ("auto", "qqp")
    data, data_source = load_dataset_or_synthesise(
        prefer_qqp=prefer_qqp,
        n_ingest=args.scale,
        n_tp=args.n_tp,
        n_fp=args.n_fp,
    )

    print("=" * 64)
    print("GPTCache QQP benchmark - 4-cell matrix")
    print("=" * 64)
    print(f"  Scale        : {args.scale}")
    print(f"  TP / FP      : {args.n_tp} / {args.n_fp}")
    print(f"  Threshold    : {args.threshold}")
    print(f"  Threads      : {args.threads} (faiss.omp_set_num_threads)")
    print(f"  Warmup/Reps  : {args.warmup} / {args.repeats}")
    print(f"  Data source  : {data_source}")
    print(f"  Encoder mode : {args.encoder}")
    print(f"  FAISS        : {faiss.__version__}")

    selected = [c for c in CELLS if c["name"] in set(args.cells.split(","))]
    results = []
    encoder_cache = {}  # encoder_kind -> (encoder, corpus_embeddings | None)
    for spec in selected:
        out = run_cell(spec, data, args, args.threads, encoder_cache)
        if out is not None:
            results.append(out)

    # Summary table
    print("\n" + "=" * 64)
    print("SUMMARY (search-only and end-to-end latency in ms)")
    print("=" * 64)
    print(f"{'Cell':<6}{'Config':<22}{'TP%':>6}{'FP%':>6}"
          f"{'srchP50':>9}{'srchP95':>9}{'e2eP50':>9}{'e2eP95':>9}"
          f"{'RAM/MB':>9}{'Disk/MB':>9}{'SQL/MB':>9}")
    for r in results:
        m = r["metrics"]
        mem = r["memory"]
        print(
            f"{r['name']:<6}{r['label']:<22}"
            f"{r['metrics']['tp_hit_rate']*100:>6.1f}{r['metrics']['fp_hit_rate']*100:>6.1f}"
            f"{m['search_latency_ms']['p50']:>9.3f}{m['search_latency_ms']['p95']:>9.3f}"
            f"{m['e2e_latency_ms']['p50']:>9.3f}{m['e2e_latency_ms']['p95']:>9.3f}"
            f"{mem['faiss_ram_bytes']/1e6:>9.2f}{mem['faiss_disk_bytes']/1e6:>9.2f}"
            f"{mem['sqlite_disk_bytes']/1e6:>9.2f}"
        )

    out_path = args.out or os.path.join(args.workdir, "results.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args),
            "data_source": data_source,
            "faiss_version": faiss.__version__,
            "results": results,
        }, f, indent=2)
    print(f"\nWrote JSON results to {out_path}")


if __name__ == "__main__":
    main()
