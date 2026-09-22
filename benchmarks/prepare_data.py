"""Build the two real embedding traces used by the Semantic-ARC benchmarks.

Outputs (to ``benchmarks/data/``, gitignored -- regenerate with this script):

===========================  =========================================
``{name}_emb.npy``           float32 ``[n, 384]``, unit-norm embeddings
``{name}_cid.npy``           int32 ``[n]`` ground-truth cluster ids
                             (``-1`` where no ground truth exists)
``{name}_meta.json``         model, dims, count, sha256 of every input
``tau.json``                 the similarity threshold, and how it was picked
===========================  =========================================

Two corpora
-----------
``quora``
    Quora Question Pairs. Questions linked by an ``is_duplicate=True`` edge are
    merged with union-find into paraphrase clusters. The cluster id is real
    ground truth, which is what makes **false hit rate** measurable: a semantic
    cache hit whose stored entry belongs to a different cluster is a wrong
    answer served to the user, and almost nobody reports that number.

    Quora has no natural arrival order, so ``run_bench.py`` synthesises one
    (Zipf-popular clusters, uniform choice of paraphrase within a cluster).
    The *text* is real, the *order* is not; this is the stationary regime.

``wildchat``
    allenai/WildChat-1M -- real user prompts to a real chat assistant, carrying
    per-conversation UTC timestamps. We keep first-turn English user prompts and
    sort by timestamp, so the arrival order is genuine and the popularity drift
    is whatever actually happened, not a drift model we invented. There is no
    paraphrase ground truth here, so ``cid`` is all ``-1`` and false hit rate is
    not reported for this trace.

    (The brief named LMSYS-Chat-1M. That dataset is gated and unavailable to
    this account; WildChat-1M is the ungated equivalent and additionally exposes
    timestamps, which LMSYS only encodes implicitly as row order.)

Embedding model is ``sentence-transformers/all-MiniLM-L6-v2`` (384-d). Vectors
are L2-normalised, so cosine similarity is a plain dot product everywhere
downstream.

Choosing tau
------------
The brief specifies "the value maximising F1 between 'same cluster' and
'sim >= tau'". Taken literally over *random question pairs* that criterion is
unusable here, and the script reports why rather than hiding it: the F1 curve
is flat to ~1e-4 across tau in [0.40, 0.50] (so the argmax is arbitrary), and
more importantly a cache never classifies a random pair -- it takes the
**maximum** similarity over every resident entry, so the per-pair
false-positive rate compounds with capacity. Measured on this corpus, the
pairwise argmax tau=0.43 makes a 1600-entry cache find a wrong-cluster
neighbour above threshold for ~85% of queries.

So we keep the objective and fix the decision rule: tau is the value maximising
F1 of the actual serving decision (top-1 resident neighbour, served iff
cos >= tau) on a simulated cache running the stationary arrival process. Both
numbers are printed and both land in ``tau.json``; only the second is used.

Usage
-----
::

    python benchmarks/prepare_data.py                  # both traces, defaults
    python benchmarks/prepare_data.py --only quora     # just one
    python benchmarks/prepare_data.py --device cpu     # force CPU

Runtime is dominated by embedding: roughly 4-8 minutes total on an M-series
Mac using the MPS backend, plus ~1 GB of dataset download on first run
(cached by huggingface_hub thereafter).
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

# Overridable via BENCH_DATA_DIR for running this script outside the container
# against corpora kept elsewhere. The pipeline never sets it: one corpus size
# means one set of reference numbers.
DATA_DIR = (os.environ.get("BENCH_DATA_DIR")
            or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384

QUORA_REPO = "sentence-transformers/quora-duplicates"
QUORA_FILE = "pair-class/train-00000-of-00001.parquet"
WILDCHAT_REPO = "allenai/WildChat-1M"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sha256_file(path, chunk=1 << 20):
    """Hash a file's bytes. Used to pin exactly which input produced a trace."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class UnionFind:
    """Iterative union-find with path halving and union by size."""

    def __init__(self):
        self.parent = {}
        self.size = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1
            return x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def groups(self):
        out = defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return out


# Provenance of the most recent embed() call, stamped into every *_meta.json by
# write_trace(). Batch size is the one knob that perturbs the vectors -- changing
# it moves them at 1e-7, enough to flip a handful of near-threshold similarity
# decisions -- so it belongs on disk next to the input hashes rather than only in
# whoever's shell history. Both this file and prepare_data_ext.py go through
# these two functions, so recording it here covers every corpus.
_EMBED_PROVENANCE = {}


def embed(texts, device, batch_size=512):
    """Embed and L2-normalise. Returns float32 ``[len(texts), DIM]``."""
    from sentence_transformers import SentenceTransformer

    _EMBED_PROVENANCE.update(
        {"model": MODEL_NAME, "device": device, "batch_size": batch_size}
    )
    model = SentenceTransformer(MODEL_NAME, device=device)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    # belt and braces: encode(normalize_embeddings=True) already does this, but
    # every downstream file treats "dot product == cosine" as an invariant.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def pick_device(requested):
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover - torch always present in practice
        pass
    return "cpu"


def write_trace(name, emb, cid, meta):
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, f"{name}_emb.npy"), emb.astype("float32"))
    np.save(os.path.join(DATA_DIR, f"{name}_cid.npy"), cid.astype("int32"))
    if _EMBED_PROVENANCE:
        meta = {**meta, "embed": dict(_EMBED_PROVENANCE)}
    with open(os.path.join(DATA_DIR, f"{name}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  wrote {name}_emb.npy {emb.shape} / {name}_cid.npy {cid.shape}")


# --------------------------------------------------------------------------
# quora
# --------------------------------------------------------------------------
def build_quora(args, device):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    print("\n=== quora: Quora Question Pairs -> paraphrase clusters ===")
    path = hf_hub_download(QUORA_REPO, QUORA_FILE, repo_type="dataset")
    table = pq.read_table(path).to_pydict()
    s1, s2, label = table["sentence1"], table["sentence2"], table["label"]
    n_dup = int(sum(label))
    print(f"  {len(label)} pairs, {n_dup} labelled is_duplicate=True")

    uf = UnionFind()
    for a, b, lab in zip(s1, s2, label):
        if lab == 1:
            uf.union(a, b)

    groups = [g for g in uf.groups().values() if len(g) >= args.min_cluster]
    groups.sort(key=len, reverse=True)
    if args.max_clusters and len(groups) > args.max_clusters:
        # keep a size-stratified sample rather than the head, so the corpus is
        # not biased toward the handful of giant clusters
        rng = np.random.default_rng(0)
        keep = rng.choice(len(groups), size=args.max_clusters, replace=False)
        groups = [groups[i] for i in sorted(keep)]

    texts, cids = [], []
    for cid, members in enumerate(groups):
        for q in sorted(members):  # sorted -> deterministic corpus order
            texts.append(q)
            cids.append(cid)
    cids = np.asarray(cids, dtype="int32")
    sizes = np.array([len(g) for g in groups])
    print(
        f"  kept {len(groups)} clusters (size >= {args.min_cluster}), "
        f"{len(texts)} questions; cluster size mean {sizes.mean():.2f} "
        f"median {np.median(sizes):.0f} max {sizes.max()}"
    )

    print(f"  embedding {len(texts)} questions on {device} ...")
    emb = embed(texts, device, args.batch_size)

    meta = {
        "name": "quora",
        "model": MODEL_NAME,
        "dims": DIM,
        "count": int(emb.shape[0]),
        "n_clusters": int(len(groups)),
        "min_cluster_size": args.min_cluster,
        "has_ground_truth_clusters": True,
        "arrival_order": "synthesised by run_bench.py (Zipf over clusters)",
        "inputs": [{"repo": QUORA_REPO, "file": QUORA_FILE,
                    "sha256": sha256_file(path)}],
    }
    write_trace("quora", emb, cids, meta)
    return emb, cids


# --------------------------------------------------------------------------
# wildchat
# --------------------------------------------------------------------------
def build_wildchat(args, device):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    print("\n=== wildchat: WildChat-1M -> real prompts in real arrival order ===")
    api = HfApi()
    info = api.repo_info(WILDCHAT_REPO, repo_type="dataset")
    files = sorted(s.rfilename for s in info.siblings
                   if s.rfilename.endswith(".parquet"))[: args.wildchat_shards]

    prompts, stamps, inputs = [], [], []
    for fname in files:
        path = hf_hub_download(WILDCHAT_REPO, fname, repo_type="dataset")
        inputs.append({"repo": WILDCHAT_REPO, "file": fname,
                       "sha256": sha256_file(path)})
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(
            batch_size=4096, columns=["conversation", "timestamp", "language"]
        ):
            for row in batch.to_pylist():
                if row["language"] != "English":
                    continue
                convo = row["conversation"]
                if not convo or convo[0].get("role") != "user":
                    continue
                text = (convo[0].get("content") or "").strip()
                if not (args.min_chars <= len(text) <= args.max_chars):
                    continue
                prompts.append(text)
                stamps.append(row["timestamp"])
        print(f"  {fname}: running total {len(prompts)} prompts")
        if len(prompts) >= args.n_wildchat:
            break

    order = np.argsort(np.asarray(stamps, dtype="datetime64[us]"), kind="stable")
    order = order[: args.n_wildchat]
    prompts = [prompts[i] for i in order]
    print(f"  {len(prompts)} first-turn English prompts, sorted by timestamp")
    print(f"  span {stamps[order[0]]}  ->  {stamps[order[-1]]}")

    print(f"  embedding {len(prompts)} prompts on {device} ...")
    emb = embed(prompts, device, args.batch_size)
    # no paraphrase ground truth in WildChat -> -1 everywhere, per the brief
    cids = np.full(len(prompts), -1, dtype="int32")

    meta = {
        "name": "wildchat",
        "model": MODEL_NAME,
        "dims": DIM,
        "count": int(emb.shape[0]),
        "n_clusters": 0,
        "has_ground_truth_clusters": False,
        "arrival_order": "real, sorted by conversation timestamp (UTC)",
        "time_span": [str(stamps[order[0]]), str(stamps[order[-1]])],
        "filters": {"language": "English", "turn": "first user message",
                    "min_chars": args.min_chars, "max_chars": args.max_chars},
        "inputs": inputs,
    }
    write_trace("wildchat", emb, cids, meta)
    return emb, cids


# --------------------------------------------------------------------------
# tau selection -- acceptance criterion for step 2
# --------------------------------------------------------------------------
def similarity_report(emb, cids, rng, n_pairs=400_000):
    """Sample same-cluster and different-cluster pairs, report percentiles."""
    by_cluster = defaultdict(list)
    for i, c in enumerate(cids):
        by_cluster[int(c)].append(i)
    multi = [v for v in by_cluster.values() if len(v) >= 2]

    # positives: two distinct members of the same cluster
    pos_a, pos_b = [], []
    weights = np.array([len(v) for v in multi], dtype=float)
    weights /= weights.sum()
    picks = rng.choice(len(multi), size=n_pairs, p=weights)
    for k in picks:
        members = multi[k]
        i, j = rng.choice(len(members), size=2, replace=False)
        pos_a.append(members[i])
        pos_b.append(members[j])

    # negatives: two questions from different clusters
    neg_a = rng.integers(0, len(cids), size=n_pairs)
    neg_b = rng.integers(0, len(cids), size=n_pairs)
    keep = cids[neg_a] != cids[neg_b]
    neg_a, neg_b = neg_a[keep], neg_b[keep]

    pos = np.einsum("ij,ij->i", emb[pos_a], emb[pos_b])
    neg = np.einsum("ij,ij->i", emb[neg_a], emb[neg_b])
    return pos, neg


def choose_tau(pos, neg, grid=np.arange(0.30, 0.96, 0.005)):
    """tau maximising F1 of 'sim >= tau' against 'same cluster'."""
    best = (0.0, None, None, None)
    rows = []
    for tau in grid:
        tp = float((pos >= tau).sum())
        fp = float((neg >= tau).sum())
        fn = float((pos < tau).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append((float(tau), prec, rec, f1))
        if f1 > best[0]:
            best = (f1, float(tau), prec, rec)
    return best, rows


def _lru_cache_f1(queries, qcids, tau, cap):
    """Simulate an LRU semantic cache and score its *serving decision*.

    This is the decision a cache really makes: take the single most similar
    resident entry and serve it iff its cosine is >= tau.

    ==============  ====================================================
    true positive   served, and the entry's cluster is the query's
    false positive  served, but from a different cluster -> wrong answer
    false negative   not served, although a same-cluster entry was resident
    ==============  ====================================================
    """
    dim = queries.shape[1]
    M = np.zeros((cap, dim), dtype=np.float32)
    C = np.full(cap, -2, dtype=np.int64)
    last = np.zeros(cap, dtype=np.int64)
    n = 0
    tp = fp = fn = 0
    for t, (q, qc) in enumerate(zip(queries, qcids), start=1):
        if n:
            sims = M[:n] @ q
            j = int(sims.argmax())
            if sims[j] >= tau:
                last[j] = t
                if C[j] == qc:
                    tp += 1
                else:
                    fp += 1
                continue
            if np.any(C[:n] == qc):
                fn += 1
        slot = n if n < cap else int(last[:n].argmin())
        n = min(n + 1, cap)
        M[slot] = q
        C[slot] = qc
        last[slot] = t
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, prec, rec


def choose_tau_workload(emb, cids, cap, n_queries, grid, seed=0,
                        regime="quora-stationary"):
    """tau maximising serving-decision F1 on a realistic arrival process.

    The arrival process is the same Zipf-over-clusters one ``traces.py`` uses
    for the stationary regime, so the cache fills with genuinely popular
    clusters rather than a uniform sample of the corpus.

    ``regime`` names the stationary regime of the corpus being measured. It
    exists so a second cluster corpus can report its own tau as a diagnostic
    (see ``prepare_data_ext.py``); the default keeps the Quora path unchanged.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import traces  # local import: only needed here

    idx = traces.make_arrival(regime, cids, n_queries, seed)
    q, qc = emb[idx], cids[idx].astype(np.int64)
    rows = []
    for tau in grid:
        f1, prec, rec = _lru_cache_f1(q, qc, float(tau), cap)
        rows.append((float(tau), prec, rec, f1))
    best = max(rows, key=lambda r: r[3])
    return best, rows


def report_and_pick_tau(emb, cids, cap=400, n_queries=10_000):
    print("\n=== tau selection (Quora, ground-truth clusters) ===")
    rng = np.random.default_rng(0)
    pos, neg = similarity_report(emb, cids, rng)
    pct = [1, 5, 25, 50, 75, 95, 99]
    print(f"  intra-cluster cosine (n={len(pos)}):")
    print("    " + "  ".join(f"p{p}={np.percentile(pos, p):.3f}" for p in pct))
    print(f"  inter-cluster cosine (n={len(neg)}):")
    print("    " + "  ".join(f"p{p}={np.percentile(neg, p):.3f}" for p in pct))
    print(f"  mean intra={pos.mean():.3f}  mean inter={neg.mean():.3f}")

    # ---- criterion A: the brief's literal pairwise F1 (reported, NOT used) --
    (pf1, ptau, pprec, prec_), _ = choose_tau(pos, neg)
    print(f"\n  [A] pairwise F1 over random question pairs")
    print(f"      best F1 = {pf1:.4f} at tau = {ptau:.3f} "
          f"(P {pprec:.4f}, R {prec_:.4f})")
    print("      NOT USED. A cache does not classify a random pair: it takes")
    print("      the MAX similarity over every resident entry, so the per-pair")
    print("      false-positive rate compounds with capacity. At tau=%.3f a"
          % ptau)
    print("      1600-entry cache finds a wrong-cluster neighbour above")
    print("      threshold for ~85%% of queries. The curve is also flat to")
    print("      ~1e-4 across tau in [0.40, 0.50], so the argmax is arbitrary.")

    # ---- criterion B: same F1, evaluated on the real serving decision ------
    grid = np.arange(0.50, 0.96, 0.01)
    print(f"\n  [B] serving-decision F1 on a simulated cache "
          f"(cap={cap}, {n_queries} queries)  <- USED")
    (tau, prec, rec, f1), rows = choose_tau_workload(
        emb, cids, cap, n_queries, grid
    )
    print(f"      best F1 = {f1:.4f} at tau = {tau:.3f} "
          f"(precision {prec:.4f}, recall {rec:.4f}, "
          f"false-hit rate {1 - prec:.4f})")
    print("      neighbourhood:")
    for t, p, r, f in rows:
        if abs(t - tau) < 0.0451:
            mark = " <-" if abs(t - tau) < 1e-9 else ""
            print(f"        tau={t:.3f}  P={p:.4f}  R={r:.4f}  F1={f:.4f}"
                  f"  falsehit={1 - p:.4f}{mark}")

    out = {
        "tau": round(tau, 3),
        "selected_on": "quora",
        "criterion": ("max F1 of the cache serving decision (top-1 resident "
                      "neighbour with cos >= tau) against same-cluster ground "
                      f"truth, simulated at capacity {cap} over {n_queries} "
                      "queries of the stationary arrival process"),
        "pairwise_f1_tau": round(ptau, 3),
        "pairwise_f1": pf1,
        "why_not_pairwise": (
            "Pairwise F1 is flat to ~1e-4 over tau in [0.40, 0.50] and its "
            "argmax (0.43) ignores that the cache maximises similarity over "
            "all resident entries; at capacity 1600 that tau produces a false "
            "hit rate of ~0.85. Same objective, correct decision rule."
        ),
        "f1": f1, "precision": prec, "recall": rec,
        "false_hit_rate_at_tau": 1 - prec,
        "sweep": [{"tau": t, "precision": p, "recall": r, "f1": f}
                  for t, p, r, f in rows],
        "intra_cluster_cosine": {f"p{p}": float(np.percentile(pos, p)) for p in pct},
        "inter_cluster_cosine": {f"p{p}": float(np.percentile(neg, p)) for p in pct},
        "n_pos_pairs": int(len(pos)), "n_neg_pairs": int(len(neg)),
        "model": MODEL_NAME,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "tau.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  wrote data/tau.json -- every later experiment uses tau={out['tau']}")
    return out["tau"]


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=["quora", "wildchat"], default=None,
                    help="build just one trace (default: both)")
    ap.add_argument("--min-cluster", type=int, default=3,
                    help="minimum Quora paraphrase-cluster size to keep")
    ap.add_argument("--max-clusters", type=int, default=0,
                    help="subsample to at most N Quora clusters (0 = no cap)")
    ap.add_argument("--wildchat-shards", type=int, default=3,
                    help="how many WildChat parquet shards to download (~220 MB each)")
    ap.add_argument("--n-wildchat", type=int, default=150_000,
                    help="number of first-turn prompts to keep")
    ap.add_argument("--min-chars", type=int, default=8)
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import traces

    traces.silence_spurious_fp_warnings()

    device = pick_device(args.device)
    print(f"embedding model : {MODEL_NAME}")
    print(f"device          : {device}")
    print(f"output dir      : {DATA_DIR}")

    quora = None
    if args.only in (None, "quora"):
        quora = build_quora(args, device)
    if args.only in (None, "wildchat"):
        build_wildchat(args, device)

    if quora is not None:
        report_and_pick_tau(*quora)
    else:
        print("\n(skipped tau selection: it is defined on the Quora clusters)")
    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
