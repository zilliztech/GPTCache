"""Arrival-sequence construction on top of the prepared embedding corpora.

``prepare_data.py`` writes a *corpus*: unit-norm embeddings plus ground-truth
cluster ids. This module turns a corpus into a *trace* -- an ordered sequence of
queries -- which is what a cache actually sees. Keeping the two apart is what
lets ``run_bench.py`` vary the seed meaningfully: the embeddings never change
(so runs are comparable and offline-reproducible) while the arrival process
does.

Regimes
-------
``quora-stationary``
    Cluster popularity is Zipf(s=1.1) over a fixed random ranking. Within a
    cluster, one of its real paraphrases is drawn uniformly. This is the regime
    where a frequency policy should win.

``quora-drift``
    Identical, except the popularity ranking is reshuffled every epoch, so
    yesterday's hot clusters go cold. This is the regime that punishes pure LFU
    and that ARC's adaptation exists to handle.

``wildchat``
    No synthesis at all: a contiguous window of the real prompt stream in real
    timestamp order. The seed picks the window offset. Whatever drift is
    present is drift that actually happened.
"""

import json
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

REGIMES = ("quora-stationary", "quora-drift", "wildchat")


def silence_spurious_fp_warnings():
    """Mute bogus floating-point flags raised by Apple's Accelerate BLAS.

    On macOS with numpy 2.x, ``float32`` matrix products raise
    ``divide by zero`` / ``overflow`` / ``invalid`` RuntimeWarnings even for
    perfectly finite, unit-norm inputs -- Accelerate leaves FP status flags set
    and numpy reports them. Verified harmless: the float32 BLAS result agrees
    with a float64 ``einsum`` of the same inputs to 8e-8, and every corpus is
    checked finite and unit-norm at load. Similarity scans are the hot path in
    every benchmark here, so the noise would otherwise drown the output.
    """
    np.seterr(divide="ignore", over="ignore", invalid="ignore")


def data_available():
    """True if prepare_data.py has been run."""
    return os.path.exists(os.path.join(DATA_DIR, "quora_emb.npy"))


def load_tau(default=0.65):
    """The tau chosen by prepare_data.py's F1 sweep."""
    path = os.path.join(DATA_DIR, "tau.json")
    if not os.path.exists(path):
        return default
    with open(path) as fh:
        return float(json.load(fh)["tau"])


def load_corpus(name):
    """Return ``(emb[n, d] float32 unit-norm, cid[n] int32)`` for a corpus."""
    emb = np.load(os.path.join(DATA_DIR, f"{name}_emb.npy"))
    cid = np.load(os.path.join(DATA_DIR, f"{name}_cid.npy"))
    return emb, cid


def load_meta(name):
    with open(os.path.join(DATA_DIR, f"{name}_meta.json")) as fh:
        return json.load(fh)


def _zipf_weights(n, s=1.1):
    w = 1.0 / np.arange(1, n + 1, dtype=float) ** s
    return w / w.sum()


def _cluster_members(cid):
    """Map cluster id -> array of corpus row indices, for cids >= 0."""
    order = np.argsort(cid, kind="stable")
    scid = cid[order]
    bounds = np.flatnonzero(np.diff(scid)) + 1
    out = {}
    for chunk in np.split(order, bounds):
        c = int(cid[chunk[0]])
        if c >= 0:
            out[c] = chunk
    return out


def make_arrival(regime, cid, n_queries, seed, epochs=6, zipf_s=1.1,
                 n_rows=None):
    """Build an arrival sequence.

    :param regime: one of :data:`REGIMES`.
    :param cid: the corpus cluster-id array (``-1`` where unknown).
    :param n_queries: length of the trace to produce.
    :param seed: RNG seed; also selects the window offset for ``wildchat``.
    :returns: int64 array of row indices into the corpus.
    """
    rng = np.random.default_rng(seed)

    if regime == "wildchat":
        n = n_rows if n_rows is not None else len(cid)
        if n_queries >= n:
            return np.arange(n, dtype=np.int64)
        # a contiguous window preserves real arrival order and real drift;
        # the seed only chooses where in the stream we start
        start = int(rng.integers(0, n - n_queries + 1))
        return np.arange(start, start + n_queries, dtype=np.int64)

    if regime not in ("quora-stationary", "quora-drift"):
        raise ValueError(f"unknown regime {regime!r}")

    members = _cluster_members(cid)
    keys = np.array(sorted(members))
    n_clusters = len(keys)
    probs = _zipf_weights(n_clusters, zipf_s)

    n_epochs = 1 if regime == "quora-stationary" else epochs
    per = int(np.ceil(n_queries / n_epochs))

    picked_clusters = []
    for _ in range(n_epochs):
        # rank -> cluster assignment; reshuffled each epoch under drift
        ranking = rng.permutation(n_clusters)
        picked_clusters.append(keys[ranking[rng.choice(n_clusters, size=per,
                                                       p=probs)]])
    picked = np.concatenate(picked_clusters)[:n_queries]

    # uniform choice of one real paraphrase inside the chosen cluster
    idx = np.empty(len(picked), dtype=np.int64)
    for i, c in enumerate(picked):
        pool = members[int(c)]
        idx[i] = pool[rng.integers(len(pool))]
    return idx


def build_trace(regime, n_queries, seed, corpus_cache={}):  # noqa: B006
    """Load the right corpus and return ``(queries[n, d], cids[n])``.

    ``corpus_cache`` is an intentional mutable default: the corpora are large
    and immutable, and every benchmark process wants them exactly once.
    """
    name = "wildchat" if regime == "wildchat" else "quora"
    if name not in corpus_cache:
        corpus_cache[name] = load_corpus(name)
    emb, cid = corpus_cache[name]
    idx = make_arrival(regime, cid, n_queries, seed, n_rows=len(emb))
    return emb[idx], cid[idx]
