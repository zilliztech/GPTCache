"""Parameterised arrival processes, for locating the ARC/LRU crossover.

``traces.py`` ships three fixed regimes: stationary, total-reshuffle drift, and
a real prompt stream. That is enough to show ARC has the best worst case, but
it cannot answer *where* the boundary is -- the two synthetic regimes are the
two endpoints of an axis with nothing sampled in between.

This module makes that axis continuous. Popularity is still Zipf over
ground-truth paraphrase clusters, but two knobs are exposed:

``drift_rate`` (delta)
    Fraction of the popularity ranking re-permuted at each epoch boundary.
    ``0.0`` holds the ranking fixed for the whole trace and reproduces
    ``quora-stationary``; ``1.0`` re-permutes everything every epoch and
    reproduces ``quora-drift``. Intermediate values displace a random delta of
    ranks -- so delta is exactly the per-epoch probability that any given hot
    cluster is knocked out of the head of the distribution.

``zipf_s``
    Popularity skew. Low s is a flat distribution with no meaningful hot set
    (nothing is worth caching); high s concentrates traffic on a handful of
    clusters (everything fits, and the policy stops mattering).

``n_epochs`` sets drift *granularity*: many epochs with small delta is gradual
churn, few epochs with large delta is abrupt regime change. The product
``n_epochs * drift_rate`` is roughly the number of times the hot set turns
over across the trace.

The corpus itself is never regenerated -- these are arrival orders over the
embeddings ``prepare_data.py`` and ``prepare_data_ext.py`` already wrote, so
every cell of a sweep is comparable and offline-reproducible.
"""

import numpy as np

from traces import _cluster_members, _zipf_weights, load_corpus

CLUSTER_CORPORA = ("quora", "stackexchange", "wikianswers")
REAL_CORPORA = ("wildchat", "wildchat-long")


def make_param_arrival(cid, n_queries, seed, zipf_s=1.1, drift_rate=0.0,
                       n_epochs=6, max_clusters=0):
    """Zipf-over-clusters arrival with a tunable per-epoch drift rate.

    :param cid: corpus cluster-id array (``-1`` where unknown).
    :param drift_rate: fraction of the ranking re-permuted per epoch, in [0, 1].
    :param n_epochs: number of equal segments the trace is split into.
    :param max_clusters: if non-zero, restrict the universe to this many
        clusters, which shrinks the working set without touching capacity.
    :returns: int64 row indices into the corpus.
    """
    if not 0.0 <= drift_rate <= 1.0:
        raise ValueError(f"drift_rate must be in [0, 1], got {drift_rate}")
    rng = np.random.default_rng(seed)

    members = _cluster_members(cid)
    keys = np.array(sorted(members))
    if max_clusters and len(keys) > max_clusters:
        keys = keys[rng.choice(len(keys), size=max_clusters, replace=False)]
        keys.sort()
    n_clusters = len(keys)
    probs = _zipf_weights(n_clusters, zipf_s)

    per = int(np.ceil(n_queries / n_epochs))
    ranking = rng.permutation(n_clusters)     # rank -> index into keys
    k = int(round(drift_rate * n_clusters))

    picked = []
    for epoch in range(n_epochs):
        if epoch and k >= 2:
            # displace a random delta-fraction of ranks by permuting the
            # clusters sitting at those ranks among themselves. Selecting
            # positions uniformly means every rank -- hot or cold -- has
            # probability delta of moving, which is what makes delta readable
            # as "chance a hot cluster goes cold this epoch".
            pos = rng.choice(n_clusters, size=k, replace=False)
            ranking[pos] = ranking[pos][rng.permutation(k)]
        picked.append(keys[ranking[rng.choice(n_clusters, size=per, p=probs)]])
    picked = np.concatenate(picked)[:n_queries]

    idx = np.empty(len(picked), dtype=np.int64)
    for i, c in enumerate(picked):
        pool = members[int(c)]
        idx[i] = pool[rng.integers(len(pool))]
    return idx


_CORPUS_CACHE = {}


def _corpus(name):
    if name not in _CORPUS_CACHE:
        _CORPUS_CACHE[name] = load_corpus(name)
    return _CORPUS_CACHE[name]


def build_param_trace(corpus, n_queries, seed, **kw):
    """``(queries[n, d], cids[n])`` for a synthetic regime over ``corpus``."""
    emb, cid = _corpus(corpus)
    idx = make_param_arrival(cid, n_queries, seed, **kw)
    return emb[idx], cid[idx]


def build_real_trace(corpus, n_queries, seed, stride=1):
    """A contiguous window of a real, timestamp-ordered prompt stream.

    :param stride: keep every ``stride``-th arrival. ``1`` is the stream as
        recorded. Larger values thin it in time *without* changing its span,
        which is the control for ``wildchat-long``: that corpus is both longer
        and (because it is subsampled to 150k) sparser than ``wildchat``, so
        the two effects have to be separated before either is credited.
    """
    emb, cid = _corpus(corpus)
    n = len(emb)
    need = n_queries * stride
    if need >= n:
        idx = np.arange(0, n, stride)[:n_queries]
    else:
        rng = np.random.default_rng(seed)
        start = int(rng.integers(0, n - need + 1))
        idx = np.arange(start, start + need, stride)[:n_queries]
    return emb[idx], cid[idx]
