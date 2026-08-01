"""
policy_sim.py -- go/no-go screening harness for GPTCache eviction policies.

Trace-driven simulation of a SEMANTIC cache under two regimes:
  STATIONARY : Zipf-popular paraphrase clusters, fixed popularity
  DRIFTING   : popularity ranking reshuffled every epoch

Run:  python3 policy_sim.py
Reproduces the screening results used to reject redundancy-aware eviction
and to validate Semantic-ARC before committing implementation time.
"""
import numpy as np
from collections import OrderedDict
from collections import defaultdict

RNG = np.random.default_rng(0)


def make_trace(n_queries=20000, n_clusters=400, dim=64, zipf_s=1.1,
               spread=0.075, seed=0, n_topics=40, topic_spread=0.30,
               anisotropy=0.35):
    """Queries from Zipf-popular paraphrase clusters, clusters grouped into
    topics (so some clusters are genuinely close -> false-hit risk), plus a
    shared background direction to mimic embedding anisotropy."""
    rng = np.random.default_rng(seed)
    bg = rng.normal(size=dim); bg /= np.linalg.norm(bg)
    topics = rng.normal(size=(n_topics, dim))
    topics /= np.linalg.norm(topics, axis=1, keepdims=True)
    tof = rng.integers(n_topics, size=n_clusters)
    centers = topics[tof] + topic_spread * rng.normal(size=(n_clusters, dim))
    centers += anisotropy * bg
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    ranks = np.arange(1, n_clusters + 1)
    p = 1.0 / ranks ** zipf_s
    p /= p.sum()
    cids = rng.choice(n_clusters, size=n_queries, p=p)

    q = centers[cids] + spread * rng.normal(size=(n_queries, dim))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q, cids


class SemanticCache:
    """Cache of (embedding, cluster_id). Hit = cosine sim >= tau."""

    def __init__(self, capacity, tau, policy, redundancy_w=1.0):
        self.cap = capacity
        self.tau = tau
        self.policy = policy
        self.rw = redundancy_w
        self.E = np.zeros((capacity, 0))  # filled lazily
        self.emb = []      # list of vectors
        self.cid = []      # ground-truth cluster of stored entry
        self.last = []     # last-access time
        self.freq = []     # access count
        self.t = 0

    def _matrix(self):
        return np.asarray(self.emb)

    def lookup(self, q, true_cid):
        self.t += 1
        if not self.emb:
            return False, False
        sims = self._matrix() @ q
        j = int(np.argmax(sims))
        if sims[j] >= self.tau:
            self.last[j] = self.t
            self.freq[j] += 1
            false_hit = (self.cid[j] != true_cid)
            return True, false_hit
        return False, False

    def _victim(self):
        n = len(self.emb)
        if self.policy == "LRU":
            return int(np.argmin(self.last))
        if self.policy == "LFU":
            return int(np.argmin(self.freq))
        if self.policy == "FIFO":
            return 0
        if self.policy == "RANDOM":
            return int(RNG.integers(n))

        # --- redundancy-aware family ---
        M = self._matrix()
        S = M @ M.T
        np.fill_diagonal(S, -1.0)
        redundancy = S.max(axis=1)          # how well covered by a neighbour

        if self.policy == "COVERAGE":       # pure: evict most-redundant
            return int(np.argmax(redundancy))

        if self.policy == "HYBRID":
            # utility = value kept if retained. Redundant entries are cheap to
            # drop; hot entries are expensive to drop. Evict min utility.
            f = np.asarray(self.freq, dtype=float)
            rec = (np.asarray(self.last, dtype=float) - self.t) / max(self.cap, 1)
            util = np.log1p(f) + rec - self.rw * np.clip(redundancy, 0, None) * 3.0
            return int(np.argmin(util))
        raise ValueError(self.policy)

    def insert(self, q, cid):
        if len(self.emb) >= self.cap:
            v = self._victim()
            self.emb.pop(v); self.cid.pop(v); self.last.pop(v); self.freq.pop(v)
        self.emb.append(q); self.cid.append(cid)
        self.last.append(self.t); self.freq.append(1)


def run(trace, cids, capacity, tau, policy):
    c = SemanticCache(capacity, tau, policy)
    hits = false_hits = 0
    for q, cid in zip(trace, cids):
        h, fh = c.lookup(q, cid)
        if h:
            hits += 1
            false_hits += fh
        else:
            c.insert(q, cid)
    n = len(cids)
    return hits / n, (false_hits / hits if hits else 0.0)




def drift_trace(n_queries=15000, n_clusters=800, dim=64, zipf_s=1.1,
                seed=1, epochs=6, **kw):
    """Popularity ranking is reshuffled every epoch -> yesterday's hot
    clusters go cold. This is the regime where pure LFU pollutes."""
    rng = np.random.default_rng(seed)
    import numpy.linalg as la
    bg = rng.normal(size=dim); bg /= la.norm(bg)
    topics = rng.normal(size=(40, dim)); topics /= la.norm(topics,axis=1,keepdims=True)
    tof = rng.integers(40, size=n_clusters)
    centers = topics[tof] + 0.30*rng.normal(size=(n_clusters, dim)) + 0.35*bg
    centers /= la.norm(centers,axis=1,keepdims=True)
    p = 1.0/np.arange(1,n_clusters+1)**zipf_s; p/=p.sum()
    per = n_queries//epochs; cids=[]
    for e in range(epochs):
        perm = rng.permutation(n_clusters)
        cids.append(perm[rng.choice(n_clusters, size=per, p=p)])
    cids = np.concatenate(cids)
    qs = centers[cids] + 0.075*rng.normal(size=(len(cids), dim))
    qs /= la.norm(qs,axis=1,keepdims=True)
    return qs, cids



class SemanticARC:
    def __init__(self, capacity, tau):
        self.c = capacity
        self.tau = tau
        self.p = 0.0
        # each list: dict id -> embedding ; OrderedDict gives LRU order
        self.T1, self.T2 = OrderedDict(), OrderedDict()
        self.B1, self.B2 = OrderedDict(), OrderedDict()
        self.cid = {}          # id -> ground-truth cluster (eval only)
        self.next_id = 0

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _nn(d, q, tau):
        if not d:
            return None
        ids = list(d.keys())
        M = np.asarray([d[i] for i in ids])
        s = M @ q
        j = int(np.argmax(s))
        return ids[j] if s[j] >= tau else None

    def _replace(self, in_b2):
        if self.T1 and (len(self.T1) > self.p or
                        (in_b2 and len(self.T1) == self.p)):
            k, v = self.T1.popitem(last=False)
            self.B1[k] = v
        elif self.T2:
            k, v = self.T2.popitem(last=False)
            self.B2[k] = v

    # ---- main ----------------------------------------------------------
    def lookup(self, q, true_cid):
        k = self._nn(self.T1, q, self.tau)
        if k is not None:                       # T1 hit -> promote to T2
            v = self.T1.pop(k); self.T2[k] = v
            return True, self.cid[k] != true_cid
        k = self._nn(self.T2, q, self.tau)
        if k is not None:                       # T2 hit -> refresh
            self.T2.move_to_end(k)
            return True, self.cid[k] != true_cid
        return False, False

    def insert(self, q, cid):
        # --- ghost lookups drive adaptation ---
        gb1 = self._nn(self.B1, q, self.tau)
        gb2 = self._nn(self.B2, q, self.tau)

        if gb1 is not None:                     # recency was right -> grow T1
            d = max(1.0, len(self.B2) / max(len(self.B1), 1))
            self.p = min(self.c, self.p + d)
            self._replace(in_b2=False)
            self.B1.pop(gb1, None)
        elif gb2 is not None:                   # frequency was right -> grow T2
            d = max(1.0, len(self.B1) / max(len(self.B2), 1))
            self.p = max(0.0, self.p - d)
            self._replace(in_b2=True)
            self.B2.pop(gb2, None)
        else:                                   # true miss
            l1 = len(self.T1) + len(self.B1)
            if l1 == self.c:
                if len(self.T1) < self.c and self.B1:
                    self.B1.popitem(last=False); self._replace(False)
                elif self.T1:
                    self.T1.popitem(last=False)
            elif l1 < self.c and (l1 + len(self.T2) + len(self.B2)) >= self.c:
                if (l1 + len(self.T2) + len(self.B2)) == 2 * self.c and self.B2:
                    self.B2.popitem(last=False)
                self._replace(False)

        i = self.next_id; self.next_id += 1
        self.T1[i] = q; self.cid[i] = cid
        # promoted entries (ghost hits) belong in T2
        if gb1 is not None or gb2 is not None:
            self.T1.pop(i); self.T2[i] = q


def run_arc(trace, cids, cap, tau):
    c = SemanticARC(cap, tau); hits = 0
    for q, cid in zip(trace, cids):
        h, _ = c.lookup(q, cid)
        if h:
            hits += 1
        else:
            c.insert(q, cid)
    return hits / len(cids)




def run_arc(trace, cids, cap, tau):
    c = SemanticARC(cap, tau); hits = 0
    for q, cid in zip(trace, cids):
        h, _ = c.lookup(q, cid)
        if h: hits += 1
        else: c.insert(q, cid)
    return hits / len(cids)


if __name__ == "__main__":
    TAU = 0.65
    for name, fn in [("STATIONARY", make_trace), ("DRIFTING", drift_trace)]:
        tr, cids = fn(n_queries=15000, n_clusters=800, seed=1)
        print(f"\n=== {name} (tau={TAU}) ===")
        print(f"{'cap':>5} | {'LRU':>8} | {'LFU':>8} | {'COVERAGE':>9} | {'ARC':>8}")
        for cap in [100, 200, 400]:
            lru,_ = run(tr,cids,cap,TAU,"LRU")
            lfu,_ = run(tr,cids,cap,TAU,"LFU")
            cov,_ = run(tr,cids,cap,TAU,"COVERAGE")
            arc   = run_arc(tr,cids,cap,TAU)
            print(f"{cap:>5} | {lru*100:7.2f}% | {lfu*100:7.2f}% | {cov*100:8.2f}% | {arc*100:7.2f}%")
