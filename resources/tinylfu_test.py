"""
Does TinyLFU's admission filter survive a SEMANTIC cache?

TinyLFU (Einziger & Friedman) estimates item frequency with a Count-Min
Sketch keyed by the item's exact key hash. That is sound when keys recur
verbatim. In a semantic cache the "key" is a natural-language prompt and
near-duplicates are PARAPHRASES -- every one is a distinct string, so every
one hashes to a different CMS bucket. Prediction: the sketch never
accumulates, every estimate is ~1, and the admission filter becomes a
coin flip.

Fix under test: key the sketch by an LSH signature of the prompt EMBEDDING
(banded random-hyperplane bits) instead of the exact string. Paraphrases
then collide *on purpose* and frequency accumulates.
"""
import numpy as np
from collections import OrderedDict
import policy_sim as ps


# ---------------------------------------------------------------- sketch
class CountMinSketch:
    def __init__(self, width=2048, depth=4, seed=0, aging_period=None):
        self.w, self.d = width, depth
        self.C = np.zeros((depth, width), dtype=np.int32)
        rng = np.random.default_rng(seed)
        self.salt = rng.integers(1, 2**31, size=depth)
        self.aging_period = aging_period
        self.n = 0

    def _idx(self, key):
        return [(hash((key, int(s))) % self.w) for s in self.salt]

    def add(self, key):
        for r, c in enumerate(self._idx(key)):
            if self.C[r, c] < 15:                     # 4-bit saturating counters
                self.C[r, c] += 1
        self.n += 1
        if self.aging_period and self.n >= self.aging_period:
            self.C >>= 1                              # halve everything
            self.n = 0

    def estimate(self, key):
        return int(min(self.C[r, c] for r, c in enumerate(self._idx(key))))


class LSHKeyer:
    """Banded random-hyperplane LSH. Returns one bucket key per band."""

    def __init__(self, dim, bits_per_band=6, bands=8, seed=0):
        rng = np.random.default_rng(seed)
        self.R = rng.normal(size=(bands * bits_per_band, dim))
        self.b, self.k = bands, bits_per_band

    def keys(self, q):
        bits = (self.R @ q > 0).astype(np.uint8)
        out = []
        for i in range(self.b):
            band = bits[i * self.k:(i + 1) * self.k]
            out.append((i, int(np.packbits(band, bitorder="little")[0])))
        return out


class SemanticSketch:
    """Frequency estimator over embeddings. mode: 'exact' | 'lsh'."""

    def __init__(self, dim, mode, capacity, seed=0, **lsh):
        self.mode = mode
        self.cms = CountMinSketch(width=max(2048, 8 * capacity), depth=4,
                                  seed=seed, aging_period=10 * capacity)
        self.lsh = LSHKeyer(dim, seed=seed, **lsh) if mode == "lsh" else None

    def add(self, q, uid):
        if self.mode == "exact":
            self.cms.add(uid)
        else:
            for k in self.lsh.keys(q):
                self.cms.add(k)

    def estimate(self, q, uid):
        if self.mode == "exact":
            return self.cms.estimate(uid)
        return max(self.cms.estimate(k) for k in self.lsh.keys(q))


# ------------------------------------------------------------ W-TinyLFU
class WTinyLFU:
    """Window LRU (1%) + main SLRU (20% probation / 80% protected).
    Admission filter arbitrates window-victim vs probation-victim."""

    def __init__(self, capacity, tau, dim, sketch_mode, seed=0):
        self.tau = tau
        self.wcap = max(1, capacity // 100)
        self.mcap = capacity - self.wcap
        self.pcap = max(1, int(0.2 * self.mcap))          # probation
        self.W, self.P, self.T = OrderedDict(), OrderedDict(), OrderedDict()
        self.vec, self.cid, self.uid = {}, {}, {}
        self.sk = SemanticSketch(dim, sketch_mode, capacity, seed=seed)
        self.next_id = 0
        self.admit_calls = self.admit_yes = 0

    def _nn(self, d, q):
        if not d:
            return None
        ids = list(d.keys())
        s = np.asarray([self.vec[i] for i in ids]) @ q
        j = int(np.argmax(s))
        return ids[j] if s[j] >= self.tau else None

    def lookup(self, q, true_cid, uid):
        self.sk.add(q, uid)
        for d, promote in ((self.T, False), (self.P, True), (self.W, False)):
            k = self._nn(d, q)
            if k is not None:
                if promote:                                # probation -> protected
                    del self.P[k]
                    self.T[k] = True
                    if len(self.T) > self.mcap - self.pcap:
                        dk, _ = self.T.popitem(last=False)
                        self.P[dk] = True
                else:
                    d.move_to_end(k)
                return True, self.cid[k] != true_cid
        return False, False

    def _drop(self, k):
        self.vec.pop(k, None); self.cid.pop(k, None); self.uid.pop(k, None)

    def insert(self, q, cid, uid):
        i = self.next_id; self.next_id += 1
        self.vec[i] = q; self.cid[i] = cid; self.uid[i] = uid
        self.W[i] = True
        if len(self.W) <= self.wcap:
            return
        cand, _ = self.W.popitem(last=False)               # window victim
        if len(self.P) + len(self.T) < self.mcap:
            self.P[cand] = True
            return
        victim = next(iter(self.P)) if self.P else next(iter(self.T))
        self.admit_calls += 1
        fc = self.sk.estimate(self.vec[cand], self.uid[cand])
        fv = self.sk.estimate(self.vec[victim], self.uid[victim])
        if fc > fv:                                        # admit candidate
            self.admit_yes += 1
            (self.P if victim in self.P else self.T).pop(victim)
            self._drop(victim)
            self.P[cand] = True
        else:
            self._drop(cand)


def run_wtlfu(trace, cids, cap, tau, mode, seed=0):
    c = WTinyLFU(cap, tau, trace.shape[1], mode, seed=seed)
    hits = 0
    for n, (q, cid) in enumerate(zip(trace, cids)):
        uid = f"q{n}"                                      # every prompt unique
        h, _ = c.lookup(q, cid, uid)
        if h:
            hits += 1
        else:
            c.insert(q, cid, uid)
    rate = c.admit_yes / c.admit_calls if c.admit_calls else 0.0
    return hits / len(cids), rate


# ------------------------------------------------------- sketch diagnostic
def sketch_health(trace, cids, mode, cap=200, seed=0):
    """Does the sketch estimate track TRUE cluster popularity?"""
    sk = SemanticSketch(trace.shape[1], mode, cap, seed=seed)
    for n, q in enumerate(trace[:8000]):
        sk.add(q, f"q{n}")
    est, true = [], []
    counts = np.bincount(cids[:8000], minlength=int(cids.max()) + 1)
    for n in range(8000, 9000):
        est.append(sk.estimate(trace[n], f"q{n}"))
        true.append(counts[cids[n]])
    est, true = np.asarray(est), np.asarray(true)
    r = np.corrcoef(est, true)[0, 1] if est.std() > 0 else 0.0
    return est.mean(), est.max(), r


if __name__ == "__main__":
    TAU = 0.65
    for name, fn in [("STATIONARY", ps.make_trace), ("DRIFTING", ps.drift_trace)]:
        tr, cids = fn(n_queries=15000, n_clusters=800, seed=1)
        print(f"\n{'='*74}\n{name}  (tau={TAU})\n{'='*74}")

        for mode in ("exact", "lsh"):
            m, mx, r = sketch_health(tr, cids, mode)
            print(f"  sketch[{mode:5}]  mean est={m:5.2f}  max={mx:3d}  "
                  f"corr(est, true popularity)={r:+.3f}")

        print(f"\n  {'cap':>5} | {'LRU':>8} | {'LFU':>8} | {'W-TLFU exact':>13} "
              f"| {'W-TLFU LSH':>11} | {'admit% ex/lsh':>14}")
        for cap in [100, 200, 400]:
            lru, _ = ps.run(tr, cids, cap, TAU, "LRU")
            lfu, _ = ps.run(tr, cids, cap, TAU, "LFU")
            he, ae = run_wtlfu(tr, cids, cap, TAU, "exact")
            hl, al = run_wtlfu(tr, cids, cap, TAU, "lsh")
            print(f"  {cap:>5} | {lru*100:7.2f}% | {lfu*100:7.2f}% | {he*100:12.2f}% "
                  f"| {hl*100:10.2f}% | {ae*100:6.1f}% /{al*100:5.1f}%")
