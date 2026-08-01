"""STEP 3 GATE -- replicate the screening results on real embeddings.

The whole project rests on two measurements that were first made on synthetic
64-d embeddings (see ``resources/RESULTS.md``). Before a line of policy code is
written they have to survive contact with real sentence embeddings of real
questions. This script reruns both, using the corpora built by
``prepare_data.py``.

Check 1 -- an exact-key frequency sketch carries no signal
    A Count-Min Sketch keyed by the item's exact identity is the history
    mechanism of the TinyLFU family (Einziger & Friedman, PDP 2014; Einziger,
    Friedman & Manes, ACM TOS 2017). It is sound whenever keys recur verbatim.
    In a semantic cache the identity of a query is a fresh row id (GPTCache
    assigns one per miss) and the text is a paraphrase that never repeats
    verbatim -- so every query lands in a different bucket and the estimate
    never accumulates.

    We measure the correlation between the sketch's estimate for a query and
    the *true* popularity of that query's paraphrase cluster.
    **Pass if |corr| < 0.15 for the exact-keyed sketch.**

    Note carefully what is and is not being claimed. This says nothing about
    whether W-TinyLFU is a good policy -- it is an excellent one. It says that
    the *keying* it inherits from the exact-match world carries no signal on a
    semantic workload. That is a property of the keying, not of the policy. An
    LSH-keyed sketch is reported alongside purely as a contrast, to show the
    signal is recoverable; implementing one is explicitly out of scope
    (future work).

Check 2 -- semantic ghosts beat exact ghosts
    ARC (Megiddo & Modha, FAST 2003) adapts by consulting ghost lists of
    recently-evicted items. Keyed exactly, those ghosts can never fire in a
    semantic cache, so ``p`` never moves and the adaptive machinery is inert.
    Matching ghosts by embedding similarity revives it.
    **Pass if semantic-ghost ARC beats exact-ghost ARC by a clear margin
    (>= 3 percentage points) on a drifting trace.**

Usage::

    python benchmarks/prepare_data.py     # first
    python benchmarks/gate_screening.py

Exit status is 0 if both checks pass, 1 otherwise. Results are written to
``benchmarks/results/gate_screening.json`` and ``gate_screening.txt``.
"""

import hashlib
import json
import os
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traces  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SKETCH_CORR_MAX = 0.15      # |corr| below this == "no signal"
GHOST_MARGIN_MIN = 3.0      # percentage points


# ==========================================================================
# a similarity-searchable ordered set
# ==========================================================================
class VecSet:
    """Ordered set of ids with unit-norm vectors, supporting nearest-neighbour.

    Insertion order is LRU order (leftmost == least recently used), exactly as
    ``OrderedDict`` gives. The vectors live in one contiguous matrix so a
    similarity scan is a single BLAS call instead of a Python loop; removal is
    O(1) via swap-with-last.
    """

    def __init__(self, dim, capacity_hint=64):
        self._dim = dim
        self._M = np.zeros((max(capacity_hint, 1), dim), dtype=np.float32)
        self._row_of = {}          # id -> row in _M
        self._id_at = []           # row -> id
        self._order = OrderedDict()  # id -> None, in LRU order

    def __len__(self):
        return len(self._order)

    def __contains__(self, key):
        return key in self._order

    def __iter__(self):
        return iter(self._order)

    def add(self, key, vec):
        """Insert (or refresh) ``key`` at the MRU end."""
        if key in self._order:
            self._M[self._row_of[key]] = vec
            self._order.move_to_end(key)
            return
        n = len(self._id_at)
        if n == self._M.shape[0]:
            self._M = np.resize(self._M, (max(2 * n, 4), self._dim))
        self._M[n] = vec
        self._row_of[key] = n
        self._id_at.append(key)
        self._order[key] = None

    def touch(self, key):
        self._order.move_to_end(key)

    def remove(self, key):
        """Drop ``key``. O(1): the last row is swapped into its slot."""
        row = self._row_of.pop(key)
        last = len(self._id_at) - 1
        if row != last:
            moved = self._id_at[last]
            self._M[row] = self._M[last]
            self._id_at[row] = moved
            self._row_of[moved] = row
        self._id_at.pop()
        del self._order[key]

    def pop_lru(self):
        key = next(iter(self._order))
        vec = self._M[self._row_of[key]].copy()
        self.remove(key)
        return key, vec

    def nearest(self, q, tau):
        """Id of the most similar member if its cosine >= ``tau``, else None."""
        n = len(self._id_at)
        if n == 0:
            return None
        sims = self._M[:n] @ q
        j = int(sims.argmax())
        return self._id_at[j] if sims[j] >= tau else None


# ==========================================================================
# baseline semantic cache (LRU / LFU / FIFO / RR)
# ==========================================================================
class SemanticCache:
    """Cache of (embedding, cluster id). A hit is cosine >= tau.

    Mirrors ``resources/policy_sim.py`` so the screening numbers are directly
    comparable; the only change is the O(1) vector bookkeeping.
    """

    def __init__(self, capacity, tau, policy, dim, seed=0):
        self.cap = capacity
        self.tau = tau
        self.policy = policy
        self.store = VecSet(dim, capacity)
        self.cid = {}
        self.freq = {}
        self.rng = np.random.default_rng(seed)
        self.next_id = 0

    def lookup(self, q, true_cid):
        k = self.store.nearest(q, self.tau)
        if k is None:
            return False, False
        self.store.touch(k)
        self.freq[k] += 1
        return True, (self.cid[k] != true_cid and true_cid >= 0)

    def _victim(self):
        if self.policy in ("LRU", "FIFO"):
            return next(iter(self.store))
        if self.policy == "LFU":
            return min(self.store, key=lambda k: self.freq[k])
        if self.policy == "RR":
            keys = list(self.store)
            return keys[int(self.rng.integers(len(keys)))]
        raise ValueError(self.policy)

    def insert(self, q, cid):
        if len(self.store) >= self.cap:
            v = self._victim()
            self.store.remove(v)
            del self.cid[v], self.freq[v]
        i = self.next_id
        self.next_id += 1
        self.store.add(i, q)
        self.cid[i] = cid
        self.freq[i] = 1


def run_baseline(queries, cids, cap, tau, policy, seed=0):
    c = SemanticCache(cap, tau, policy, queries.shape[1], seed)
    hits = false_hits = 0
    for q, cid in zip(queries, cids):
        h, fh = c.lookup(q, int(cid))
        if h:
            hits += 1
            false_hits += fh
        else:
            c.insert(q, int(cid))
    n = len(cids)
    return hits / n, (false_hits / hits if hits else 0.0)


# ==========================================================================
# screening ARC
# ==========================================================================
class ScreeningARC:
    """ARC over a semantic cache, with switchable ghost matching.

    This is the *screening* implementation -- deliberately standalone and frozen
    at the state of the gate. The shipped policy is
    ``gptcache/manager/eviction/arc.py``; the two are checked against each other
    in the unit tests.

    ``ghost_matching='exact'`` reproduces classic ARC: a ghost fires only on an
    identical key. Since GPTCache mints a fresh row id for every miss, that can
    never happen -- which is exactly the point being measured.
    """

    def __init__(self, capacity, tau, dim, ghost_matching="semantic"):
        self.c = capacity
        self.tau = tau
        self.p = 0.0
        self.semantic = ghost_matching == "semantic"
        self.T1 = VecSet(dim, capacity)
        self.T2 = VecSet(dim, capacity)
        self.B1 = VecSet(dim, capacity)
        self.B2 = VecSet(dim, capacity)
        self.cid = {}
        self.next_id = 0
        self.p_trace = []

    def _ghost(self, d, q, key=None):
        if self.semantic:
            return d.nearest(q, self.tau)
        return key if (key is not None and key in d) else None

    def _replace(self, in_b2):
        if len(self.T1) > 0 and (
            len(self.T1) > self.p or (in_b2 and len(self.T1) == self.p)
        ):
            k, v = self.T1.pop_lru()
            self.B1.add(k, v)
        elif len(self.T2) > 0:
            k, v = self.T2.pop_lru()
            self.B2.add(k, v)

    def lookup(self, q, true_cid):
        k = self.T1.nearest(q, self.tau)
        if k is not None:                      # T1 hit -> promote to T2
            vec = self.T1._M[self.T1._row_of[k]].copy()
            self.T1.remove(k)
            self.T2.add(k, vec)
            return True, (self.cid[k] != true_cid and true_cid >= 0)
        k = self.T2.nearest(q, self.tau)
        if k is not None:                      # T2 hit -> refresh
            self.T2.touch(k)
            return True, (self.cid[k] != true_cid and true_cid >= 0)
        return False, False

    def insert(self, q, cid):
        i = self.next_id
        self.next_id += 1
        gb1 = self._ghost(self.B1, q, i)
        gb2 = self._ghost(self.B2, q, i)

        if gb1 is not None:                    # Case II: recency was right
            delta = max(1.0, len(self.B2) / max(len(self.B1), 1))
            self.p = min(float(self.c), self.p + delta)
            self._replace(in_b2=False)
            self.B1.remove(gb1)
            self.T2.add(i, q)
        elif gb2 is not None:                  # Case III: frequency was right
            delta = max(1.0, len(self.B1) / max(len(self.B2), 1))
            self.p = max(0.0, self.p - delta)
            self._replace(in_b2=True)
            self.B2.remove(gb2)
            self.T2.add(i, q)
        else:                                  # Case IV: true miss
            l1 = len(self.T1) + len(self.B1)
            if l1 == self.c:
                if len(self.T1) < self.c:
                    self.B1.pop_lru()
                    self._replace(in_b2=False)
                else:
                    self.T1.pop_lru()
            elif l1 < self.c and l1 + len(self.T2) + len(self.B2) >= self.c:
                if l1 + len(self.T2) + len(self.B2) >= 2 * self.c:
                    self.B2.pop_lru()
                self._replace(in_b2=False)
            self.T1.add(i, q)

        self.cid[i] = cid
        self.p_trace.append(self.p)


def run_arc(queries, cids, cap, tau, ghost_matching="semantic", keep_p=False):
    c = ScreeningARC(cap, tau, queries.shape[1], ghost_matching)
    hits = false_hits = 0
    for q, cid in zip(queries, cids):
        h, fh = c.lookup(q, int(cid))
        if h:
            hits += 1
            false_hits += fh
        else:
            c.insert(q, int(cid))
    n = len(cids)
    out = {
        "hit_rate": hits / n,
        "false_hit_rate": (false_hits / hits if hits else 0.0),
        "final_p": c.p,
        "max_p": max(c.p_trace) if c.p_trace else 0.0,
    }
    if keep_p:
        out["p_trace"] = c.p_trace
    return out


# ==========================================================================
# check 1: frequency sketches
# ==========================================================================
def _stable_hash(obj):
    """Deterministic hash, unlike the builtin.

    Python salts ``hash()`` of ``str`` and ``bytes`` per process
    (``PYTHONHASHSEED``), so a sketch keyed through the builtin gives
    different numbers on every run. That is invisible in the LSH arm, whose
    keys are tuples of ints, and *only* affects the exact-key arm -- which is
    precisely the measurement the gate reports. Hashing the repr with blake2b
    makes both arms reproducible run to run.
    """
    return int.from_bytes(
        hashlib.blake2b(repr(obj).encode(), digest_size=8).digest(), "big"
    )


class CountMinSketch:
    """4-bit saturating Count-Min Sketch with periodic halving (as TinyLFU)."""

    def __init__(self, width=2048, depth=4, seed=0, aging_period=None):
        self.w, self.d = width, depth
        self.C = np.zeros((depth, width), dtype=np.int32)
        rng = np.random.default_rng(seed)
        self.salt = rng.integers(1, 2 ** 31, size=depth)
        self.aging_period = aging_period
        self.n = 0

    def _idx(self, key):
        return [(_stable_hash((key, int(s))) % self.w) for s in self.salt]

    def add(self, key):
        for r, c in enumerate(self._idx(key)):
            if self.C[r, c] < 15:
                self.C[r, c] += 1
        self.n += 1
        if self.aging_period and self.n >= self.aging_period:
            self.C >>= 1
            self.n = 0

    def estimate(self, key):
        return int(min(self.C[r, c] for r, c in enumerate(self._idx(key))))


class LSHKeyer:
    """Banded random-hyperplane LSH; one bucket key per band.

    Defaults are the 12 bits x 16 bands configuration reported in
    ``resources/RESULTS.md``.

    .. note::
       ``resources/tinylfu_test.py`` packs a band with
       ``np.packbits(band, bitorder="little")[0]``, which keeps only the first
       byte and so silently truncates any band wider than 8 bits -- a 12-bit
       band was really an 8-bit band. This packs the whole band.
    """

    def __init__(self, dim, bits_per_band=12, bands=16, seed=0):
        rng = np.random.default_rng(seed)
        self.R = rng.normal(size=(bands * bits_per_band, dim))
        self.b, self.k = bands, bits_per_band
        self._weights = (1 << np.arange(bits_per_band, dtype=np.int64))

    def keys(self, q):
        bits = (self.R @ q > 0).astype(np.int64)
        return [(i, int(bits[i * self.k:(i + 1) * self.k] @ self._weights))
                for i in range(self.b)]


class SemanticSketch:
    """Frequency estimator over embeddings. ``mode`` is 'exact' or 'lsh'."""

    def __init__(self, dim, mode, capacity, seed=0):
        self.mode = mode
        self.cms = CountMinSketch(width=max(2048, 8 * capacity), depth=4,
                                  seed=seed, aging_period=10 * capacity)
        self.lsh = LSHKeyer(dim, seed=seed) if mode == "lsh" else None

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


def sketch_health(queries, cids, mode, cap=200, seed=0, warm=None, probe=1000):
    """Does the sketch's estimate track true cluster popularity?

    Warms the sketch on the first ``warm`` queries, then, on the next ``probe``
    queries, correlates the estimate against how often that query's cluster
    actually appeared during the warm-up.
    """
    warm = warm or max(1, len(queries) - probe)
    sk = SemanticSketch(queries.shape[1], mode, cap, seed=seed)
    for n in range(warm):
        sk.add(queries[n], f"q{n}")
    counts = np.bincount(cids[:warm], minlength=int(cids.max()) + 1)
    est, true = [], []
    for n in range(warm, min(warm + probe, len(queries))):
        est.append(sk.estimate(queries[n], f"q{n}"))
        true.append(counts[cids[n]])
    est, true = np.asarray(est, float), np.asarray(true, float)
    r = float(np.corrcoef(est, true)[0, 1]) if est.std() > 0 else 0.0
    return {"mean_estimate": float(est.mean()), "max_estimate": float(est.max()),
            "corr_with_true_popularity": r}


# ==========================================================================
def main():
    traces.silence_spurious_fp_warnings()
    if not traces.data_available():
        print("benchmarks/data/ is empty -- run prepare_data.py first.")
        return 2

    tau = traces.load_tau()
    n_queries = 20_000
    caps = [100, 200, 400]
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 78)
    say("STEP 3 GATE -- screening replication on real embeddings")
    say("=" * 78)
    say(f"tau = {tau} (chosen by prepare_data.py F1 sweep on Quora clusters)")
    say(f"queries per trace = {n_queries}")
    for name in ("quora", "wildchat"):
        m = traces.load_meta(name)
        say(f"corpus {name:9s}: {m['count']} items, {m['n_clusters']} clusters, "
            f"{m['model']}")

    report = {"tau": tau, "n_queries": n_queries}

    # ---------------- check 1 ----------------
    say("\n" + "-" * 78)
    say("CHECK 1  exact-key frequency sketch carries no signal")
    say("-" * 78)
    q_stat, c_stat = traces.build_trace("quora-stationary", n_queries, seed=1)
    say(f"{'keying':<12} {'mean est':>9} {'max':>6} "
        f"{'corr(est, true cluster popularity)':>36}")
    sketch = {}
    for mode in ("exact", "lsh"):
        h = sketch_health(q_stat, c_stat, mode)
        sketch[mode] = h
        say(f"{mode:<12} {h['mean_estimate']:>9.2f} {h['max_estimate']:>6.0f} "
            f"{h['corr_with_true_popularity']:>36.3f}")
    corr = sketch["exact"]["corr_with_true_popularity"]
    check1 = abs(corr) < SKETCH_CORR_MAX
    say(f"\nsynthetic reference (resources/RESULTS.md): exact mean 0.21, "
        f"corr -0.059 | lsh corr +0.743")
    say(f"PASS CONDITION |corr_exact| < {SKETCH_CORR_MAX}  ->  "
        f"|{corr:.3f}| = {abs(corr):.3f}  ->  {'PASS' if check1 else 'FAIL'}")
    report["check1"] = {"sketch": sketch, "corr_exact": corr,
                        "threshold": SKETCH_CORR_MAX, "passed": bool(check1)}

    # ---------------- check 2 ----------------
    say("\n" + "-" * 78)
    say("CHECK 2  semantic ghosts beat exact ghosts on drifting traffic")
    say("-" * 78)
    report["check2"] = {}
    margins = []
    for regime in ("quora-drift", "wildchat"):
        q, c = traces.build_trace(regime, n_queries, seed=1)
        say(f"\n[{regime}]")
        say(f"{'cap':>5} | {'LRU':>8} | {'LFU':>8} | {'ARC exact':>10} | "
            f"{'ARC semantic':>13} | {'margin':>8} | {'final p (ex/sem)':>18}")
        per_cap = {}
        for cap in caps:
            lru, _ = run_baseline(q, c, cap, tau, "LRU")
            lfu, _ = run_baseline(q, c, cap, tau, "LFU")
            ex = run_arc(q, c, cap, tau, "exact")
            se = run_arc(q, c, cap, tau, "semantic")
            margin = (se["hit_rate"] - ex["hit_rate"]) * 100
            margins.append(margin)
            per_cap[cap] = {"lru": lru, "lfu": lfu,
                            "arc_exact": ex, "arc_semantic": se,
                            "margin_pp": margin}
            say(f"{cap:>5} | {lru*100:7.2f}% | {lfu*100:7.2f}% | "
                f"{ex['hit_rate']*100:9.2f}% | {se['hit_rate']*100:12.2f}% | "
                f"{margin:+7.2f}pp | "
                f"{ex['final_p']:8.1f} /{se['final_p']:8.1f}")
        report["check2"][regime] = per_cap

    worst = min(margins)
    check2 = worst >= GHOST_MARGIN_MIN
    say(f"\nsynthetic reference (resources/RESULTS.md): "
        f"exact 52.63% vs semantic 68.63% (+16.00pp), exact final p = 0")
    say(f"PASS CONDITION every margin >= {GHOST_MARGIN_MIN}pp  ->  "
        f"worst = {worst:+.2f}pp  ->  {'PASS' if check2 else 'FAIL'}")
    report["check2_summary"] = {"worst_margin_pp": worst,
                                "threshold_pp": GHOST_MARGIN_MIN,
                                "passed": bool(check2)}

    # ---------------- verdict ----------------
    say("\n" + "=" * 78)
    ok = check1 and check2
    say(f"CHECK 1 (sketch inert)      : {'PASS' if check1 else 'FAIL'}")
    say(f"CHECK 2 (semantic ghosts)   : {'PASS' if check2 else 'FAIL'}")
    verdict = "PASS -- proceed to implementation" if ok else "FAIL -- STOP"
    say(f"GATE                        : {verdict}")
    say("=" * 78)
    report["gate_passed"] = bool(ok)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "gate_screening.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(RESULTS_DIR, "gate_screening.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {RESULTS_DIR}/gate_screening.{{json,txt}}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
