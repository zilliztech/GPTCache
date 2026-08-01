"""Tests for ARCCache -- ARC with semantic ghost lists.

The invariant fuzz test is the load-bearing one: it asserts every structural
invariant from Megiddo & Modha after *every single operation*, which is what
actually establishes that ARC is implemented correctly rather than merely
producing plausible hit rates.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.manager.eviction.arc import ARCCache, _VecList
from gptcache.manager.eviction.manager import EvictionBase

DIM = 32
TAU = 0.8


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def unit(rng, dim=DIM):
    v = rng.normal(size=dim).astype("float32")
    return v / np.linalg.norm(v)


def paraphrase_trace(n_queries, n_clusters, seed=0, dim=DIM, spread=0.07,
                     zipf_s=1.1, epochs=1):
    """Zipf-popular clusters of tight paraphrases.

    ``spread`` is tuned so intra-cluster cosine sits near 0.93 (above ``TAU``)
    while unrelated clusters sit near 0 -- the same separation the real Quora
    embeddings show, without needing the downloaded corpus.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    probs = 1.0 / np.arange(1, n_clusters + 1) ** zipf_s
    probs /= probs.sum()
    per = int(np.ceil(n_queries / epochs))
    cids = []
    for _ in range(epochs):
        ranking = rng.permutation(n_clusters)
        cids.append(ranking[rng.choice(n_clusters, size=per, p=probs)])
    cids = np.concatenate(cids)[:n_queries]

    q = centers[cids] + spread * rng.normal(size=(len(cids), dim))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.astype("float32"), cids


class Harness:
    """Drives an ARCCache exactly the way SSDataManager does.

    GPTCache resolves the resident match in its vector store and only then
    calls ``get(id)``; a miss becomes ``put([id], embeddings=[vec])``. The
    policy never performs resident lookup itself, so the test has to supply
    that half.
    """

    def __init__(self, cache: ARCCache, tau=TAU):
        self.cache = cache
        self.tau = tau
        self.vec = {}
        self.next_id = 0
        self.evicted = []
        self.p_trace = []

    def on_evict(self, ids):
        for i in ids:
            self.evicted.append(i)
            self.vec.pop(i, None)

    def query(self, q):
        """:returns: True on a cache hit."""
        ids = self.cache.resident_ids()
        hit = False
        if ids:
            sims = np.stack([self.vec[i] for i in ids]) @ q
            j = int(sims.argmax())
            if sims[j] >= self.tau:
                self.cache.get(ids[j])
                hit = True
        if not hit:
            i = self.next_id
            self.next_id += 1
            self.vec[i] = q
            self.cache.put([i], embeddings=[q])
        self.p_trace.append(self.cache.p)
        return hit

    def run(self, queries):
        return sum(self.query(q) for q in queries)

    @property
    def max_p(self):
        return max(self.p_trace) if self.p_trace else 0.0


def build(maxsize, ghost_matching="semantic", tau=TAU):
    h = Harness.__new__(Harness)
    cache = ARCCache(maxsize=maxsize, tau=tau, ghost_matching=ghost_matching,
                     on_evict=lambda ids: h.on_evict(ids), dim=DIM)
    Harness.__init__(h, cache, tau)
    return h


# --------------------------------------------------------------------------
class TestARCInvariants(unittest.TestCase):
    """The structural correctness of the ARC implementation."""

    def test_invariant_fuzz(self):
        """5000 random operations; all invariants hold after every one.

        This is the single test that proves ARC is implemented correctly.
        """
        rng = np.random.default_rng(7)
        c = 50
        h = build(c)
        for step in range(5000):
            if h.cache.resident_ids() and rng.random() < 0.4:
                # exercise Case I on a resident id, and the miss path otherwise
                ids = h.cache.resident_ids()
                h.cache.get(ids[rng.integers(len(ids))])
            else:
                h.query(unit(rng))
            try:
                h.cache.check_invariants()
            except AssertionError as exc:  # pragma: no cover - failure path
                self.fail(f"invariant broken at step {step}: {exc}")

    def test_invariant_fuzz_on_paraphrases(self):
        """Same, but on a trace that actually fires the ghost lists."""
        q, _ = paraphrase_trace(4000, n_clusters=120, seed=3, epochs=4)
        h = build(40)
        for step, vec in enumerate(q):
            h.query(vec)
            try:
                h.cache.check_invariants()
            except AssertionError as exc:  # pragma: no cover
                self.fail(f"invariant broken at step {step}: {exc}")
        self.assertGreater(h.cache.sizes["B1"] + h.cache.sizes["B2"], 0,
                           "ghosts never formed, so this test proved nothing")

    def test_capacity_never_exceeded(self):
        rng = np.random.default_rng(1)
        for c in (1, 2, 7, 64):
            h = build(c)
            for _ in range(400):
                h.query(unit(rng))
                self.assertLessEqual(len(h.cache.resident_ids()), c)
            self.assertEqual(len(h.cache.resident_ids()), c)

    def test_p_stays_in_range_adversarial(self):
        """p must never leave [0, c], including when B1/B2 are lopsided."""
        c = 16
        h = build(c)
        # alternate between a tight recurring cluster and a stream of novelty,
        # which drives ghost hits into B1 and B2 unevenly
        hot, _ = paraphrase_trace(1, n_clusters=1, seed=11)
        rng = np.random.default_rng(11)
        for i in range(3000):
            h.query(hot[0] if i % 3 == 0 else unit(rng))
            self.assertGreaterEqual(h.cache.p, 0.0)
            self.assertLessEqual(h.cache.p, c)

    def test_no_unbounded_vector_growth(self):
        """Ghost embeddings are freed; live vectors stay bounded by 2c."""
        rng = np.random.default_rng(5)
        c = 32
        h = build(c)
        for _ in range(10_000):
            h.query(unit(rng))
            self.assertLessEqual(h.cache.n_vectors, 2 * c)
        self.assertLessEqual(h.cache.n_vectors, 2 * c)


class TestARCEviction(unittest.TestCase):
    """Eviction accounting: what leaves, when, and how often it is announced."""

    def test_cold_start_evicts_nothing(self):
        rng = np.random.default_rng(2)
        c = 20
        h = build(c)
        for _ in range(c):
            h.query(unit(rng))
        self.assertEqual(h.evicted, [])
        self.assertEqual(len(h.cache.resident_ids()), c)

    def test_on_evict_fires_exactly_once_per_id(self):
        """No id is announced twice -- in particular not again when its ghost
        is dropped, which is a demotion that was already counted."""
        q, _ = paraphrase_trace(5000, n_clusters=150, seed=9, epochs=5)
        h = build(30)
        h.run(q)
        self.assertEqual(len(h.evicted), len(set(h.evicted)),
                         "an id was announced as evicted more than once")
        # every announced id really is gone from the resident set
        resident = set(h.cache.resident_ids())
        self.assertEqual(resident & set(h.evicted), set())
        # conservation: admitted == resident + evicted
        self.assertEqual(h.next_id, len(resident) + len(h.evicted))

    def test_ghost_demotion_is_announced_when_it_happens(self):
        """REPLACE moves an entry to a ghost list: the response is gone, so the
        eviction is announced then, not later."""
        q, _ = paraphrase_trace(2000, n_clusters=60, seed=13, epochs=3)
        h = build(20)
        h.run(q)
        sizes = h.cache.sizes
        self.assertGreater(sizes["B1"] + sizes["B2"], 0)
        # ids sitting on a ghost list have already been announced
        ghosts = set(h.cache._b1) | set(h.cache._b2)
        self.assertTrue(ghosts.issubset(set(h.evicted)))


class TestSemanticGhostAblation(unittest.TestCase):
    """The contribution, encoded as a regression test.

    Same trace, one flag. ``exact`` reproduces classic ARC, whose ghosts key on
    identity -- and since every miss mints a fresh id, they can never fire.
    """

    TRACE = paraphrase_trace(6000, n_clusters=200, seed=21, epochs=6)

    def test_p_moves_under_semantic_matching(self):
        """The adaptation machinery is live: p leaves 0 during the run.

        Note this asserts on the *trajectory*, not the final value. ARC's step
        sizes are asymmetric -- with |B1| >> |B2| a Case III hit subtracts
        |B1|/|B2| while a Case II hit adds only |B2|/|B1| -- so p is perfectly
        entitled to end a trace back at 0 having ranged widely in between.
        Asserting on the final value alone would be testing the trace, not the
        policy.
        """
        q, _ = self.TRACE
        h = build(50, ghost_matching="semantic")
        h.run(q)
        self.assertGreater(h.max_p, 0.0,
                           "semantic ghosts never fired: p stayed at 0")
        self.assertGreater(len(set(h.p_trace)), 1, "p took only one value")

    def test_p_stays_zero_under_exact_matching(self):
        """The ablation. Ghosts are *populated* but can never fire.

        This is the sharp form of the claim: it is not that exact-key ghosts
        are a worse heuristic, it is that the machinery is inert. B1 fills up
        with evicted entries and p never once moves off 0, because every miss
        mints a fresh id and an id-keyed ghost can only match an id that by
        construction never comes back.
        """
        q, _ = self.TRACE
        h = build(50, ghost_matching="exact")
        h.run(q)
        self.assertEqual(set(h.p_trace), {0.0},
                         "exact-key ghosts cannot fire, so p must stay at 0")
        self.assertGreater(h.cache.sizes["B1"], 0,
                           "ghosts were never even populated, so the test does "
                           "not show what it claims to show")

    def test_semantic_matching_wins_on_drifting_traffic(self):
        q, _ = self.TRACE
        sem = build(50, ghost_matching="semantic")
        exa = build(50, ghost_matching="exact")
        hits_sem = sem.run(q) / len(q)
        hits_exa = exa.run(q) / len(q)
        self.assertGreater(hits_sem, hits_exa + 0.03,
                           f"semantic {hits_sem:.4f} vs exact {hits_exa:.4f}: "
                           f"expected a clear margin")


class TestARCDegenerate(unittest.TestCase):
    """Edge cases and argument validation."""

    def test_capacity_one(self):
        rng = np.random.default_rng(4)
        h = build(1)
        for _ in range(200):
            h.query(unit(rng))
            h.cache.check_invariants()
            self.assertEqual(len(h.cache.resident_ids()), 1)

    def test_get_on_empty_cache(self):
        cache = ARCCache(maxsize=8, tau=TAU)
        self.assertIsNone(cache.get("nothing"))
        self.assertEqual(cache.sizes, {"T1": 0, "T2": 0, "B1": 0, "B2": 0})
        cache.check_invariants()

    def test_get_missing_id(self):
        rng = np.random.default_rng(6)
        h = build(4)
        for _ in range(4):
            h.query(unit(rng))
        self.assertIsNone(h.cache.get("no-such-id"))

    def test_duplicate_id_in_put_is_an_access_not_an_admission(self):
        rng = np.random.default_rng(8)
        cache = ARCCache(maxsize=4, tau=TAU, dim=DIM)
        v = unit(rng)
        cache.put([1], embeddings=[v])
        self.assertEqual(cache.sizes["T1"], 1)
        cache.put([1], embeddings=[v])          # same id again
        cache.check_invariants()
        self.assertEqual(cache.sizes["T1"], 0)  # promoted, not duplicated
        self.assertEqual(cache.sizes["T2"], 1)
        self.assertEqual(cache.resident_ids(), [1])

    def test_put_without_embeddings(self):
        """SSDataManager registers pre-existing rows with no vectors."""
        evicted = []
        cache = ARCCache(maxsize=3, on_evict=evicted.extend)
        cache.put([1, 2, 3, 4, 5])
        cache.check_invariants()
        self.assertEqual(len(cache.resident_ids()), 3)
        self.assertEqual(evicted, [1, 2])

    def test_put_with_none_embedding_never_ghost_matches(self):
        rng = np.random.default_rng(12)
        cache = ARCCache(maxsize=4, tau=TAU, dim=DIM, on_evict=lambda x: None)
        cache.put([1, 2, 3, 4], embeddings=[unit(rng) for _ in range(4)])
        cache.put([5], embeddings=None)     # zero vector: matches nothing
        cache.check_invariants()
        self.assertEqual(cache.p, 0.0)

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            ARCCache(maxsize=0)
        with self.assertRaises(ValueError):
            ARCCache(maxsize=8, ghost_matching="fuzzy")
        with self.assertRaises(ValueError):
            ARCCache(maxsize=8, dim=DIM).put([1, 2], embeddings=[np.zeros(DIM)])
        with self.assertRaises(ValueError):
            ARCCache(maxsize=8, dim=DIM).put([1], embeddings=[np.zeros(DIM + 1)])

    def test_normalises_non_unit_embeddings(self):
        cache = ARCCache(maxsize=4, tau=TAU, dim=DIM, on_evict=lambda x: None)
        base = np.zeros(DIM, dtype="float32")
        base[0] = 5.0                      # deliberately not unit-norm
        cache.put([1], embeddings=[base])
        cache.put([2], embeddings=[base * 3.0])
        cache.check_invariants()

    def test_policy_name(self):
        self.assertEqual(ARCCache(maxsize=4).policy, "ARC")


class TestVecList(unittest.TestCase):
    """The O(1)-removal backing store, checked against the naive definition."""

    def test_nearest_matches_naive_scan(self):
        """Differential test: swap-with-last bookkeeping must not corrupt the
        id/row mapping."""
        rng = np.random.default_rng(31)
        vl = _VecList(DIM, 4)
        ref = {}
        for step in range(600):
            if ref and rng.random() < 0.35:
                victim = list(ref)[rng.integers(len(ref))]
                vl.remove(victim)
                del ref[victim]
            else:
                key = f"k{step}"
                v = unit(rng)
                vl.add(key, v)
                ref[key] = v

            self.assertEqual(len(vl), len(ref))
            q = unit(rng)
            got = vl.nearest(q, tau=-2.0)   # tau=-2 -> always returns the argmax
            if not ref:
                self.assertIsNone(got)
                continue
            keys = list(ref)
            # see the errstate note in arc.py: Accelerate raises spurious flags
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                sims = np.stack([ref[k] for k in keys]) @ q
            want = keys[int(sims.argmax())]
            self.assertEqual(got, want, f"diverged at step {step}")

    def test_nearest_respects_tau(self):
        rng = np.random.default_rng(32)
        vl = _VecList(DIM, 4)
        v = unit(rng)
        vl.add("a", v)
        self.assertEqual(vl.nearest(v, tau=0.99), "a")
        self.assertIsNone(vl.nearest(-v, tau=0.5))

    def test_lru_order_and_touch(self):
        rng = np.random.default_rng(33)
        vl = _VecList(DIM, 4)
        for k in "abcd":
            vl.add(k, unit(rng))
        self.assertEqual(list(vl), ["a", "b", "c", "d"])
        vl.touch("a")
        self.assertEqual(list(vl), ["b", "c", "d", "a"])
        key, _ = vl.pop_lru()
        self.assertEqual(key, "b")


class TestARCThroughGPTCache(unittest.TestCase):
    """End to end, through the public factory."""

    def test_eviction_base_factory(self):
        evicted = []
        eb = EvictionBase.get(name="memory", policy="ARC", maxsize=5,
                              clean_size=2, on_evict=evicted.extend, tau=TAU)
        self.assertEqual(eb.policy, "ARC")
        rng = np.random.default_rng(41)
        for i in range(12):
            eb.put([i], embeddings=[unit(rng)])
        self.assertEqual(eb.get(11), True)
        self.assertIsNone(eb.get(0))
        self.assertEqual(len(evicted), 7)

    def test_data_manager_end_to_end(self):
        with TemporaryDirectory(dir="./") as root:
            db_path = Path(root) / "sqlite.db"
            cache_base = CacheBase("sqlite", sql_url="sqlite:///" + str(db_path))
            # index_path must stay inside the temp dir: the default is a fixed
            # "faiss.index" in the cwd, and closing the manager persists it,
            # which would leave a 32-d index behind for other tests to load
            vector_base = VectorBase("faiss", dimension=DIM,
                                     index_path=str(Path(root) / "faiss.index"))
            data_manager = get_data_manager(
                cache_base, vector_base, max_size=10, clean_size=2,
                eviction="ARC", eviction_params={"tau": TAU},
            )
            rng = np.random.default_rng(43)
            for i in range(30):
                data_manager.save(f"foo{i}", f"receiver the foo {i}", unit(rng))
            # ARC keeps exactly maxsize entries resident, evicting one at a
            # time, unlike the cachetools policies which drop clean_size at once
            self.assertEqual(data_manager.s.count(), 10)
            data_manager.close()

    def test_data_manager_passes_tunables_through(self):
        with TemporaryDirectory(dir="./") as root:
            db_path = Path(root) / "sqlite.db"
            data_manager = get_data_manager(
                CacheBase("sqlite", sql_url="sqlite:///" + str(db_path)),
                VectorBase("faiss", dimension=DIM,
                           index_path=str(Path(root) / "faiss.index")),
                max_size=8, eviction="ARC",
                eviction_params={"tau": 0.42, "ghost_matching": "exact"},
            )
            arc = data_manager.eviction_base._cache
            self.assertEqual(arc.tau, 0.42)
            self.assertEqual(arc.ghost_matching, "exact")
            data_manager.close()


if __name__ == "__main__":
    unittest.main()
