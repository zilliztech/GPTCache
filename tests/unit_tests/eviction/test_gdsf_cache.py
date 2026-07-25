import unittest

from gptcache.manager.eviction.gdsf import GDSFCache
from gptcache.manager.eviction.memory_cache import MemoryCacheEviction


class TestGDSFCache(unittest.TestCase):
    def test_evicts_cheapest_when_frequencies_equal(self):
        cache = GDSFCache(maxsize=3)
        cache["cheap"] = 10.0
        cache["mid"] = 100.0
        cache["expensive"] = 1000.0
        key, value = cache.popitem()
        self.assertEqual(key, "cheap")
        self.assertEqual(value, 10.0)

    def test_frequency_beats_cost(self):
        # a cheap entry accessed often should outrank a costly one-hit entry
        cache = GDSFCache(maxsize=3)
        cache["cheap"] = 10.0
        cache["expensive"] = 30.0
        for _ in range(5):
            _ = cache["cheap"]  # freq 6 -> priority 60 > 30
        key, _ = cache.popitem()
        self.assertEqual(key, "expensive")

    def test_eviction_on_overflow(self):
        cache = GDSFCache(maxsize=2)
        cache["a"] = 10.0
        cache["b"] = 20.0
        cache["c"] = 30.0  # overflow: "a" (lowest priority) must go
        self.assertNotIn("a", cache)
        self.assertIn("b", cache)
        self.assertIn("c", cache)

    def test_inflation_ages_out_stale_entries(self):
        cache = GDSFCache(maxsize=2)
        cache["old_hot"] = 10.0
        for _ in range(9):
            _ = cache["old_hot"]  # freq 10 -> priority 100
        cache["a"] = 30.0
        cache["b"] = 50.0  # evicts "a" (prio 30), inflation L becomes 30
        self.assertEqual(cache.inflation, 30.0)
        # each eviction raises L, so newly inserted entries start ever
        # higher and "old_hot" (stuck at 100) is eventually overtaken
        cache["c"] = 80.0  # evicts "b" (prio 80), L becomes 80
        self.assertEqual(cache.inflation, 80.0)
        cache["d"] = 30.0  # now old_hot (100) is the minimum -> evicted
        self.assertNotIn("old_hot", cache)
        self.assertIn("c", cache)
        self.assertIn("d", cache)

    def test_true_value_means_cost_one(self):
        # GPTCache stores True when no cost is given -> behaves like LFU
        cache = GDSFCache(maxsize=3)
        cache["a"] = True
        cache["b"] = True
        _ = cache["a"]
        key, _ = cache.popitem()
        self.assertEqual(key, "b")

    def test_update_existing_key_keeps_frequency(self):
        cache = GDSFCache(maxsize=3)
        cache["a"] = 10.0
        _ = cache["a"]  # freq 2
        cache["a"] = 20.0  # cost update, freq stays 2 -> priority 40
        cache["b"] = 30.0  # freq 1 -> priority 30
        key, _ = cache.popitem()
        self.assertEqual(key, "b")

    def test_popitem_on_empty_cache_raises(self):
        cache = GDSFCache(maxsize=2)
        with self.assertRaises(KeyError):
            cache.popitem()

    def test_get_missing_key_raises(self):
        cache = GDSFCache(maxsize=2)
        with self.assertRaises(KeyError):
            _ = cache["missing"]


class TestMemoryCacheEvictionGDSF(unittest.TestCase):
    def test_gdsf_policy_via_eviction_base(self):
        evicted = []
        eviction = MemoryCacheEviction(
            policy="GDSF", maxsize=3, clean_size=1, on_evict=evicted.extend
        )
        eviction.put(["cheap", "mid", "expensive"], costs=[10.0, 100.0, 1000.0])
        eviction.put(["new"], costs=[500.0])  # overflow -> "cheap" evicted
        self.assertEqual(evicted, ["cheap"])
        self.assertIsNone(eviction.get("cheap"))
        self.assertIsNotNone(eviction.get("expensive"))
        self.assertEqual(eviction.policy, "GDSF")

    def test_put_without_costs_is_backward_compatible(self):
        eviction = MemoryCacheEviction(
            policy="GDSF", maxsize=2, clean_size=1, on_evict=lambda keys: None
        )
        eviction.put(["a", "b"])  # no costs -> all cost 1.0, LFU-like
        self.assertIsNotNone(eviction.get("a"))
        eviction.put(["c"])
        # "b" was never accessed, so it is the one evicted
        self.assertIsNone(eviction.get("b"))
        self.assertIsNotNone(eviction.get("c"))

    def test_vanilla_policies_ignore_costs(self):
        eviction = MemoryCacheEviction(
            policy="LRU", maxsize=2, clean_size=1, on_evict=lambda keys: None
        )
        eviction.put(["a", "b"], costs=[10.0, 20.0])  # must not raise
        self.assertIsNotNone(eviction.get("a"))


if __name__ == "__main__":
    unittest.main()
