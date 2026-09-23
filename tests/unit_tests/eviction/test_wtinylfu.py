import unittest

from gptcache.manager.eviction.manager import EvictionBase


class TestWTinyLFU(unittest.TestCase):

    def test_basic_put_get(self):
        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=10, on_evict=lambda x: None
        )
        eviction.put([1, 2, 3])
        self.assertIsNotNone(eviction.get(1))
        self.assertIsNotNone(eviction.get(2))
        self.assertIsNotNone(eviction.get(3))

    def test_policy_property(self):
        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=10, on_evict=lambda x: None
        )
        self.assertEqual(eviction.policy, "WTINYLFU")

    def test_eviction_triggered(self):
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=5, clean_size=1, on_evict=on_evict
        )
        # Insert more than maxsize
        for i in range(10):
            eviction.put([i])
        # Some items should have been evicted
        self.assertGreater(len(evicted), 0)

    def test_frequent_items_retained(self):
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=10, clean_size=1, on_evict=on_evict
        )

        # Insert items 0-9
        eviction.put(list(range(10)))

        # Access items 0-4 many times to boost their frequency
        for _ in range(20):
            for i in range(5):
                eviction.get(i)

        # Now insert items 10-19, forcing eviction
        for i in range(10, 20):
            eviction.put([i])

        # Items 0-4 should mostly survive due to high frequency
        surviving_popular = sum(1 for i in range(5) if i not in evicted)
        self.assertGreaterEqual(surviving_popular, 3)

    def test_cost_aware_retention(self):
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=10, clean_size=1,
            on_evict=on_evict, cost_aware=True
        )

        # Insert items 0-9
        eviction.put(list(range(10)))

        # Set high cost for items 0-4 (expensive to regenerate)
        for i in range(5):
            eviction.set_cost(i, 1000.0)
        # Set low cost for items 5-9
        for i in range(5, 10):
            eviction.set_cost(i, 1.0)

        # Access all items equally
        for _ in range(5):
            for i in range(10):
                eviction.get(i)

        # Now insert items 10-19, forcing eviction
        for i in range(10, 20):
            eviction.set_cost(i, 1.0)
            eviction.put([i])

        # High-cost items (0-4) should be preferentially retained
        high_cost_evicted = sum(1 for i in range(5) if i in evicted)
        low_cost_evicted = sum(1 for i in range(5, 10) if i in evicted)
        self.assertLessEqual(high_cost_evicted, low_cost_evicted)

    def test_one_hit_wonders_evicted(self):
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=10, clean_size=1, on_evict=on_evict
        )

        # Insert items 0-4 and access them heavily
        eviction.put(list(range(5)))
        for _ in range(20):
            for i in range(5):
                eviction.get(i)

        # Insert items 5-9 (no re-access -> one-hit wonders)
        eviction.put(list(range(5, 10)))

        # Insert items 10-14, forcing eviction
        for i in range(10, 15):
            eviction.put([i])

        # One-hit wonders (5-9) should be evicted before frequent items (0-4)
        frequent_evicted = sum(1 for i in range(5) if i in evicted)
        one_hit_evicted = sum(1 for i in range(5, 10) if i in evicted)
        self.assertLessEqual(frequent_evicted, one_hit_evicted)

    def test_clean_size_default(self):
        """Default clean_size should be 20% of maxsize."""
        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=100, on_evict=lambda x: None
        )
        self.assertEqual(eviction._clean_size, 20)

    def test_set_cost(self):
        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=10, on_evict=lambda x: None
        )
        eviction.put([1])
        eviction.set_cost(1, 500.0)
        self.assertEqual(eviction._cost_map[1], 500.0)

    def test_doorkeeper_cleared_on_sketch_reset(self):
        """Per the TinyLFU paper, the doorkeeper must be cleared when the
        CMS counters are halved to prevent stale false positives."""
        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=100, on_evict=lambda x: None,
            reset_multiplier=1,  # sample_size = 100
        )
        # Plant a sentinel in the doorkeeper before any resets
        sentinel = hash(0xDEADBEEF)
        eviction._doorkeeper.add(sentinel)
        self.assertTrue(eviction._doorkeeper.contains(sentinel))

        # Drive sketch increments past sample_size to trigger reset+clear.
        # put() seeds the doorkeeper; subsequent get() calls pass through
        # and increment the sketch.  100 items × 5 rounds > sample_size.
        eviction.put(list(range(50)))
        for _ in range(5):
            for i in range(50):
                eviction.get(i)

        # The sentinel was planted before any reset.  After clear it must
        # be gone (bloom filter has ~958 bits with ≤50 items → FP < 0.01%).
        self.assertFalse(eviction._doorkeeper.contains(sentinel),
                         "Doorkeeper should be cleared when CMS resets")

    def test_matches_existing_test_pattern_lru_style(self):
        """Mirrors the existing test_lru pattern to verify compatibility."""
        datas = []

        def on_evict(deletes):
            for delete in deletes:
                if delete in datas:
                    datas.remove(delete)

        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=4, clean_size=2, on_evict=on_evict
        )

        def add_data(data):
            datas.append(data)
            eviction.put([data])

        add_data(1)
        add_data(2)
        add_data(3)
        add_data(4)
        # Access item 1 many times to boost frequency
        for _ in range(10):
            eviction.get(1)
        add_data(5)
        # After eviction, item 1 should survive (highest frequency)
        self.assertIn(1, datas)


if __name__ == "__main__":
    unittest.main()
