import unittest

from gptcache.manager.eviction.count_min_sketch import CountMinSketch


class TestCountMinSketch(unittest.TestCase):

    def test_increment_and_estimate(self):
        cms = CountMinSketch(capacity=100)
        for _ in range(5):
            cms.increment(42)
        self.assertEqual(cms.estimate(42), 5)

    def test_estimate_returns_minimum(self):
        """Estimate should be the min across all 4 rows (no overcount for the key itself)."""
        cms = CountMinSketch(capacity=1000)
        for _ in range(10):
            cms.increment(100)
        est = cms.estimate(100)
        self.assertEqual(est, 10)

    def test_unseen_key_returns_zero(self):
        cms = CountMinSketch(capacity=100)
        cms.increment(1)
        self.assertEqual(cms.estimate(999), 0)

    def test_counter_cap_at_15(self):
        cms = CountMinSketch(capacity=100, sample_size_multiplier=10000)
        for _ in range(50):
            cms.increment(7)
        self.assertLessEqual(cms.estimate(7), 15)

    def test_reset_halves_counters(self):
        cms = CountMinSketch(capacity=100, sample_size_multiplier=10000)
        for _ in range(10):
            cms.increment(55)
        before = cms.estimate(55)
        cms.reset()
        after = cms.estimate(55)
        self.assertEqual(after, before // 2)

    def test_additions_tracked(self):
        """CMS tracks additions so the caller can decide when to reset."""
        capacity = 16
        cms = CountMinSketch(capacity=capacity, sample_size_multiplier=2)
        for i in range(10):
            cms.increment(i)
        self.assertEqual(cms.additions, 10)
        self.assertEqual(cms._sample_size, capacity * 2)

    def test_multiple_keys(self):
        cms = CountMinSketch(capacity=1000)
        for _ in range(8):
            cms.increment(1)
        for _ in range(3):
            cms.increment(2)
        self.assertGreater(cms.estimate(1), cms.estimate(2))


if __name__ == "__main__":
    unittest.main()
