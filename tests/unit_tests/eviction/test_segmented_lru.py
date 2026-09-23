import unittest

from gptcache.manager.eviction.segmented_lru import SegmentedLRU


class TestSegmentedLRU(unittest.TestCase):

    def test_insert_into_probation(self):
        slru = SegmentedLRU(probation_capacity=5, protected_capacity=10)
        slru.put("a", 1)
        slru.put("b", 2)
        self.assertEqual(slru.probation_size, 2)
        self.assertEqual(slru.protected_size, 0)

    def test_promote_on_hit(self):
        slru = SegmentedLRU(probation_capacity=5, protected_capacity=10)
        slru.put("a", 1)
        self.assertEqual(slru.probation_size, 1)
        # Hit in probation -> promote to protected
        slru.get("a")
        self.assertEqual(slru.probation_size, 0)
        self.assertEqual(slru.protected_size, 1)

    def test_demote_on_protected_overflow(self):
        slru = SegmentedLRU(probation_capacity=5, protected_capacity=2)
        # Fill probation with 3 items and promote all to protected
        slru.put("a")
        slru.put("b")
        slru.put("c")
        slru.get("a")  # promote a
        slru.get("b")  # promote b
        slru.get("c")  # promote c -> protected overflows, "a" demoted
        self.assertEqual(slru.protected_size, 2)
        # "a" was demoted back to probation
        self.assertIn("a", slru)
        self.assertEqual(len(slru), 3)

    def test_evict_from_probation(self):
        slru = SegmentedLRU(probation_capacity=3, protected_capacity=5)
        slru.put("a")
        slru.put("b")
        slru.put("c")
        victim = slru.evict()
        self.assertEqual(victim[0], "a")  # LRU item
        self.assertEqual(len(slru), 2)

    def test_peek_victim(self):
        slru = SegmentedLRU(probation_capacity=3, protected_capacity=5)
        slru.put("x")
        slru.put("y")
        victim = slru.peek_victim()
        self.assertEqual(victim, "x")
        self.assertEqual(len(slru), 2)  # peek doesn't remove

    def test_remove(self):
        slru = SegmentedLRU(probation_capacity=5, protected_capacity=5)
        slru.put("a")
        slru.get("a")  # promote to protected
        slru.put("b")  # in probation
        self.assertTrue(slru.remove("a"))
        self.assertTrue(slru.remove("b"))
        self.assertEqual(len(slru), 0)

    def test_contains(self):
        slru = SegmentedLRU(probation_capacity=5, protected_capacity=5)
        slru.put("a")
        self.assertIn("a", slru)
        self.assertNotIn("b", slru)

    def test_get_miss_returns_none(self):
        slru = SegmentedLRU(probation_capacity=5, protected_capacity=5)
        self.assertIsNone(slru.get("nonexistent"))

    def test_protected_hit_refreshes(self):
        slru = SegmentedLRU(probation_capacity=5, protected_capacity=3)
        slru.put("a")
        slru.put("b")
        slru.put("c")
        slru.get("a")  # promote
        slru.get("b")  # promote
        slru.get("c")  # promote -> "a" demoted
        # Now access "b" in protected to refresh it
        slru.get("b")
        # "b" should be MRU in protected, safe from demotion


if __name__ == "__main__":
    unittest.main()
