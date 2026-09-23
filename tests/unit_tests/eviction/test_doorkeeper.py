import unittest

from gptcache.manager.eviction.doorkeeper import Doorkeeper


class TestDoorkeeper(unittest.TestCase):

    def test_first_access_rejected(self):
        dk = Doorkeeper(capacity=1000)
        self.assertFalse(dk.allow(42))

    def test_second_access_allowed(self):
        dk = Doorkeeper(capacity=1000)
        dk.allow(42)  # first access
        self.assertTrue(dk.allow(42))  # second access

    def test_contains_without_add(self):
        dk = Doorkeeper(capacity=1000)
        self.assertFalse(dk.contains(99))
        dk.add(99)
        self.assertTrue(dk.contains(99))

    def test_clear_resets(self):
        dk = Doorkeeper(capacity=1000)
        dk.add(42)
        dk.add(99)
        dk.clear()
        self.assertFalse(dk.contains(42))
        self.assertFalse(dk.contains(99))

    def test_false_positive_rate(self):
        dk = Doorkeeper(capacity=10000, fp_rate=0.01)
        # Insert 5000 items
        for i in range(5000):
            dk.add(i)
        # Check 5000 items that were NOT inserted
        false_positives = sum(1 for i in range(5000, 10000) if dk.contains(i))
        fp_rate = false_positives / 5000
        # Allow some slack: target is 1%, accept up to 3%
        self.assertLess(fp_rate, 0.03, f"FP rate {fp_rate:.3f} too high")

    def test_many_distinct_keys(self):
        dk = Doorkeeper(capacity=100)
        for i in range(100):
            dk.add(i)
        # All inserted keys should be found
        for i in range(100):
            self.assertTrue(dk.contains(i))


if __name__ == "__main__":
    unittest.main()
