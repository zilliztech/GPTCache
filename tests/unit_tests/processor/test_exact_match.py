"""Tests for the pre-embedding exact-match shortcut."""
import time
import unittest

from gptcache.processor.exact_match import (
    ExactMatchCache,
    normalize_query,
    query_key,
)


class TestNormalize(unittest.TestCase):
    def test_normalize_lower_and_strip(self):
        self.assertEqual(normalize_query("  Hello World\n"), "hello world")

    def test_normalize_nfkc(self):
        # Fullwidth digit "１" -> "1"
        self.assertEqual(normalize_query("topic１"), "topic1")

    def test_normalize_none_and_nonstr(self):
        self.assertEqual(normalize_query(None), "")
        self.assertEqual(normalize_query(42), "42")

    def test_key_stable_across_trivial_variants(self):
        self.assertEqual(
            query_key("What is 2+2?"),
            query_key("  What is 2+2?  "),
        )
        self.assertEqual(
            query_key("What is 2+2?"),
            query_key("what is 2+2?"),
        )

    def test_key_differs_for_real_diffs(self):
        self.assertNotEqual(
            query_key("What is 2+2?"),
            query_key("What is 3+3?"),
        )


class TestExactMatchCacheBasic(unittest.TestCase):
    def test_put_get_round_trip(self):
        c = ExactMatchCache(max_size=10, ttl_seconds=None)
        c.put("hello", "world")
        self.assertEqual(c.get("hello"), "world")
        self.assertEqual(c.hits, 1)
        self.assertEqual(c.misses, 0)

    def test_miss_increments_misses(self):
        c = ExactMatchCache(max_size=10, ttl_seconds=None)
        self.assertIsNone(c.get("never_set"))
        self.assertEqual(c.misses, 1)

    def test_normalization_used_for_lookup(self):
        c = ExactMatchCache(max_size=10, ttl_seconds=None)
        c.put("Hello World", "answer")
        self.assertEqual(c.get("hello world"), "answer")
        self.assertEqual(c.get("  HELLO WORLD\n"), "answer")

    def test_lru_eviction_bounds_size(self):
        c = ExactMatchCache(max_size=2, ttl_seconds=None)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)  # forces eviction of "a"
        self.assertIsNone(c.get("a"))
        self.assertEqual(c.get("b"), 2)
        self.assertEqual(c.get("c"), 3)


class TestExactMatchCoherence(unittest.TestCase):
    """Coherence requirements from Step 4 of docs/memory-speed-plan.md."""

    def test_invalidate_removes_entry(self):
        """Explicit invalidation drops the entry so it can't be served stale."""
        c = ExactMatchCache(max_size=10, ttl_seconds=None)
        c.put("evicted_question", "stale_answer")
        self.assertEqual(c.get("evicted_question"), "stale_answer")

        # Semantic layer evicts the entry; coupling code calls invalidate.
        removed = c.invalidate("evicted_question")
        self.assertTrue(removed)
        # Now the exact-match cache MUST NOT serve the stale answer.
        self.assertIsNone(c.get("evicted_question"))

    def test_invalidate_idempotent(self):
        c = ExactMatchCache(max_size=10, ttl_seconds=None)
        self.assertFalse(c.invalidate("never_existed"))

    def test_invalidate_normalises(self):
        c = ExactMatchCache(max_size=10, ttl_seconds=None)
        c.put("Hello World", "x")
        # Trivial variant of the same query removes the canonical entry.
        self.assertTrue(c.invalidate("  hello world\n"))
        self.assertIsNone(c.get("Hello World"))

    def test_clear_drops_everything(self):
        """clear() lets a rebuild_compact reset the entire shortcut layer."""
        c = ExactMatchCache(max_size=10, ttl_seconds=None)
        for i in range(5):
            c.put(f"q{i}", i)
        self.assertEqual(len(c), 5)
        c.clear()
        self.assertEqual(len(c), 0)
        self.assertIsNone(c.get("q0"))

    def test_ttl_expires_entry(self):
        """TTL provides automatic staleness bounding when explicit invalidation is missed."""
        c = ExactMatchCache(max_size=10, ttl_seconds=0.05)
        c.put("q", "a")
        self.assertEqual(c.get("q"), "a")
        time.sleep(0.08)
        self.assertIsNone(c.get("q"),
                          "TTL must expire entries so an evicted answer cannot be served forever")

    def test_ttl_none_means_no_expiry(self):
        c = ExactMatchCache(max_size=10, ttl_seconds=None)
        c.put("q", "a")
        time.sleep(0.05)
        self.assertEqual(c.get("q"), "a")


if __name__ == "__main__":
    unittest.main()
