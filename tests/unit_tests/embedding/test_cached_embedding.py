import numpy as np
import pytest

from gptcache.embedding.base import BaseEmbedding
from gptcache.embedding.cached_embedding import CachedEmbedding


class _CountingFakeEmbedding(BaseEmbedding):
    """Deterministic fake encoder that counts how many times it was actually
    invoked, so tests can assert the wrapper really skipped recomputation."""

    def __init__(self, dim: int = 8):
        self._dim = dim
        self.call_count = 0

    def to_embeddings(self, data, **_):
        self.call_count += 1
        # Deterministic "embedding": hash of the text broadcast into a vector,
        # so equal inputs would produce equal outputs even without caching --
        # the tests check call_count, not just output equality, to make sure
        # the cache is actually short-circuiting the call.
        key = tuple(data) if isinstance(data, list) else data
        seed = abs(hash(key)) % (2 ** 32)
        rng = np.random.default_rng(seed)
        return rng.random(self._dim).astype("float32")

    @property
    def dimension(self) -> int:
        return self._dim


def test_repeated_text_hits_cache_and_skips_recompute():
    fake = _CountingFakeEmbedding()
    cached = CachedEmbedding(fake, cache_size=10)

    first = cached.to_embeddings("hello world")
    second = cached.to_embeddings("hello world")

    assert fake.call_count == 1  # second call served from cache
    assert np.array_equal(first, second)
    assert cached.stats["hits"] == 1
    assert cached.stats["misses"] == 1


def test_distinct_texts_each_miss():
    fake = _CountingFakeEmbedding()
    cached = CachedEmbedding(fake, cache_size=10)

    cached.to_embeddings("alpha")
    cached.to_embeddings("beta")
    cached.to_embeddings("gamma")

    assert fake.call_count == 3
    assert cached.stats["hits"] == 0
    assert cached.stats["misses"] == 3


def test_cache_respects_maxsize_eviction():
    fake = _CountingFakeEmbedding()
    cached = CachedEmbedding(fake, cache_size=2)

    cached.to_embeddings("a")
    cached.to_embeddings("b")
    cached.to_embeddings("c")  # evicts "a" (LRU)
    cached.to_embeddings("a")  # must recompute, was evicted

    assert fake.call_count == 4
    assert cached.stats["size"] == 2


def test_returned_array_is_a_copy_not_a_shared_reference():
    """Mutating a returned embedding must never corrupt the cached value."""
    fake = _CountingFakeEmbedding()
    cached = CachedEmbedding(fake, cache_size=10)

    first = cached.to_embeddings("hello")
    first[0] = 999.0
    second = cached.to_embeddings("hello")

    assert second[0] != 999.0
    assert fake.call_count == 1


def test_non_string_input_bypasses_cache_safely():
    """List input (batch mode) isn't cached, but must still work correctly."""
    fake = _CountingFakeEmbedding()
    cached = CachedEmbedding(fake, cache_size=10)

    cached.to_embeddings(["a", "b"])
    cached.to_embeddings(["a", "b"])

    # both calls went straight through -- no crash, no false cache hit
    assert fake.call_count == 2
    assert cached.stats["hits"] == 0
    assert cached.stats["misses"] == 0


def test_dimension_delegates_to_wrapped_embedding():
    fake = _CountingFakeEmbedding(dim=16)
    cached = CachedEmbedding(fake, cache_size=10)
    assert cached.dimension == 16


def test_clear_resets_cache_and_stats():
    fake = _CountingFakeEmbedding()
    cached = CachedEmbedding(fake, cache_size=10)

    cached.to_embeddings("x")
    cached.to_embeddings("x")
    cached.clear()

    assert cached.stats == {"hits": 0, "misses": 0, "hit_rate": 0.0, "size": 0, "maxsize": 10}

    cached.to_embeddings("x")
    assert fake.call_count == 2  # recomputed after clear


def test_invalid_cache_size_raises():
    fake = _CountingFakeEmbedding()
    with pytest.raises(ValueError):
        CachedEmbedding(fake, cache_size=0)