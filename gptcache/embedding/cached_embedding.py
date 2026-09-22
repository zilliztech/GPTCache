from typing import Any, Optional

from cachetools import LRUCache

from gptcache.embedding.base import BaseEmbedding


class CachedEmbedding(BaseEmbedding):
    """Wraps any :class:`BaseEmbedding` with an in-memory LRU cache keyed on the
    exact input text, so that repeated calls with the same text skip the
    (relatively expensive) model forward pass.

    This targets a specific, narrow inefficiency: whenever the *exact same*
    string is embedded more than once during a process's lifetime (e.g. a
    user re-sends an identical prompt, or a benchmark replay revisits an
    entry), the second+ call is served from memory instead of re-running the
    encoder.

    It intentionally does **not** attempt any notion of semantic/near-duplicate
    matching -- that is already the job of the vector index. This cache only
    ever returns a hit for a byte-for-byte identical input, so it cannot
    change *which* answer is served, only how fast an identical embedding is
    produced.

    :param embedding: the underlying embedding backend to wrap.
    :type embedding: BaseEmbedding
    :param cache_size: maximum number of distinct texts to keep cached.
    :type cache_size: int

    Example:
        .. code-block:: python

            from gptcache.embedding import SBERT, CachedEmbedding

            base = SBERT("all-MiniLM-L6-v2")
            encoder = CachedEmbedding(base, cache_size=10_000)
            embed = encoder.to_embeddings("Hello, world.")
            # second call with identical text is served from cache
            embed_again = encoder.to_embeddings("Hello, world.")
    """

    def __init__(self, embedding: BaseEmbedding, cache_size: int = 10_000):
        if cache_size <= 0:
            raise ValueError(f"cache_size must be positive, got {cache_size}")
        self._embedding = embedding
        self._cache: "LRUCache[Any, Any]" = LRUCache(maxsize=cache_size)
        self._hits = 0
        self._misses = 0

    def to_embeddings(self, data, **kwargs):
        """Generate (or retrieve from cache) the embedding for ``data``.

        Only single, hashable string inputs are cached; any other input
        shape (e.g. a list, or an unhashable object) is passed straight
        through to the underlying embedding without touching the cache, so
        behavior for non-text embeddings (image/audio backends etc.) is
        unaffected.
        """
        key = self._cache_key(data, kwargs)
        if key is not None:
            cached = self._cache.get(key)
            if cached is not None:
                self._hits += 1
                return cached.copy()

        result = self._embedding.to_embeddings(data, **kwargs)

        if key is not None:
            self._misses += 1
            self._cache[key] = result.copy()

        return result

    @staticmethod
    def _cache_key(data, kwargs) -> Optional[tuple]:
        """Return a hashable cache key for ``data``, or None if ``data``
        (or any kwarg) isn't safely cacheable."""
        if not isinstance(data, str):
            return None
        try:
            hash(data)
            frozen_kwargs = tuple(sorted(kwargs.items()))
            hash(frozen_kwargs)
        except TypeError:
            return None
        return (data, frozen_kwargs)

    @property
    def dimension(self) -> int:
        return self._embedding.dimension

    @property
    def stats(self) -> dict:
        """Cache statistics: hits, misses, hit_rate, and current size."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / total) if total else 0.0,
            "size": len(self._cache),
            "maxsize": self._cache.maxsize,
        }

    def clear(self) -> None:
        """Empty the cache and reset hit/miss counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0