"""Pre-embedding exact-match shortcut for the GPTCache adapter pipeline.

Real LLM workloads have a heavy exact-repeat tail (canned system prompts,
agent self-talk, frequently asked questions, etc.). Computing an embedding
and running a vector search for those is pure waste - the answer is already
known if we've seen the exact (normalized) query before.

This module provides a small, in-process LRU keyed by ``hash(normalize(query))``
that the adapter consults BEFORE the embedder. On hit, we skip both the
embedding compute (typically 5-30 ms on CPU) and the vector search.

Coherence with semantic eviction
--------------------------------
The semantic layer (FAISS + SQLite) can evict an entry that the
exact-match cache still holds; serving that stale answer would be a
correctness bug. We defend with two mechanisms:

  1. TTL (primary, automatic): every entry has a max age. After expiry it
     is silently ignored, so the longest a stale answer can be served is
     ``ttl_seconds``. Default 300s.

  2. ``invalidate(question)`` (explicit, optional): callers that know
     which questions were evicted can drop those entries directly. The
     ``MemoryCacheEviction`` callback can be wired to this if its
     ``marked_keys`` are mapped back to question text.

For most LLM cache deployments TTL alone is sufficient because semantic
evictions happen on cache-miss writes (a cold path), and a 5-minute
staleness window for repeated identical queries is acceptable. Tighten
``Config.exact_match_ttl_seconds`` if your eviction rate is high.
"""

import hashlib
import threading
import time
import unicodedata
from typing import Any, Optional

from cachetools import LRUCache


def normalize_query(text: str) -> str:
    """Canonicalise a query string before hashing.

    Steps:
      1. NFKC Unicode normalisation (collapse compatible forms)
      2. Strip leading/trailing whitespace
      3. Lowercase

    These are conservative: they preserve internal whitespace and
    punctuation. The goal is to catch trivially-different repeats
    (capitalisation, smart-quote forms, trailing newline), not to do
    real semantic matching - that's what the FAISS layer is for.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.strip().lower()
    return text


def query_key(text: str) -> str:
    """Stable hash of the normalised query, suitable for use as a dict key."""
    n = normalize_query(text)
    return hashlib.blake2s(n.encode("utf-8"), digest_size=16).hexdigest()


class ExactMatchCache:
    """Thread-safe TTL+LRU keyed by normalized-query hash.

    Stores arbitrary "answer" payloads. The value type is intentionally
    opaque - typically a string or a list of strings, whatever the
    adapter is wrapping.

    Operations are O(1) amortised. Expired entries are evicted lazily on
    read (rather than eagerly via a timer) to keep the path off any
    background thread.
    """

    def __init__(self, max_size: int = 10_000, ttl_seconds: Optional[float] = 300.0):
        self._max_size = max(int(max_size), 1)
        self._ttl = float(ttl_seconds) if ttl_seconds is not None else None
        self._lock = threading.Lock()
        self._lru: LRUCache = LRUCache(maxsize=self._max_size)
        # (hits, misses) for testing and lightweight observability
        self.hits = 0
        self.misses = 0

    def get(self, question: str) -> Optional[Any]:
        """Return the cached answer for ``question``, or None on miss/expiry."""
        key = query_key(question)
        with self._lock:
            entry = self._lru.get(key)
            if entry is None:
                self.misses += 1
                return None
            value, ts = entry
            if self._ttl is not None and (time.monotonic() - ts) > self._ttl:
                # Stale - drop and report miss so the caller goes to the
                # semantic layer (which is the source of truth).
                self._lru.pop(key, None)
                self.misses += 1
                return None
            self.hits += 1
            return value

    def put(self, question: str, answer: Any) -> None:
        """Store ``answer`` under the normalized form of ``question``."""
        key = query_key(question)
        with self._lock:
            self._lru[key] = (answer, time.monotonic())

    def invalidate(self, question: str) -> bool:
        """Drop the entry for ``question`` if present. Returns True if removed."""
        key = query_key(question)
        with self._lock:
            return self._lru.pop(key, None) is not None

    def invalidate_by_key(self, key: str) -> bool:
        """Drop the entry for a precomputed key (from ``query_key``)."""
        with self._lock:
            return self._lru.pop(key, None) is not None

    def clear(self) -> None:
        """Drop every entry. Used after index rebuilds."""
        with self._lock:
            self._lru.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._lru)
