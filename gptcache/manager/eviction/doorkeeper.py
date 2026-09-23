"""Bloom filter doorkeeper for TinyLFU admission control.

Filters out one-hit-wonders: only items seen at least twice get their
Count-Min Sketch counters incremented. This prevents long-tail pollution.
"""

import math

import numpy as np


class Doorkeeper:
    """Simple Bloom filter that tracks whether an item has been seen before.

    :param capacity: expected number of insertions before reset
    :param fp_rate: target false positive rate (default 1%)
    """

    def __init__(self, capacity: int = 10000, fp_rate: float = 0.01):
        # Optimal sizing: m = -n*ln(p) / (ln2)^2, k = (m/n)*ln2
        if capacity <= 0:
            capacity = 16
        m = int(-capacity * math.log(fp_rate) / (math.log(2) ** 2))
        m = max(m, 64)
        self._num_bits = m
        self._num_hashes = max(int((m / capacity) * math.log(2)), 1)
        # Bit array stored as uint64 words
        self._words = np.zeros((m + 63) // 64, dtype=np.uint64)

    def allow(self, key_hash: int) -> bool:
        """Check if key was seen before, then add it.

        Returns True if the key was already present (second+ access).
        Always adds the key regardless.
        """
        already_present = self.contains(key_hash)
        self.add(key_hash)
        return already_present

    def contains(self, key_hash: int) -> bool:
        """Check membership without modifying the filter."""
        for i in range(self._num_hashes):
            bit_pos = self._hash_pos(key_hash, i)
            word_idx = bit_pos >> 6  # bit_pos // 64
            bit_idx = np.uint64(bit_pos & 63)
            if not (self._words[word_idx] & (np.uint64(1) << bit_idx)):
                return False
        return True

    def add(self, key_hash: int):
        """Add a key to the filter."""
        for i in range(self._num_hashes):
            bit_pos = self._hash_pos(key_hash, i)
            word_idx = bit_pos >> 6
            bit_idx = np.uint64(bit_pos & 63)
            self._words[word_idx] |= np.uint64(1) << bit_idx

    def clear(self):
        """Reset the filter (remove all entries)."""
        self._words[:] = np.uint64(0)

    def _hash_pos(self, key_hash: int, i: int) -> int:
        # Double hashing: h(i) = (h1 + i*h2) mod m
        h1 = key_hash & 0xFFFFFFFF
        h2 = (key_hash >> 32) & 0xFFFFFFFF
        return ((h1 + i * h2) & 0xFFFFFFFFFFFFFFFF) % self._num_bits
