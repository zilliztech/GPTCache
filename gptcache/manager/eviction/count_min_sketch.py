"""4-bit packed Count-Min Sketch for frequency estimation.

Uses the same design as Caffeine/Theine: 4 hash functions, 4-bit counters
packed 16 per uint64 word, periodic halving for aging.
"""

import numpy as np


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


def _rehash(h: int) -> int:
    h = (h ^ (h >> 32)) & 0xFFFFFFFFFFFFFFFF
    h = (h * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    h = (h ^ (h >> 32)) & 0xFFFFFFFFFFFFFFFF
    return h


_RESET_MASK = np.uint64(0x7777777777777777)
_MAX_COUNT = 15


class CountMinSketch:
    """4-bit packed Count-Min Sketch with 4 hash functions.

    Each counter is 4 bits (max value 15). 16 counters are packed into
    one uint64 word. The sketch uses 4 independent hash functions derived
    via iterative rehashing.

    :param capacity: expected max number of tracked items (determines width)
    :param width_multiplier: width = next_power_of_2(capacity * multiplier)
    :param sample_size_multiplier: reset after this * capacity increments
    """

    def __init__(
        self,
        capacity: int,
        width_multiplier: int = 1,
        sample_size_multiplier: int = 10,
    ):
        self._width = _next_power_of_2(max(capacity * width_multiplier, 16))
        self._mask = self._width - 1
        # 4 rows, each row has width counters, packed 16 per uint64
        words_per_row = max(self._width // 16, 1)
        self._table = np.zeros(4 * words_per_row, dtype=np.uint64)
        self._words_per_row = words_per_row
        self._additions = 0
        self._sample_size = max(capacity * sample_size_multiplier, 16)

    def increment(self, key_hash: int) -> bool:
        """Increment counters for the given hash. Returns True if any counter changed."""
        h0 = _rehash(key_hash)
        h1 = _rehash(h0)
        h2 = _rehash(h1)
        h3 = _rehash(h2)

        added = self._inc_counter(0, h0 & self._mask)
        added |= self._inc_counter(1, h1 & self._mask)
        added |= self._inc_counter(2, h2 & self._mask)
        added |= self._inc_counter(3, h3 & self._mask)

        if added:
            self._additions += 1

        return added

    def estimate(self, key_hash: int) -> int:
        """Return the estimated frequency (minimum across all rows)."""
        h0 = _rehash(key_hash)
        h1 = _rehash(h0)
        h2 = _rehash(h1)
        h3 = _rehash(h2)

        c0 = self._read_counter(0, h0 & self._mask)
        c1 = self._read_counter(1, h1 & self._mask)
        c2 = self._read_counter(2, h2 & self._mask)
        c3 = self._read_counter(3, h3 & self._mask)

        return min(c0, c1, c2, c3)

    def reset(self):
        """Halve all counters (aging / decay)."""
        self._table = (self._table >> np.uint64(1)) & _RESET_MASK
        self._additions = self._additions // 2

    def _inc_counter(self, row: int, index: int) -> bool:
        word_idx = row * self._words_per_row + index // 16
        nibble_pos = np.uint64((index % 16) * 4)
        current = int((self._table[word_idx] >> nibble_pos) & np.uint64(0xF))
        if current < _MAX_COUNT:
            self._table[word_idx] += np.uint64(1) << nibble_pos
            return True
        return False

    def _read_counter(self, row: int, index: int) -> int:
        word_idx = row * self._words_per_row + index // 16
        nibble_pos = np.uint64((index % 16) * 4)
        return int((self._table[word_idx] >> nibble_pos) & np.uint64(0xF))

    @property
    def additions(self) -> int:
        return self._additions
