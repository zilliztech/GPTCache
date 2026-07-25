import heapq
import itertools

import cachetools


class GDSFCache(cachetools.Cache):
    """Greedy-Dual-Size-Frequency (GDSF) cache.

    Classic cost-aware eviction policy (Cherkasova, 1998), originally used in
    web proxy caches. Each entry gets a priority:

        H(key) = L + frequency(key) * cost(key) / size(key)

    where ``cost`` is how expensive the entry was to produce (for an LLM
    cache: the number of generated tokens) and ``L`` is an inflation value
    that rises to the priority of the last evicted entry. ``L`` acts as an
    aging mechanism: entries that were popular long ago but are not accessed
    anymore eventually fall below newly inserted entries and get evicted.

    The entry's cost is passed as its *value* (``cache[key] = cost``).
    Storing ``True`` (what GPTCache does for cost-unaware policies) makes the
    cost 1.0 for every entry, and the policy degrades to LFU-with-aging.

    Eviction uses a lazy heap: accesses push a new (priority, key) entry and
    stale heap entries are skipped during eviction, so both accesses and
    evictions stay O(log n).
    """

    def __init__(self, maxsize, getsizeof=None):
        cachetools.Cache.__init__(self, maxsize, getsizeof)
        self._heap = []  # (priority, tiebreak, key); stale entries skipped
        self._counter = itertools.count()
        self._freq = {}
        self._prio = {}  # current (valid) priority of each cached key
        self._inflation = 0.0  # the "L" aging value

    @staticmethod
    def _cost_of(value) -> float:
        # bool is checked first because GPTCache stores True as a placeholder
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 1.0
        return float(value)

    def _push(self, key, value):
        size = self.getsizeof(value)
        if not size:
            size = 1
        prio = self._inflation + self._freq[key] * self._cost_of(value) / size
        self._prio[key] = prio
        heapq.heappush(self._heap, (prio, next(self._counter), key))

    def __setitem__(self, key, value):
        cachetools.Cache.__setitem__(self, key, value)
        if key not in self._freq:
            self._freq[key] = 1
        self._push(key, value)

    def __getitem__(self, key):
        value = cachetools.Cache.__getitem__(self, key)
        if key in self._freq:  # a real hit, not __missing__
            self._freq[key] += 1
            self._push(key, value)
        return value

    def __delitem__(self, key):
        cachetools.Cache.__delitem__(self, key)
        self._freq.pop(key, None)
        self._prio.pop(key, None)

    def popitem(self):
        """Evict and return the entry with the lowest priority."""
        while self._heap:
            prio, _, key = heapq.heappop(self._heap)
            if self._prio.get(key) == prio:
                value = cachetools.Cache.__getitem__(self, key)
                self._inflation = prio
                del self[key]
                return key, value
        raise KeyError("cache is empty")

    @property
    def inflation(self) -> float:
        return self._inflation
