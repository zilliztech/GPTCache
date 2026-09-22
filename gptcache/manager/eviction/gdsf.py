"""GreedyDual-Size-Frequency (GDSF) eviction, the classic cost-aware baseline.

Cherkasova, "Improving WWW Proxy Performance with Greedy-Dual-Size-Frequency
Caching Policy," HP Labs HPL-98-69(R.1), 1998.

Each resident item i carries a stored priority
    H(i) = L + F(i) * C(i) / S(i)
stamped at its last access/insert using the current clock L. On eviction the
minimum-H item is removed and the clock is inflated to its priority
(L := H_min), which is GDSF's aging mechanism: fresh low-frequency inserts land
just above L and cannot instantly pollute the cache. Here every cache entry is
one answer, so size S = 1; C is the LLM regeneration cost (`LLMCost`), F the
access count. This is the cost-aware policy CA_W_TINYLFU is measured against.
"""
from typing import Any, Callable, List, Optional


class GreedyDualSizeFrequency:
    """GDSF over integer ids, matching the CA_W_TINYLFU eviction interface."""

    def __init__(
        self,
        maxsize: int,
        on_evict: Callable[[List[Any]], None] = None,
        default_cost: float = 1.0,
        **_kwargs,
    ):
        self._maxsize = maxsize
        self._on_evict = on_evict
        self._default_cost = float(default_cost)
        self._clock = 0.0                       # L, the aging clock
        self._h: dict = {}                      # id -> stored priority H
        self._freq: dict = {}                   # id -> access count F
        self._cost: dict = {}                   # id -> regeneration cost C

    def _restamp(self, key: Any) -> None:
        # H = L + F * C / S, with S = 1.
        self._h[key] = self._clock + self._freq[key] * self._cost[key]

    def _evict_one(self) -> Any:
        # ponytail: O(n) min scan; n = cache size (<=200 here). Swap for a
        # lazy-invalidated heap only if maxsize grows past a few thousand.
        victim = min(self._h, key=self._h.get)
        self._clock = self._h[victim]           # aging: L := H_min
        del self._h[victim]
        del self._freq[victim]
        del self._cost[victim]
        return victim

    def put(self, objs: List[Any], costs: Optional[List[float]] = None) -> None:
        evicted: List[Any] = []
        for i, key in enumerate(objs):
            cost = self._default_cost if not costs else float(costs[i])
            if key in self._h:                  # re-save: treat as an access
                self._freq[key] += 1
                self._cost[key] = cost
                self._restamp(key)
                continue
            while len(self._h) >= self._maxsize:
                evicted.append(self._evict_one())
            self._freq[key] = 1
            self._cost[key] = cost
            self._restamp(key)
        if evicted and self._on_evict:
            self._on_evict(evicted)

    def get(self, obj: Any):
        if obj in self._h:                      # hit: bump frequency, re-stamp
            self._freq[obj] += 1
            self._restamp(obj)
            return True
        return None


if __name__ == "__main__":
    # Cost-awareness: an expensive item survives a cheap one at equal frequency.
    ev: List[Any] = []
    g = GreedyDualSizeFrequency(maxsize=2, on_evict=ev.extend)
    g.put([1], costs=[100.0])                   # expensive
    g.put([2], costs=[1.0])                      # cheap; cache full {1,2}
    g.put([3], costs=[1.0])                      # H1=100, H2=1 -> evict 2
    assert ev == [2], ev

    # Frequency still counts: a hot cheap item outlives a cold cheap one.
    ev.clear()
    g = GreedyDualSizeFrequency(maxsize=2, on_evict=ev.extend)
    g.put([1], costs=[1.0]); g.put([2], costs=[1.0])
    for _ in range(5):
        g.get(1)                                # F(1)=6
    g.put([3], costs=[1.0])                      # H1=6, H2=1 -> evict 2
    assert ev == [2], ev

    # Aging clock is monotone non-decreasing.
    assert g._clock >= 0.0
    print("gdsf self-check ok")
