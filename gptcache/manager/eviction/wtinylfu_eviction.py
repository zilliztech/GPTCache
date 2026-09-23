"""Token-Cost-Aware W-TinyLFU eviction policy for GPTCache.

Combines a frequency-based admission filter (TinyLFU with Count-Min Sketch
and Bloom filter doorkeeper) with a segmented LRU main cache and an LRU
window cache. Optionally weighs eviction decisions by the regeneration cost
of cached entries (e.g., response token count).

Architecture:
    Window LRU (1%) --evict--> TinyLFU admission gate --admit--> Main SLRU (99%)
                                     |                              |
                              Count-Min Sketch              Probation (20%)
                              + Bloom doorkeeper            Protected (80%)

References:
    - TinyLFU: Gil Einziger, Roy Friedman, Ben Manes (arXiv:1512.00727)
    - Caffeine: github.com/ben-manes/caffeine
    - Theine: github.com/Yiling-J/theine
"""

import random
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

from gptcache.manager.eviction.base import EvictionBase as EvictionBaseABC
from gptcache.manager.eviction.count_min_sketch import CountMinSketch
from gptcache.manager.eviction.doorkeeper import Doorkeeper
from gptcache.manager.eviction.segmented_lru import SegmentedLRU


class WTinyLFUEviction(EvictionBaseABC):
    """W-TinyLFU eviction policy with optional cost-awareness.

    :param maxsize: total cache capacity (entry count)
    :param clean_size: number of entries to evict per batch (default 20% of maxsize)
    :param on_evict: callback receiving list of evicted entry IDs
    :param window_pct: window cache as percentage of total capacity (default 1.0)
    :param probation_pct: probation as percentage of main cache (default 20.0)
    :param cost_aware: enable cost-weighted eviction decisions (default True)
    :param cms_width_multiplier: CMS width = next_power_of_2(maxsize * this)
    :param reset_multiplier: CMS resets every maxsize * this increments
    """

    def __init__(
        self,
        maxsize: int = 1000,
        clean_size: int = 0,
        on_evict: Optional[Callable[[List[Any]], None]] = None,
        window_pct: float = 1.0,
        probation_pct: float = 20.0,
        cost_aware: bool = True,
        cms_width_multiplier: int = 1,
        reset_multiplier: int = 10,
        **kwargs,
    ):
        self._maxsize = max(maxsize, 4)
        self._clean_size = clean_size if clean_size else int(self._maxsize * 0.2)
        self._on_evict = on_evict
        self._cost_aware = cost_aware

        # Segment sizes — ensure minimum viable sizes for small caches
        # Caffeine uses 1% window, but at small sizes we need at least 1 slot
        # per segment. For cache_size < 20, use larger percentages.
        window_size = max(int(self._maxsize * window_pct / 100.0), 1)
        main_size = max(self._maxsize - window_size, 2)
        probation_size = max(int(main_size * probation_pct / 100.0), 1)
        protected_size = max(main_size - probation_size, 1)
        # Recalculate total to match
        self._maxsize = window_size + probation_size + protected_size

        # Data structures
        self._window = OrderedDict()  # LRU window cache
        self._window_cap = window_size
        self._main = SegmentedLRU(probation_size, protected_size)
        self._sketch = CountMinSketch(
            self._maxsize,
            width_multiplier=cms_width_multiplier,
            sample_size_multiplier=reset_multiplier,
        )
        self._doorkeeper = Doorkeeper(
            capacity=self._maxsize * reset_multiplier
        )

        # Cost metadata: entry_id -> cost weight
        self._cost_map: Dict[int, float] = {}
        # Track which segment each key is in for fast lookup
        self._key_location: Dict[Any, str] = {}  # key -> "window" | "main"

    def put(self, objs: List[Any]):
        """Register entry IDs after insertion into scalar/vector stores.

        Triggers eviction via the W-TinyLFU admission pipeline when the
        cache is full.
        """
        evicted = []

        for obj in objs:
            # If already tracked, just touch it
            if obj in self._key_location:
                self.get(obj)
                continue

            key_hash = hash(obj)
            self._increment_sketch(key_hash)

            # Insert into window cache
            self._window[obj] = True
            self._window.move_to_end(obj)
            self._key_location[obj] = "window"

            # If window overflows, run admission
            while len(self._window) > self._window_cap:
                # Evict LRU from window -> becomes candidate
                cand_key, _ = self._window.popitem(last=False)
                del self._key_location[cand_key]

                if len(self._main) < (self._maxsize - self._window_cap):
                    # Main has room: admit unconditionally
                    self._main.put(cand_key, True)
                    self._key_location[cand_key] = "main"
                else:
                    # Main full: candidate competes with probation victim
                    victim_key = self._main.peek_victim()
                    if victim_key is not None and self._admit(cand_key, victim_key):
                        self._main.evict()
                        del self._key_location[victim_key]
                        self._cost_map.pop(victim_key, None)
                        evicted.append(victim_key)
                        # Admit candidate
                        self._main.put(cand_key, True)
                        self._key_location[cand_key] = "main"
                    else:
                        # Candidate loses, discard it
                        self._cost_map.pop(cand_key, None)
                        evicted.append(cand_key)

        if evicted and self._on_evict:
            self._on_evict(evicted)

    def get(self, obj: Any):
        """Touch an entry on cache hit, updating frequency and LRU position."""
        key_hash = hash(obj)
        self._increment_sketch(key_hash)

        loc = self._key_location.get(obj)
        if loc == "window":
            if obj in self._window:
                self._window.move_to_end(obj)
                return True
        elif loc == "main":
            result = self._main.get(obj)
            if result is not None:
                return result

        return None

    @property
    def policy(self) -> str:
        return "WTINYLFU"

    def set_cost(self, obj_id: Any, cost: float):
        """Set the regeneration cost for a cache entry.

        Cost is used in admission decisions when cost_aware=True.
        Higher cost = more valuable to keep cached.
        """
        self._cost_map[obj_id] = cost

    def _increment_sketch(self, key_hash: int):
        """Doorkeeper-gated sketch increment."""
        if self._doorkeeper.allow(key_hash):
            self._sketch.increment(key_hash)
        if self._sketch.additions >= self._sketch._sample_size:
            self._sketch.reset()
            self._doorkeeper.clear()

    def _estimate_frequency(self, key: Any) -> int:
        """Estimate access frequency for a key."""
        key_hash = hash(key)
        if self._doorkeeper.contains(key_hash):
            return self._sketch.estimate(key_hash) + 1
        return 0

    def _get_cost(self, key: Any) -> float:
        """Get the cost weight for a key (default 1.0)."""
        if not self._cost_aware:
            return 1.0
        return self._cost_map.get(key, 1.0)

    _ADMIT_HASHDOS_THRESHOLD = 6

    def _admit(self, candidate_key: Any, victim_key: Any) -> bool:
        """W-TinyLFU admission decision: should candidate replace victim?

        Follows Caffeine's admission policy:
        1. Candidate wins if its estimated value exceeds the victim's.
        2. When cost-aware, value = frequency * cost; otherwise value = frequency.
        3. At high candidate frequencies (>= 6), admit with ~1/128 probability
           as a hash-DoS defence (Caffeine's ADMIT_HASHDOS_THRESHOLD).
        4. Otherwise reject — favour cache stability at low frequencies.
        """
        freq_c = self._estimate_frequency(candidate_key)
        freq_v = self._estimate_frequency(victim_key)

        if self._cost_aware:
            cost_c = self._get_cost(candidate_key)
            cost_v = self._get_cost(victim_key)
            value_c = freq_c * cost_c
            value_v = freq_v * cost_v
        else:
            value_c = freq_c
            value_v = freq_v

        if value_c > value_v:
            return True

        # Hash-DoS defence: at high frequencies, admit with small probability
        # to prevent an attacker from pinning entries. Matches Caffeine's
        # ~1/128 random admission when candidateFreq >= 6.
        if freq_c >= self._ADMIT_HASHDOS_THRESHOLD:
            return random.randint(0, 127) == 0

        return False
