"""Cost-Aware W-TinyLFU eviction policy with EWMA time-decay frequency.

Combines the W-TinyLFU structure (Window-LRU + SLRU Probation/Protected +
Count-Min Sketch + Doorkeeper) with a lexicographic admission score that
prefers items with higher regeneration cost when frequency is equal.

Admission score (lexicographic — frequency always dominates):
    freq_score  = min(sketch.estimate(key), 15.0) * exp(-λ * dt)   [0, 15]
    cost_score  = EWMACostTracker.score(cost)     log-cost z-score in [0, 15]
    score       = freq_score * 16.0 + cost_score

    1 unit of frequency (= 16 in score) beats the full cost range (max 15),
    so a twice-accessed cheap item always wins over a once-accessed expensive
    item. Cost only breaks ties within the same frequency level.

Frequency signal (Count-Min Sketch — the W-TinyLFU core):
    The admission contest reads the sketch's frequency estimate, NOT a
    per-resident-item counter. This is what makes the policy real W-TinyLFU:
    the sketch persists across eviction, so a popular item that was evicted
    and returns is still recognised as popular — the property that lets
    TinyLFU beat LRU. Estimates are clamped to [0, 15] (Caffeine's 4-bit
    ceiling) and aged by the sketch's periodic halving.

Time decay (our twist on the sketch read):
    freq_score = min(sketch.estimate(key), 15) * exp(-λ * dt)
where dt = seconds since the item's last access (monotonic clock). Decay is
applied at *read* time, so a once-hot item that has not been touched discounts
its own frequency without disturbing the shared sketch. With λ = 1e-5 an item
idle for ~19 hours loses ~half its frequency weight; with dt ≈ 0 (a fast
replay) decay is a no-op and scoring is pure sketch frequency.

Cost normalization (prevents 600× raw-cost spread from overwhelming frequency):
    log(cost) → EWMA(mean, variance) → z-score → clamp[-1,1] → [0, 15]
    Returns 8.0 (neutral) during the first `ewma_warmup` observations.

Doorkeeper (Bloom filter, from TinyLFU paper):
    First access to a key: register in filter, skip sketch increment.
    Second+ access: pass through to sketch.
    This suppresses one-hit wonders — items seen once and never again —
    from inflating sketch counters and polluting the admission decision.
    Cleared whenever the sketch resets to prevent stale false positives.
"""

import math
import random
import statistics
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


DEFAULT_GENERATION_LATENCY_MS = 1000.0
DEFAULT_TOKEN_COUNT = 100
DEFAULT_MODEL_TIER = 1.0

EWMA_FREQ_CAP = 15.0  # aligns freq_score with the [0, 15] cost_score range

# Caffeine's ADMIT_HASHDOS_THRESHOLD adapted for the [0, 15] sketch range
# (cap=15, half≈7). A candidate whose sketch frequency >= this gets a 1/128
# random admission chance even if it loses the score contest, preventing
# frequency-flooding attacks. Uses the raw (undecayed) estimate so an attacker
# cannot evade the guard by pausing to let decay shrink the signal.
_HASHDOS_THRESHOLD = 7.0


@dataclass
class LLMCost:
    """Per-item regeneration cost signal.

    `cost` = latency_ms × model_tier × (1 + tokens/1000), capturing both
    wall-clock delay and model pricing tier in a single scalar.

    Defaults give every item an identical cost, collapsing the policy to
    W-TinyLFU + EWMA decay — still better than LRU until real latency/token
    counts are plumbed through the adapter.
    """

    generation_latency_ms: float = DEFAULT_GENERATION_LATENCY_MS
    token_count: int = DEFAULT_TOKEN_COUNT
    model_tier: float = DEFAULT_MODEL_TIER

    @property
    def cost(self) -> float:
        return (
            self.generation_latency_ms
            * self.model_tier
            * (1.0 + self.token_count / 1000.0)
        )


@dataclass
class ItemMeta:
    """Per-item state for the lexicographic scoring function.

    Frequency is NOT stored here — it lives in the shared Count-Min Sketch so
    it survives eviction. ``last_access`` only feeds the read-time decay.
    """

    key: Any
    cost: float = 0.0        # LLMCost.cost snapshot at last insert/update
    last_access: float = 0.0 # time.monotonic() of last touch


class Doorkeeper:
    """Bloom filter that suppresses one-hit wonders from the sketch.

    A key must be seen at least twice before its Count-Min Sketch counters
    are incremented. Items accessed only once — the long tail of one-off
    queries — never pollute the frequency estimator.

    Cleared on every sketch reset so stale membership does not persist
    across aging cycles (per the TinyLFU paper, Einziger et al. 2017).

    Implementation uses a bytearray bit-array with double-hashing to avoid
    any numpy dependency in this module.
    """

    def __init__(self, capacity: int, fp_rate: float = 0.01):
        capacity = max(capacity, 16)
        # Optimal Bloom sizing: m = -n·ln(p) / (ln2)², k = (m/n)·ln2
        m = int(-capacity * math.log(fp_rate) / (math.log(2) ** 2))
        self._num_bits = max(m, 64)
        self._num_hashes = max(int((self._num_bits / capacity) * math.log(2)), 1)
        self._bits = bytearray((self._num_bits + 7) // 8)

    def allow_and_add(self, key_hash: int) -> bool:
        """Return True if key was already present; always add key to filter."""
        present = self._contains(key_hash)
        self._add(key_hash)
        return present

    def clear(self) -> None:
        for i in range(len(self._bits)):
            self._bits[i] = 0

    def _contains(self, h: int) -> bool:
        for i in range(self._num_hashes):
            pos = self._hash_pos(h, i)
            if not (self._bits[pos >> 3] & (1 << (pos & 7))):
                return False
        return True

    def _add(self, h: int) -> None:
        for i in range(self._num_hashes):
            pos = self._hash_pos(h, i)
            self._bits[pos >> 3] |= 1 << (pos & 7)

    def _hash_pos(self, h: int, i: int) -> int:
        h1 = h & 0xFFFFFFFF
        h2 = (h >> 32) & 0xFFFFFFFF
        return ((h1 + i * h2) & 0xFFFFFFFFFFFFFFFF) % self._num_bits


class EWMACostTracker:
    """Maps LLM regeneration costs to [0, 15] via EWMA z-score normalization.

    Pipeline per access:
        cost  →  log(cost)  →  EWMA(mean, variance)  →  z-score  →
        clamp[-1, 1]  →  [0, 15]

    Log-transform compresses the heavy-tailed cost distribution (token counts
    span 10–4000+, latency spans 200ms–15s, producing raw costs with a ~600×
    spread). Without this, raw cost would dominate the score regardless of
    the lexicographic structure.

    EWMA alpha=0.05 gives an effective window of ~20 samples — slow enough
    to be stable, fast enough to track workload shifts over hours.

    During the first `warmup` observations the distribution is unreliable,
    so score() returns 8.0 (the neutral midpoint of [0, 15]).
    """

    def __init__(self, alpha: float = 0.05, warmup: int = 20):
        self._alpha = alpha
        self._warmup = warmup
        self._count = 0
        self._mean = 0.0
        self._variance = 1.0

    def update(self, cost: float) -> None:
        log_c = math.log(max(cost, 1.0))
        if self._count == 0:
            self._mean = log_c
        else:
            delta = log_c - self._mean
            self._mean += self._alpha * delta
            self._variance = (1.0 - self._alpha) * (
                self._variance + self._alpha * delta * delta
            )
        self._count += 1

    def score(self, cost: float) -> float:
        """Return cost score in [0.0, 15.0]. Returns 8.0 during warmup."""
        if self._count < self._warmup:
            return 8.0
        log_c = math.log(max(cost, 1.0))
        std = math.sqrt(max(self._variance, 1e-10))
        z = (log_c - self._mean) / std
        z = max(-1.0, min(1.0, z))
        return (z + 1.0) * 7.5  # [-1, 1] → [0, 15]


class CountMinSketch:
    """Count-Min Sketch with periodic halving (W-TinyLFU frequency filter).

    increment() returns True when a halve occurred so the caller can clear
    the Doorkeeper in sync with the aging cycle.
    """

    def __init__(self, width: int = 2048, depth: int = 4, seed: int = 0):
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive")
        self._width = width
        self._depth = depth
        self._table = [[0] * width for _ in range(depth)]
        self._seeds = [seed + i * 0x9E3779B1 for i in range(depth)]
        self._reset_threshold = 10 * width
        self._total = 0

    def _index(self, row: int, key: Any) -> int:
        return (hash((self._seeds[row], key)) & 0x7FFFFFFF) % self._width

    def increment(self, key: Any) -> bool:
        """Increment counters for key. Returns True if a halve occurred."""
        for r in range(self._depth):
            self._table[r][self._index(r, key)] += 1
        self._total += 1
        if self._total >= self._reset_threshold:
            self._halve()
            return True
        return False

    def estimate(self, key: Any) -> int:
        return min(self._table[r][self._index(r, key)] for r in range(self._depth))

    def _halve(self) -> None:
        for r in range(self._depth):
            row = self._table[r]
            for i in range(self._width):
                row[i] >>= 1
        self._total >>= 1


class _LRUSegment:
    """OrderedDict-backed LRU with peek_victim / evict_victim helpers."""

    def __init__(self, maxsize: int):
        self.maxsize = max(1, maxsize)
        self._od: "OrderedDict[Any, bool]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._od)

    def __contains__(self, key: Any) -> bool:
        return key in self._od

    def is_full(self) -> bool:
        return len(self._od) >= self.maxsize

    def add_mru(self, key: Any) -> None:
        self._od[key] = True
        self._od.move_to_end(key, last=True)

    def touch(self, key: Any) -> None:
        if key in self._od:
            self._od.move_to_end(key, last=True)

    def peek_victim(self) -> Optional[Any]:
        if not self._od:
            return None
        return next(iter(self._od))

    def evict_victim(self) -> Optional[Any]:
        if not self._od:
            return None
        key, _ = self._od.popitem(last=False)
        return key

    def pop_mru(self) -> Optional[Any]:
        if not self._od:
            return None
        key, _ = self._od.popitem(last=True)
        return key

    def remove(self, key: Any) -> bool:
        return self._od.pop(key, None) is not None


class CostAwareWTinyLFU:
    """Cost-Aware W-TinyLFU with EWMA time-decay and lexicographic scoring.

    Structure:
        window LRU    (~1% of maxsize)   — new items land here unconditionally
        probation LRU (~20% of main)     — admitted but not yet proven
        protected LRU (~80% of main)     — items that earned a second access

    Insert flow:
        1. Register cost in EWMACostTracker; register key in Doorkeeper.
        2. Add to window.
        3. If window was full: pop its LRU victim → admission contest.

    Admission contest (window victim vs. probation LRU end):
        winner = argmax(score); loser → on_evict callback + meta removed.
        score = freq_score * 16 + cost_score  (lexicographic)

    Get flow:
        1. Doorkeeper-gated sketch increment + last-access timestamp.
        2. Segment routing: protected → touch; probation → promote; window → touch.

    Public API matches MemoryCacheEviction (put / get / policy) so the manager
    routes to it interchangeably.
    """

    def __init__(
        self,
        maxsize: int,
        on_evict: Optional[Callable[[List[Any]], None]] = None,
        decay_rate: float = 1e-5,
        sketch_width: int = 2048,
        sketch_depth: int = 4,
        window_ratio: float = 0.01,
        protected_ratio: float = 0.8,
        ewma_alpha: float = 0.05,
        ewma_warmup: int = 20,
        time_fn: Callable[[], float] = time.monotonic,
        default_cost: Optional[LLMCost] = None,
        cost_aware: bool = True,
        freq_weight: float = 16.0,
        cost_priority: Optional[float] = None,
        adaptive_window: bool = False,
        adapt_sample_factor: float = 10.0,
        adapt_step_ratio: float = 0.0625,
        adapt_step_decay: float = 0.98,
        adapt_objective: str = "cost",
        adapt_decision_intervals: int = 3,
        **_unused,
    ):
        if maxsize < 4:
            raise ValueError(
                "maxsize must be >= 4 for W-TinyLFU to allocate all three segments"
            )
        self._maxsize = maxsize
        # ponytail: CA evicts exactly one item per admission contest by
        # construction, so the cachetools `clean_size` batch knob does not apply.
        self._on_evict = on_evict or (lambda keys: None)
        self._decay_rate = decay_rate
        self._time = time_fn
        self._default_cost = default_cost or LLMCost()
        # cost_aware=False collapses the score to frequency-only, reproducing a
        # plain W-TinyLFU (the prior-art baseline). freq_weight controls how
        # strongly frequency dominates cost: 16.0 (>= cost range of 15) keeps the
        # original lexicographic behaviour; lower values blend cost into the
        # decision so it influences eviction beyond mere tie-breaking.
        self._cost_aware = cost_aware
        # cost_priority is the user-facing money<->hit-rate dial in [0, 1].
        # 0.0 => freq_weight 16 (lexicographic: frequency dominates, cost only
        # breaks ties => maximize hit rate, spend freely on regeneration).
        # 1.0 => freq_weight 1 (cost on equal footing with frequency =>
        # maximize money saved by keeping expensive answers). Overrides
        # freq_weight when set; None keeps the raw freq_weight for backward compat.
        if cost_priority is not None:
            if not 0.0 <= cost_priority <= 1.0:
                raise ValueError("cost_priority must be in [0, 1]")
            freq_weight = 16.0 - 15.0 * cost_priority
        self._freq_weight = freq_weight

        self._protected_ratio = protected_ratio
        window_size = max(1, int(maxsize * window_ratio))
        main_size = maxsize - window_size
        protected_size = max(1, int(main_size * protected_ratio))
        probation_size = max(1, main_size - protected_size)

        self._window = _LRUSegment(window_size)
        self._probation = _LRUSegment(probation_size)
        self._protected = _LRUSegment(protected_size)

        # --- Adaptive window (Caffeine-style hill-climb; default OFF) ---
        # When enabled, the window<->main boundary is re-tuned every interval to
        # maximize the *cost-weighted* hit rate (the novel twist vs. Caffeine,
        # which climbs raw hit rate). Default off => byte-for-byte identical to
        # the fixed-window policy.
        self._adaptive = adaptive_window
        self._adapt_interval = max(1, int(maxsize * adapt_sample_factor))
        self._adapt_step = adapt_step_ratio * maxsize  # signed; + grows window
        self._adapt_decay = adapt_step_decay
        self._adapt_objective = adapt_objective
        self._adapt_accesses = 0
        self._adapt_hits = 0
        self._adapt_hit_cost = 0.0
        self._adapt_total_cost = 0.0
        self._prev_objective: Optional[float] = None
        # A decision averages the objective over this many intervals before
        # moving the boundary, so one noisy interval can't flip direction.
        self._decision_intervals = max(1, adapt_decision_intervals)
        self._interval_objs: List[float] = []   # objectives awaiting a decision
        self._regress_count = 0                  # consecutive regressions (hysteresis)
        self._seeded = False                     # has the gradient sign been probed yet
        self._probe_stage = 0                    # 0..2 two-sided gradient probe
        self._probe_up_obj = 0.0                 # objective at the +probe window
        self._probe_mag = 0                      # probe step magnitude
        self._settle = 0                         # intervals to skip after a move

        self._meta: Dict[Any, ItemMeta] = {}
        self._sketch = CountMinSketch(sketch_width, sketch_depth)
        self._doorkeeper = Doorkeeper(capacity=maxsize)
        self._cost_tracker = EWMACostTracker(alpha=ewma_alpha, warmup=ewma_warmup)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def policy(self) -> str:
        return "CA_W_TINYLFU"

    def __len__(self) -> int:
        return len(self._window) + len(self._probation) + len(self._protected)

    def __contains__(self, key: Any) -> bool:
        return key in self._meta

    def put(self, objs: List[Any], costs: Optional[List[LLMCost]] = None) -> None:
        if costs is None:
            costs = [self._default_cost] * len(objs)
        elif len(costs) != len(objs):
            raise ValueError("costs length must match objs length")
        for obj, cost in zip(objs, costs):
            self._insert(obj, cost)

    def get(self, obj: Any) -> Optional[bool]:
        if obj not in self._meta:
            return None
        hit_cost = self._meta[obj].cost
        self._record_access(obj)
        self._touch_segments(obj)
        self._tick_adapt(is_hit=True, cost=hit_cost)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_access(self, key: Any) -> None:
        """Doorkeeper-gated sketch increment + last-access timestamp.

        Frequency lives in the shared sketch (so it survives eviction); the
        first sighting of a key is suppressed by the Doorkeeper and only its
        second+ accesses reach the sketch, keeping one-hit wonders out.
        """
        k_hash = hash(key)
        if self._doorkeeper.allow_and_add(k_hash):
            if self._sketch.increment(key):
                self._doorkeeper.clear()
        self._meta[key].last_access = self._time()

    def _freq_score(self, key: Any) -> float:
        """Sketch frequency in [0, EWMA_FREQ_CAP], discounted by idle time.

        The estimate is read from the shared Count-Min Sketch (W-TinyLFU's
        core — it remembers frequency across eviction), clamped to Caffeine's
        4-bit ceiling, then multiplied by ``exp(-λ·dt)`` so an item that has
        not been touched recently discounts its own weight. dt ≈ 0 ⇒ decay is
        a no-op and the score is pure sketch frequency.
        """
        raw = min(float(self._sketch.estimate(key)), EWMA_FREQ_CAP)
        dt = max(0.0, self._time() - self._meta[key].last_access)
        decay = math.exp(-self._decay_rate * dt) if dt > 0 else 1.0
        return raw * decay

    def _score(self, key: Any) -> float:
        """Admission score: sketch frequency weighted by ``freq_weight``, plus cost.

        Both components are in [0, 15]. With ``freq_weight = 16`` (default) one
        unit of freq_score outweighs the full cost range (max 15), so the score
        is lexicographic — a twice-accessed cheap item always beats a
        once-accessed expensive one and cost only breaks exact ties. Lower
        ``freq_weight`` blends cost into the ordering so it can override small
        frequency differences. ``cost_aware=False`` drops the cost term
        entirely, reproducing a plain frequency-only W-TinyLFU.
        """
        score = self._freq_score(key) * self._freq_weight
        if self._cost_aware:
            score += self._cost_tracker.score(self._meta[key].cost)  # [0, 15]
        return score

    def _touch_segments(self, key: Any) -> None:
        if key in self._protected:
            self._protected.touch(key)
        elif key in self._probation:
            self._probation.remove(key)
            self._promote_to_protected(key)
        elif key in self._window:
            self._window.touch(key)

    def _insert(self, key: Any, cost: LLMCost) -> None:
        self._cost_tracker.update(cost.cost)

        if key in self._meta:
            m = self._meta[key]
            m.cost = cost.cost
            self._record_access(key)
            self._touch_segments(key)
            self._tick_adapt(is_hit=True, cost=cost.cost)
            return

        now = self._time()
        self._meta[key] = ItemMeta(
            key=key,
            cost=cost.cost,
            last_access=now,
        )
        self._record_access(key)  # registers in doorkeeper (first sighting, no sketch bump)

        if self._window.is_full():
            window_victim = self._window.evict_victim()
            if window_victim is not None:
                self._admit_or_reject(window_victim)
        self._window.add_mru(key)
        self._tick_adapt(is_hit=False, cost=cost.cost)

    def _admit_or_reject(self, candidate: Any) -> None:
        if not self._probation.is_full():
            self._probation.add_mru(candidate)
            return

        victim = self._probation.peek_victim()
        if victim is None:
            self._probation.add_mru(candidate)
            return

        # Strict greater-than: on a tie the victim (proven, in main) wins over
        # the candidate (unproven, from window). Matches Caffeine's admit().
        candidate_score = self._score(candidate)
        victim_score = self._score(victim)
        if candidate_score > victim_score:
            self._probation.evict_victim()
            self._emit_evict([victim])
            self._probation.add_mru(candidate)
        elif min(float(self._sketch.estimate(candidate)), EWMA_FREQ_CAP) >= _HASHDOS_THRESHOLD:
            # Hash-DoS defence: a moderately warm candidate that loses on score
            # gets a 1/128 random admission chance. Prevents an attacker from
            # pinning the victim by artificially inflating its frequency.
            # Matches Caffeine's ADMIT_HASHDOS_THRESHOLD logic (raw sketch freq).
            if random.randint(0, 127) == 0:
                self._probation.evict_victim()
                self._emit_evict([victim])
                self._probation.add_mru(candidate)
            else:
                self._emit_evict([candidate])
        else:
            self._emit_evict([candidate])

    def _promote_to_protected(self, key: Any) -> None:
        if self._protected.is_full():
            demoted = self._protected.evict_victim()
            if demoted is not None:
                self._admit_or_reject(demoted)
        self._protected.add_mru(key)

    def _emit_evict(self, keys: List[Any]) -> None:
        for k in keys:
            self._meta.pop(k, None)
        if keys:
            self._on_evict(keys)

    # ------------------------------------------------------------------
    # Adaptive window (optional; default off reproduces fixed-window behaviour)
    # ------------------------------------------------------------------

    def _tick_adapt(self, is_hit: bool, cost: float) -> None:
        """Accumulate per-interval access stats and close intervals.

        No-op unless ``adaptive_window=True`` — so the default policy is
        byte-for-byte identical to the fixed-window version (regression guard).
        """
        if not self._adaptive:
            return
        self._adapt_accesses += 1
        self._adapt_total_cost += cost
        if is_hit:
            self._adapt_hits += 1
            self._adapt_hit_cost += cost
        if self._adapt_accesses >= self._adapt_interval:
            self._end_interval()

    def _interval_objective(self) -> float:
        """Objective for the interval just closed (cost-weighted or raw hit)."""
        if self._adapt_objective == "hit":
            return (self._adapt_hits / self._adapt_accesses
                    if self._adapt_accesses else 0.0)
        return (self._adapt_hit_cost / self._adapt_total_cost
                if self._adapt_total_cost > 0 else 0.0)

    def _reset_interval(self) -> None:
        self._adapt_accesses = 0
        self._adapt_hits = 0
        self._adapt_hit_cost = 0.0
        self._adapt_total_cost = 0.0

    def _end_interval(self) -> None:
        """Bank one interval's objective; decide once enough have accumulated.

        The interval right after a boundary move is discarded as a settling
        period: its hit rate reflects the *old* split (the freshly enlarged
        segment has not refilled yet), so banking it would feed the climber a
        transient instead of the new steady state — the bias that made the
        gradient probe mis-seed.
        """
        if self._settle > 0:
            self._settle -= 1
            self._reset_interval()
            return
        self._interval_objs.append(self._interval_objective())
        self._reset_interval()
        if len(self._interval_objs) >= self._decision_intervals:
            self._climb()

    def _climb(self) -> None:
        """Hill-climb the window<->main boundary toward a better objective.

        The objective is the *cost-weighted* hit rate (``adapt_objective="cost"``)
        — the novel twist vs. Caffeine, which climbs raw hit rate. Three guards
        against the "climbs the wrong way" failure:

          1. The decision uses the *mean* objective over ``_decision_intervals``
             intervals, not one noisy interval.
          2. A two-sided gradient probe (``_seed_direction``) samples the
             objective at W0±probe and commits the step sign toward the better
             side, instead of always growing first. The large probe escapes
             locally-flat regions where a one-step gradient is pure noise.
          3. Reversals only fire after *two consecutive* regressions (temporal
             hysteresis), so one noisy decision can't flip direction, while a
             sustained regression (the optimum was passed) reverses promptly.

        # ponytail: 1-D bounded climb; adopt Caffeine's sampled scheme only if
        # this still misbehaves under the heavy validation.
        """
        measurement = statistics.fmean(self._interval_objs)
        self._interval_objs = []

        if not self._seeded:
            self._seed_direction(measurement)
            return

        if measurement < self._prev_objective:
            self._regress_count += 1
            if self._regress_count >= 2:
                # Sustained regression: the optimum is behind us — reverse and
                # shrink the step so the window settles around the peak.
                self._adapt_step = -self._adapt_step * self._adapt_decay
                self._regress_count = 0
        else:
            self._regress_count = 0
        self._prev_objective = measurement
        self._step_window()

    def _seed_direction(self, measurement: float) -> None:
        """Probe both directions over the first three decisions, then commit.

        Stage 0 (at W0): jump up by ``_probe_mag`` to sample the +direction.
        Stage 1 (at W0+probe): bank that objective, jump down to W0-probe.
        Stage 2 (at W0-probe): pick the step sign toward the better side and
        start the real climb. Probing both sides — not just continuing to grow —
        removes the "always grows first" bias, and a probe of maxsize/8
        gives a gradient signal even when W0 sits in a locally-flat region.
        """
        if self._probe_stage == 0:
            self._probe_mag = max(int(round(abs(self._adapt_step))), self._maxsize // 8)
            self._move_window(self._window.maxsize + self._probe_mag)
            self._probe_stage = 1
        elif self._probe_stage == 1:
            self._probe_up_obj = measurement
            self._move_window(self._window.maxsize - 2 * self._probe_mag)
            self._probe_stage = 2
        else:
            grow = self._probe_up_obj >= measurement  # +probe vs -probe objective
            self._adapt_step = abs(self._adapt_step) * (1.0 if grow else -1.0)
            self._prev_objective = measurement
            self._seeded = True
            self._step_window()

    def _step_window(self) -> None:
        delta = int(round(self._adapt_step))
        if delta != 0:
            self._move_window(self._window.maxsize + delta)

    def _move_window(self, target: int) -> None:
        """Resize the window and arm a one-interval settling skip."""
        self._resize_window(target)
        self._settle = 1

    def _resize_window(self, new_window_size: int) -> None:
        """Move the window<->main boundary, conserving total capacity.

        Shrinking demotes window-LRU victims into main via the normal admission
        contest; growing pulls probation-MRU items back up into the window and
        trims any resulting main overflow. Segment capacities always re-sum to
        ``maxsize`` so no slot is created or lost.
        """
        new_window_size = max(1, min(new_window_size, self._maxsize - 2))
        old_window_size = self._window.maxsize
        if new_window_size == old_window_size:
            return

        new_main = self._maxsize - new_window_size
        new_protected = max(1, int(new_main * self._protected_ratio))
        new_probation = max(1, new_main - new_protected)

        if new_window_size < old_window_size:
            # Window shrinks, main grows: demote window overflow into main.
            self._window.maxsize = new_window_size
            self._probation.maxsize = new_probation
            self._protected.maxsize = new_protected
            while len(self._window) > new_window_size:
                victim = self._window.evict_victim()
                if victim is None:
                    break
                self._admit_or_reject(victim)
        else:
            # Window grows, main shrinks: pull probation-MRU up, trim overflow.
            self._window.maxsize = new_window_size
            while len(self._window) < new_window_size and len(self._probation) > 0:
                promoted = self._probation.pop_mru()
                if promoted is None:
                    break
                self._window.add_mru(promoted)
            self._probation.maxsize = new_probation
            self._protected.maxsize = new_protected
            while len(self._probation) > self._probation.maxsize:
                evicted = self._probation.evict_victim()
                if evicted is None:
                    break
                self._emit_evict([evicted])
            while len(self._protected) > self._protected.maxsize:
                demoted = self._protected.evict_victim()
                if demoted is None:
                    break
                self._admit_or_reject(demoted)
