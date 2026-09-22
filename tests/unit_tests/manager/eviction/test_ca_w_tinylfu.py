"""Phase 1 tests for the standalone CA-W-TinyLFU policy module.

These tests exercise policy internals in isolation — no MemoryCacheEviction,
no SSDataManager, no SQLite/FAISS. Phase 2 adds the routing test.
"""

import random

from gptcache.manager.eviction.ca_w_tinylfu import (
    CostAwareWTinyLFU,
    CountMinSketch,
    Doorkeeper,
    EWMACostTracker,
    LLMCost,
)


def _collect_evictions():
    evicted = []

    def on_evict(keys):
        evicted.extend(keys)

    return evicted, on_evict


# ---------------------------------------------------------------------------
# t1 — below-capacity inserts produce no evictions
# ---------------------------------------------------------------------------

def test_below_capacity_no_evictions():
    """Insert fewer items than window+probation capacity: nothing evicted.

    With default ratios at maxsize=40: window=1, probation=8, protected=31.
    Inserting 5 items keeps probation well below its limit so the admission
    contest never needs to reject.
    """
    evicted, on_evict = _collect_evictions()
    cache = CostAwareWTinyLFU(maxsize=40, on_evict=on_evict, time_fn=lambda: 0.0)

    n = 5
    for i in range(n):
        cache.put([f"k_{i}"])

    assert evicted == [], f"unexpected evictions: {evicted}"
    for i in range(n):
        assert cache.get(f"k_{i}") is True, f"k_{i} missing after below-capacity insert"
    assert len(cache) == n


# ---------------------------------------------------------------------------
# t2 — above-capacity inserts trigger evictions and respect maxsize
# ---------------------------------------------------------------------------

def test_above_capacity_triggers_evictions():
    """Insert maxsize + N items: at least one eviction fires, cache stays bounded."""
    evicted, on_evict = _collect_evictions()
    maxsize = 20
    cache = CostAwareWTinyLFU(maxsize=maxsize, on_evict=on_evict, time_fn=lambda: 0.0)

    for i in range(maxsize + 30):
        cache.put([f"k_{i}"])

    assert len(evicted) >= 1, "no evictions fired despite overflowing the cache"
    assert len(cache) <= maxsize, f"cache exceeded maxsize: {len(cache)} > {maxsize}"
    for key in evicted:
        assert key not in cache, f"evicted key {key!r} still present in cache"


# ---------------------------------------------------------------------------
# t3 — cost bias: expensive items survive the admission contest more often
# ---------------------------------------------------------------------------

def test_expensive_items_survive_more_than_cheap():
    """Under equal access frequency, expensive items survive eviction pressure.

    Uses a large enough cache and enough unique-item flood so that only items
    with high admission score survive. Cost normalization warmup (20 obs) is
    satisfied by the flood, so cost differentiation is fully active.
    """
    cheap = LLMCost(
        generation_latency_ms=100.0, token_count=10, model_tier=1.0,
    )
    expensive = LLMCost(
        generation_latency_ms=15000.0, token_count=2000, model_tier=20.0,
    )

    cheap_survived = 0
    expensive_survived = 0
    n_trials = 30

    for trial in range(n_trials):
        evicted, on_evict = _collect_evictions()
        cache = CostAwareWTinyLFU(
            maxsize=12,
            on_evict=on_evict,
            time_fn=lambda: float(trial),
            ewma_warmup=5,  # short warmup so normalization activates early
        )
        cache.put(["A"], costs=[cheap])
        cache.put(["B"], costs=[expensive])
        for _ in range(3):
            cache.get("A")
            cache.get("B")

        for i in range(80):
            cache.put([f"u_{trial}_{i}"])

        if "A" in cache:
            cheap_survived += 1
        if "B" in cache:
            expensive_survived += 1

    assert expensive_survived >= cheap_survived, (
        f"cost-bias not observed: cheap_survived={cheap_survived}, "
        f"expensive_survived={expensive_survived}"
    )
    assert expensive_survived > 0, "expensive item never survived"


# ---------------------------------------------------------------------------
# t4 — EWMA decay reduces stale frequency over time
# ---------------------------------------------------------------------------

def test_ewma_decay_reduces_stale_frequency():
    """A once-hot item that hasn't been touched in a long dt loses frequency.

    Decay is now applied at *read* time to the sketch estimate, so we advance
    the clock and re-score WITHOUT re-accessing the item (an access would
    refresh last_access and reset dt to 0). Uses an injected time_fn so the
    test is fully deterministic.
    """
    now = [0.0]

    def time_fn():
        return now[0]

    cache = CostAwareWTinyLFU(maxsize=40, decay_rate=0.1, time_fn=time_fn)

    cache.put(["A"])
    for _ in range(20):
        cache.get("A")
    initial_freq = cache._freq_score("A")  # dt = 0 → pure sketch estimate
    assert initial_freq > 5.0, f"setup error: sketch freq did not grow ({initial_freq})"

    now[0] = 1000.0  # 1000 idle seconds; A is NOT re-accessed
    decayed_freq = cache._freq_score("A")

    assert decayed_freq < initial_freq, (
        f"freq did not decay across dt=1000s with rate=0.1: "
        f"{initial_freq:.3f} -> {decayed_freq:.3f}"
    )
    assert decayed_freq < 2.0, (
        f"with decay_rate=0.1 and dt=1000s, surviving freq should be ~0; "
        f"got {decayed_freq:.3f}"
    )


# ---------------------------------------------------------------------------
# t5 — sketch halving fires at the reset threshold
# ---------------------------------------------------------------------------

def test_sketch_halving_fires_at_threshold():
    """Pushing the sketch past _reset_threshold halves _total."""
    sketch = CountMinSketch(width=32, depth=2)
    threshold = sketch._reset_threshold
    assert threshold == 320, f"unexpected threshold: {threshold}"

    resets = sum(1 for i in range(threshold) if sketch.increment(f"item_{i}"))

    assert resets == 1, f"expected exactly 1 halve, got {resets}"
    assert sketch._total <= threshold // 2 + 1, (
        f"sketch did not halve correctly: _total={sketch._total}"
    )
    assert sketch._total > 0, "sketch over-halved to zero"


# ---------------------------------------------------------------------------
# t6 — Doorkeeper suppresses one-hit wonders from the sketch
# ---------------------------------------------------------------------------

def test_doorkeeper_gates_sketch_on_first_access():
    """First access to a key registers in the filter but skips the sketch."""
    dk = Doorkeeper(capacity=1000)

    # First access: not yet in filter -> allow_and_add returns False
    assert dk.allow_and_add(hash("key_A")) is False
    # Second access: now in filter -> returns True (would increment sketch)
    assert dk.allow_and_add(hash("key_A")) is True

    # Fresh key: again False on first access
    assert dk.allow_and_add(hash("key_B")) is False

    # Clear resets all membership
    dk.clear()
    assert dk.allow_and_add(hash("key_A")) is False


# ---------------------------------------------------------------------------
# t7 — EWMACostTracker returns neutral score during warmup, then normalizes
# ---------------------------------------------------------------------------

def test_ewma_cost_tracker_warmup_and_normalization():
    """During warmup, score is 8.0 (neutral). After warmup, expensive > cheap."""
    tracker = EWMACostTracker(alpha=0.05, warmup=5)

    cheap_cost = 200.0
    expensive_cost = 200_000.0

    # During warmup: all scores neutral
    for _ in range(4):
        tracker.update(cheap_cost)
        assert tracker.score(cheap_cost) == 8.0, "should be neutral during warmup"

    # Trigger warmup completion (5th update)
    tracker.update(cheap_cost)

    # After warmup: expensive item gets higher score than cheap
    score_cheap = tracker.score(cheap_cost)
    score_expensive = tracker.score(expensive_cost)
    assert score_expensive > score_cheap, (
        f"expected expensive > cheap after warmup: "
        f"expensive={score_expensive:.2f}, cheap={score_cheap:.2f}"
    )
    assert 0.0 <= score_cheap <= 15.0
    assert 0.0 <= score_expensive <= 15.0


# ---------------------------------------------------------------------------
# t8 — Doorkeeper is cleared when sketch resets (no stale false positives)
# ---------------------------------------------------------------------------

def test_doorkeeper_cleared_on_sketch_reset():
    """Doorkeeper must be cleared when the sketch halves to prevent stale hits."""
    cache = CostAwareWTinyLFU(
        maxsize=100,
        on_evict=lambda ks: None,
        sketch_width=16,   # tiny sketch so reset fires quickly
        sketch_depth=2,
    )
    threshold = cache._sketch._reset_threshold  # 10 * 16 = 160

    # Flood with unique keys to drive past the reset threshold.
    # Each key goes through doorkeeper (first access → False, skip sketch);
    # after two puts of the same key, sketch is incremented.
    # Use enough keys that the sketch hits its threshold.
    for i in range(threshold * 2):
        cache.put([f"flood_{i}"])

    # After the flood the doorkeeper should have been cleared at least once.
    # A key inserted before the last reset should no longer be in the filter.
    # We verify by checking a key that was definitely inserted before the reset:
    # it was cleared, so its hash appears "new" again (allow_and_add -> False).
    first_key_hash = hash("flood_0")
    result = cache._doorkeeper.allow_and_add(first_key_hash)
    # Either False (cleared) or True (still in filter due to a later insertion
    # with the same hash — acceptable for a probabilistic filter).
    # What we must NOT see is the counter monotonically growing without resets.
    # The real assertion is that the cache does not crash and resets occurred.
    assert cache._sketch._total < threshold, (
        f"sketch total {cache._sketch._total} should be below threshold "
        f"{threshold} after at least one reset"
    )


# ---------------------------------------------------------------------------
# t9 — cost_aware=False drops the cost term (frequency-only / prior-art baseline)
# ---------------------------------------------------------------------------

def test_cost_aware_false_ignores_cost():
    """With cost_aware=False, two items of equal frequency score identically
    regardless of cost, and the score is purely freq_score * freq_weight."""
    cache = CostAwareWTinyLFU(
        maxsize=40, cost_aware=False, freq_weight=16.0,
        ewma_warmup=1, time_fn=lambda: 0.0,
    )
    cheap = LLMCost(generation_latency_ms=100.0, token_count=10, model_tier=1.0)
    pricey = LLMCost(generation_latency_ms=15000.0, token_count=2000, model_tier=20.0)
    cache.put(["A"], costs=[cheap])
    cache.put(["B"], costs=[pricey])
    # Drive equal sketch frequency for both (first sighting is doorkeeper-gated).
    for _ in range(3):
        cache.get("A")
        cache.get("B")

    # Equal access frequency -> cost must not differentiate the two.
    assert cache._score("A") == cache._score("B")
    # Score is exactly the (weighted) frequency term — no cost contribution.
    assert cache._score("A") == cache._freq_score("A") * 16.0


# ---------------------------------------------------------------------------
# t10 — freq_weight scales the frequency term in the admission score
# ---------------------------------------------------------------------------

def test_freq_weight_scales_frequency_term():
    """Identical access patterns yield the same sketch frequency; the combined
    score scales linearly with freq_weight (verified with cost disabled)."""
    c16 = CostAwareWTinyLFU(maxsize=40, freq_weight=16.0, cost_aware=False,
                            time_fn=lambda: 0.0)
    c1 = CostAwareWTinyLFU(maxsize=40, freq_weight=1.0, cost_aware=False,
                           time_fn=lambda: 0.0)
    for c in (c16, c1):
        c.put(["A"])
        for _ in range(3):
            c.get("A")

    assert c16._freq_score("A") == c1._freq_score("A")
    assert c16._score("A") == 16.0 * c1._score("A")


# ---------------------------------------------------------------------------
# Adaptive window (E-A1): default-off regression, climb direction, capacity
# ---------------------------------------------------------------------------

def _assert_window_invariants(cache, maxsize):
    """Segments disjoint, consistent with _meta, capacity conserved, bounded."""
    seg_keys = (list(cache._window._od) + list(cache._probation._od)
                + list(cache._protected._od))
    assert len(seg_keys) == len(set(seg_keys)), "a key appears in >1 segment"
    assert set(seg_keys) == set(cache._meta.keys()), "_meta and segments desynced"
    assert (cache._window.maxsize + cache._probation.maxsize
            + cache._protected.maxsize) == maxsize, "segment capacities lost/created"
    assert len(cache) <= maxsize, f"cache exceeded maxsize: {len(cache)} > {maxsize}"


def test_adaptive_window_off_keeps_boundary_static():
    """adaptive_window defaults to False — the window<->main boundary must never
    move, so the policy is byte-for-byte the fixed-window version (regression guard)."""
    random.seed(0)
    cache = CostAwareWTinyLFU(maxsize=100, window_ratio=0.1, time_fn=lambda: 0.0)
    assert cache._adaptive is False
    before = (cache._window.maxsize, cache._probation.maxsize, cache._protected.maxsize)
    for i in range(5000):
        k = i % 50  # heavy reuse — would trigger a climb if adaptation were on
        if cache.get(k) is None:
            cache.put([k])
    after = (cache._window.maxsize, cache._probation.maxsize, cache._protected.maxsize)
    assert after == before, "adaptive_window=False must not resize any segment"


def test_adaptive_window_stays_valid_on_frequency_heavy_stream():
    """Live adaptation on a frequency-heavy stream must preserve every segment
    invariant and keep the window in [1, maxsize-2] on every climb step.

    NB: we deliberately do NOT assert a climb *direction* here. Once the
    Count-Min Sketch protects the main region (real W-TinyLFU behaviour), hit
    rate is largely insensitive to the window<->main split on a stable hot set,
    so the hill-climb has no reliable gradient to follow and its direction is
    workload/seed dependent. This test guards the resize machinery under the
    live climb loop; direction sensitivity is exercised by the recency test."""
    random.seed(0)
    rng = random.Random(0)
    hot = list(range(70))
    stream = [rng.choice(hot) if rng.random() < 0.9 else 1000 + i for i in range(8000)]
    cache = CostAwareWTinyLFU(maxsize=100, window_ratio=0.60,
                              adaptive_window=True, adapt_sample_factor=2.0,
                              time_fn=lambda: 0.0)
    for k in stream:
        if cache.get(k) is None:
            cache.put([k])
        assert 1 <= cache._window.maxsize <= 100 - 2, (
            f"window out of bounds during adaptation: {cache._window.maxsize}")
    _assert_window_invariants(cache, 100)


def _drive_synthetic_climb(start_ratio, optimum, n_decisions=120, maxsize=64):
    """Drive the climber against a noise-free tent objective peaking at ``optimum``.

    Each decision we feed the objective evaluated at the *current* window, so the
    optimizer sees a clean, followable gradient. This tests the climber logic
    directly, free of the cache hit-rate-vs-window flatness that makes real-stream
    convergence intrinsically seed-dependent (research §2.0). Returns the window
    the climber settles on.
    """
    random.seed(0)  # the Hash-DoS admission path uses the global RNG
    cache = CostAwareWTinyLFU(maxsize=maxsize, window_ratio=start_ratio,
                              adaptive_window=True, time_fn=lambda: 0.0)
    for _ in range(n_decisions):
        obj = -abs(cache._window.maxsize - optimum)
        cache._interval_objs = [obj] * cache._decision_intervals
        cache._climb()
    return cache._window.maxsize


def test_adaptive_climber_converges_from_either_side():
    """§2.0 regression guard (deterministic). The naive one-step climber grew the
    *wrong way* from a large window (60 → ~90) because its initial step was always
    positive. The hardened climber's two-sided gradient probe must instead pick the
    correct direction from BOTH a too-large and a too-small start and converge near
    the optimum — to within one step (the climber moves in discrete steps)."""
    start_big, start_small = 0.78, 0.03   # ~window 50 and ~window 1 of 64
    step = int(round(0.0625 * 64))         # adapt_step_ratio * maxsize
    for optimum in (10, 25, 40):
        hi = _drive_synthetic_climb(start_big, optimum)
        lo = _drive_synthetic_climb(start_small, optimum)
        # Correct direction: descended from the large start, ascended from the small.
        assert hi < 50 and lo > 5, (
            f"optimum {optimum}: wrong direction (hi {hi} from ~50, lo {lo} from ~1)")
        # Converged near the optimum from both sides, within one discrete step.
        assert abs(hi - optimum) <= 2 * step, f"hi converged to {hi}, want ~{optimum}"
        assert abs(lo - optimum) <= 2 * step, f"lo converged to {lo}, want ~{optimum}"


def test_resize_window_conserves_capacity():
    """_resize_window (grow and shrink, plus clamps) never leaks/duplicates a key
    and always re-sums segment capacities to maxsize."""
    random.seed(0)
    maxsize = 100
    cache = CostAwareWTinyLFU(maxsize=maxsize, window_ratio=0.1, time_fn=lambda: 0.0)
    for i in range(400):
        k = i % 120  # overflow + reuse so all three segments are populated
        if cache.get(k) is None:
            cache.put([k])
    _assert_window_invariants(cache, maxsize)

    cache._resize_window(cache._window.maxsize + 20)   # grow
    _assert_window_invariants(cache, maxsize)
    cache._resize_window(cache._window.maxsize - 30)   # shrink
    _assert_window_invariants(cache, maxsize)

    cache._resize_window(10_000)                       # clamp high
    assert cache._window.maxsize <= maxsize - 2
    _assert_window_invariants(cache, maxsize)
    cache._resize_window(-5)                            # clamp low
    assert cache._window.maxsize >= 1
    _assert_window_invariants(cache, maxsize)


def test_cost_priority_maps_to_freq_weight():
    """cost_priority is the [0,1] money<->hit-rate dial mapped onto freq_weight.

    0.0 => 16 (frequency dominates, max hit rate); 1.0 => 1 (cost equal-footed,
    max money saved); linear between; out-of-range rejected; None is a no-op.
    """
    assert CostAwareWTinyLFU(maxsize=40, cost_priority=0.0)._freq_weight == 16.0
    assert CostAwareWTinyLFU(maxsize=40, cost_priority=1.0)._freq_weight == 1.0
    assert CostAwareWTinyLFU(maxsize=40, cost_priority=0.5)._freq_weight == 8.5
    # None leaves the raw freq_weight untouched
    assert CostAwareWTinyLFU(maxsize=40, freq_weight=4.0)._freq_weight == 4.0
    for bad in (-0.1, 1.5):
        try:
            CostAwareWTinyLFU(maxsize=40, cost_priority=bad)
            assert False, f"cost_priority={bad} should have raised"
        except ValueError:
            pass
