"""Phase 2 tests: CA_W_TINYLFU reachable via the EvictionBase public factory.

Verifies that MemoryCacheEviction correctly routes the CA_W_TINYLFU policy
and that the put/get contract holds end-to-end through the factory. Also
confirms existing LRU/LFU/FIFO/RR policies are unaffected.
"""

import pytest

from gptcache.manager.eviction.manager import EvictionBase


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------

def test_factory_creates_ca_w_tinylfu():
    """EvictionBase.get routes 'CA_W_TINYLFU' without raising."""
    e = EvictionBase.get(
        name="memory",
        policy="CA_W_TINYLFU",
        maxsize=100,
        on_evict=lambda ks: None,
    )
    assert e.policy == "CA_W_TINYLFU"


def test_ca_w_tinylfu_put_get_contract():
    """put then get returns truthy; get on missing key returns None."""
    e = EvictionBase.get(
        name="memory",
        policy="CA_W_TINYLFU",
        maxsize=100,
        on_evict=lambda ks: None,
    )
    e.put([1, 2, 3])
    assert e.get(1) is not None
    assert e.get(2) is not None
    assert e.get(3) is not None
    assert e.get(999) is None


def test_ca_w_tinylfu_eviction_callback_fires():
    """on_evict callback fires with evicted keys when cache overflows."""
    evicted = []
    e = EvictionBase.get(
        name="memory",
        policy="CA_W_TINYLFU",
        maxsize=10,
        on_evict=lambda ks: evicted.extend(ks),
    )
    for i in range(50):
        e.put([i])

    assert len(evicted) >= 1, "on_evict never fired despite overflow"
    for k in evicted:
        assert e.get(k) is None, f"evicted key {k} still returned on get"


def test_ca_w_tinylfu_kwargs_forwarded():
    """Constructor kwargs (decay_rate, ewma_alpha) are forwarded correctly."""
    e = EvictionBase.get(
        name="memory",
        policy="CA_W_TINYLFU",
        maxsize=50,
        on_evict=lambda ks: None,
        decay_rate=1e-3,
        ewma_alpha=0.1,
        ewma_warmup=5,
    )
    assert e.policy == "CA_W_TINYLFU"
    assert e._cache._decay_rate == 1e-3
    assert e._cache._cost_tracker._alpha == 0.1


# ---------------------------------------------------------------------------
# Regression: existing policies unaffected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy", ["LRU", "LFU", "FIFO", "RR"])
def test_existing_policies_still_work(policy):
    """Existing policies route correctly and are not broken by the new branch."""
    evicted = []
    e = EvictionBase.get(
        name="memory",
        policy=policy,
        maxsize=5,
        clean_size=2,
        on_evict=lambda ks: evicted.extend(ks),
    )
    for i in range(10):
        e.put([i])

    assert e.policy == policy
    assert len(evicted) >= 1, f"{policy}: no evictions despite overflow"


def test_unknown_policy_still_raises():
    """An unrecognised policy name still raises ValueError."""
    with pytest.raises(ValueError):
        EvictionBase.get(
            name="memory",
            policy="NONEXISTENT",
            maxsize=10,
            on_evict=lambda ks: None,
        )
