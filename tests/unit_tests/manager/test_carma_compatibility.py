"""Compatibility checks for CARMA's extensions to GPTCache interfaces."""

from datetime import datetime

import numpy as np
import pytest

from gptcache.manager.data_manager import normalize
from gptcache.manager.eviction.base import EvictionBase as EvictionInterface
from gptcache.manager.eviction.carma import ClusterAdaptiveEviction
from gptcache.manager.eviction.manager import EvictionBase as EvictionFactory
from gptcache.manager.scalar_data.base import CacheStorage
from gptcache.utils.error import ParamError


class _LegacyEviction(EvictionInterface):
    def __init__(self):
        self.values = []

    def put(self, objs):
        self.values.extend(objs)
        return list(objs)

    def get(self, obj):
        return obj in self.values

    @property
    def policy(self):
        return "LEGACY"


class _LegacyStorage(CacheStorage):
    def __init__(self):
        self.value = object()

    def create(self):
        return None

    def batch_insert(self, all_data):
        return list(range(len(all_data)))

    def get_data_by_id(self, key):
        del key
        return self.value

    def mark_deleted(self, keys):
        del keys

    def clear_deleted_data(self):
        return None

    def get_ids(self, deleted=True):
        del deleted
        return []

    def count(self, state=0, is_all=False):
        del state, is_all
        return 0

    def add_session(self, question_id, session_id, session_question):
        del question_id, session_id, session_question

    def list_sessions(self, session_id=None, key=None):
        del session_id, key
        return []

    def delete_session(self, session_id):
        del session_id

    def report_cache(self, user_question, cache_question, cache_question_id, cache_answer):
        del user_question, cache_question, cache_question_id, cache_answer

    def close(self):
        return None


def test_default_metadata_hooks_preserve_third_party_eviction_behavior():
    policy = _LegacyEviction()

    assert policy.put_with_metadata([1], embeddings=[np.ones(2)]) == [1]
    assert policy.restore([2], last_accesses=[datetime(2026, 1, 1)]) == [2]
    assert policy.values == [1, 2]
    assert policy.requires_immediate_cleanup is False
    assert policy.requires_embedding_restore is False
    assert policy.accepts_embedding_metadata is False


def test_default_scalar_hooks_preserve_third_party_storage_behavior():
    storage = _LegacyStorage()

    assert storage.peek_data_by_id(7) is storage.value
    assert storage.clear_deleted_data_by_ids([7]) is False


@pytest.mark.parametrize(
    "value, message",
    [
        (object(), "finite one-dimensional"),
        ([1.0, np.nan], "finite one-dimensional"),
        ([0.0, 0.0], "positive finite norm"),
    ],
)
def test_normalize_rejects_malformed_embeddings(value, message):
    with pytest.raises(ParamError, match=message):
        normalize(value)


def test_eviction_factory_keeps_lru_and_dispatches_carma_exactly():
    callback = lambda _keys: None

    with pytest.raises(ValueError, match="positive integer"):
        EvictionFactory.get("memory", maxsize=0, on_evict=callback)

    carma = EvictionFactory.get(
        "memory",
        policy="carma",
        maxsize=3,
        clean_size=0,
        on_evict=callback,
    )
    lru = EvictionFactory.get(
        "memory",
        policy="LRU",
        maxsize=3,
        clean_size=1,
        on_evict=callback,
    )

    assert isinstance(carma, ClusterAdaptiveEviction)
    assert carma.clean_size == 1
    assert lru.policy == "LRU"


def test_carma_defaults_unhashable_lookup_and_opaque_replacement():
    evicted = []
    policy = ClusterAdaptiveEviction(
        maxsize=1,
        clean_size=None,
        on_evict=lambda keys: evicted.extend(keys),
        admission_enabled=False,
        quota_enabled=False,
    )

    assert policy.demand_half_life == 10
    assert policy.get([]) is None
    policy.put(["first"])
    outcome = policy.put(["second"])[0]

    assert outcome["action"] == "replace_unknown"
    assert outcome["admitted"] is True
    assert policy.snapshot()["entry_ids"] == ["second"]
    assert evicted == ["first"]


def test_carma_requires_callback_and_rejects_uncoercible_embedding():
    with pytest.raises(ValueError, match="callable on_evict"):
        ClusterAdaptiveEviction(maxsize=1)

    evicted = []
    policy = ClusterAdaptiveEviction(
        maxsize=1,
        on_evict=lambda keys: evicted.extend(keys),
    )
    outcome = policy.put_with_metadata([1], [object()])[0]

    assert outcome["action"] == "reject_invalid"
    assert policy.snapshot()["entry_ids"] == []
    assert evicted == [1]


def test_restore_accepts_datetime_numeric_and_fallback_timestamps():
    policy = ClusterAdaptiveEviction(
        maxsize=3,
        on_evict=lambda _keys: None,
        admission_enabled=False,
        quota_enabled=False,
    )
    policy.restore(
        ["datetime", "numeric", "fallback"],
        [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0])],
        [datetime(2026, 1, 1), 7, "unknown"],
    )

    assert set(policy.snapshot()["entry_ids"]) == {
        "datetime",
        "numeric",
        "fallback",
    }
