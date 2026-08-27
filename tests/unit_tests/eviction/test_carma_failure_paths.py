"""Adversarial integrity tests for CARMA and its storage integration."""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.manager.eviction.carma import ClusterAdaptiveEviction
from gptcache.manager.eviction_manager import EvictionManager
from gptcache.manager.factory import manager_factory
from gptcache.manager.scalar_data.base import CacheData
from gptcache.utils.error import ParamError


def _vector(*values):
    return np.asarray(values, dtype=np.float32)


def _policy(callback, **overrides):
    params = {
        "maxsize": 2,
        "clean_size": 1,
        "on_evict": callback,
        "topic_threshold": 0.70,
        "cell_threshold": 0.95,
        "demand_half_life": float("inf"),
        "max_topics": 4,
        "max_cells": 4,
        "admission_enabled": False,
        "quota_enabled": False,
    }
    params.update(overrides)
    return ClusterAdaptiveEviction(**params)


def test_callback_failure_rolls_back_and_marks_policy_unhealthy():
    def fail(_keys):
        raise RuntimeError("storage failed")

    policy = _policy(fail, maxsize=1, max_cells=2)
    policy.put_with_metadata([1], [_vector(1, 0)])

    with pytest.raises(RuntimeError, match="storage failed"):
        policy.put_with_metadata([2], [_vector(0, 1)])

    assert set(policy.snapshot()["entries"]) == {1}
    assert policy.stats()["healthy"] is False
    assert policy.stats()["callback_failures"] == 1
    with pytest.raises(RuntimeError, match="unhealthy"):
        policy.get(1)


def test_late_unhashable_batch_id_is_rejected_before_any_transition():
    evicted = []
    policy = _policy(lambda keys: evicted.extend(keys), maxsize=1, max_cells=2)
    policy.put_with_metadata([1], [_vector(1, 0)])

    with pytest.raises(ValueError, match="hashable"):
        policy.put_with_metadata([2, []], [_vector(0, 1), _vector(1, 1)])

    assert set(policy.snapshot()["entries"]) == {1}
    assert evicted == []


def test_distinct_same_repr_ids_are_not_collapsed_in_callback():
    class SameRepr:
        def __repr__(self):
            return "same"

    first, second, third, fourth = (SameRepr() for _ in range(4))
    callbacks = []
    policy = _policy(lambda keys: callbacks.extend(keys))
    policy.put_with_metadata(
        [first, second],
        [_vector(1, 0), _vector(0, 1)],
    )
    policy.put_with_metadata(
        [third, fourth],
        [_vector(-1, 0), _vector(0, -1)],
    )

    assert len(callbacks) == 2
    assert callbacks[0] is first
    assert callbacks[1] is second


def test_batch_callback_never_deletes_an_id_resident_in_final_state():
    callbacks = []
    policy = _policy(lambda keys: callbacks.extend(keys), maxsize=1, max_cells=2)
    policy.put_with_metadata([1], [_vector(1, 0)])

    policy.put_with_metadata(
        [2, 1],
        [_vector(0, 1), _vector(1, 0)],
    )

    assert policy.snapshot()["entry_ids"] == [1]
    assert callbacks == [2]


def test_full_cell_bound_never_cross_assigns_topic_ownership():
    callbacks = []
    policy = _policy(
        lambda keys: callbacks.extend(keys),
        maxsize=2,
        max_topics=3,
        max_cells=2,
        admission_enabled=True,
        quota_enabled=False,
    )
    policy.put_with_metadata([1, 2], [_vector(1, 0), _vector(0, 1)])
    outcome = policy.put_with_metadata([3], [_vector(-1, 0)])[0]

    assert outcome["action"] == "reject_cell_capacity"
    snapshot = policy.snapshot()
    for entry in snapshot["entries"].values():
        if entry["cell_id"] is not None:
            assert (
                snapshot["cells"][entry["cell_id"]]["topic_id"]
                == entry["topic_id"]
            )


def test_restore_over_capacity_uses_stored_lru_not_semantic_support():
    evicted = []
    policy = _policy(lambda keys: evicted.extend(keys), maxsize=2, max_cells=4)
    base = datetime(2026, 1, 1)
    policy.restore(
        [1, 2, 3],
        [_vector(1, 0), _vector(0, 1), _vector(1, 0)],
        [base, base + timedelta(seconds=1), base + timedelta(seconds=2)],
    )

    assert set(policy.snapshot()["entries"]) == {2, 3}
    assert evicted == [1]


def test_admission_disabled_does_not_reject_zero_quota_candidate():
    evicted = []
    policy = _policy(
        lambda keys: evicted.extend(keys),
        maxsize=1,
        max_cells=2,
        quota_enabled=True,
        admission_enabled=False,
    )
    policy.put_with_metadata([1], [_vector(1, 0)])
    outcome = policy.put_with_metadata([2], [_vector(0, 1)])[0]

    assert outcome["admitted"] is True
    assert set(policy.snapshot()["entries"]) == {2}
    assert evicted == [1]


def test_wrong_vector_dimension_is_rejected_before_scalar_persistence():
    with TemporaryDirectory() as root:
        scalar = CacheBase(
            "sqlite", sql_url="sqlite:///" + str(Path(root) / "sqlite.db")
        )
        vector = VectorBase(
            "faiss",
            dimension=2,
            index_path=str(Path(root) / "faiss.index"),
        )
        manager = get_data_manager(
            scalar,
            vector,
            max_size=2,
            clean_size=1,
            eviction="CARMA",
        )
        with pytest.raises(ParamError, match="dimension"):
            manager.save("bad", "answer", _vector(1, 0, 0))
        assert manager.s.count(is_all=True) == 0
        assert manager.v.count() == 0
        assert manager.eviction_base.stats()["size"] == 0
        manager.close()


def test_restart_reconstruction_does_not_touch_persisted_last_access():
    with TemporaryDirectory() as root:
        params = {
            "eviction": "CARMA",
            "max_size": 2,
            "clean_size": 1,
        }
        manager = manager_factory(
            "sqlite,faiss",
            data_dir=root,
            vector_params={"dimension": 2},
            eviction_params=params,
        )
        manager.save("question", "answer", _vector(1, 0))
        cache_id = manager.s.get_ids(deleted=False)[0]
        before = manager.s.peek_data_by_id(cache_id).last_access
        manager.close()

        reopened = manager_factory(
            "sqlite,faiss",
            data_dir=root,
            vector_params={"dimension": 2},
            eviction_params=params,
        )
        after = reopened.s.peek_data_by_id(cache_id).last_access
        assert after == before
        reopened.close()


def test_vector_delete_failure_keeps_scalar_rows_recoverable():
    events = []

    class Scalar:
        def get_ids(self, deleted=True):
            assert deleted is True
            events.append("get_ids")
            return [7]

        def clear_deleted_data(self):
            events.append("clear")

    class Vector:
        def delete(self, ids):
            events.append(("vector", list(ids)))
            raise RuntimeError("delete failed")

    manager = EvictionManager(Scalar(), Vector())
    with pytest.raises(RuntimeError, match="delete failed"):
        manager.delete()
    assert events == ["get_ids", ("vector", [7])]


def test_transient_vector_failure_recovers_without_clearing_unrelated_tombstones():
    with TemporaryDirectory() as root:
        manager = manager_factory(
            "sqlite,faiss",
            data_dir=root,
            vector_params={"dimension": 2, "top_k": 2},
            eviction_params={
                "eviction": "CARMA",
                "max_size": 1,
                "clean_size": 1,
                "policy_params": {
                    "admission_enabled": False,
                    "quota_enabled": False,
                    "max_topics": 2,
                    "max_cells": 2,
                },
            },
        )
        manager.save("old", "answer", _vector(1, 0))
        old_id = manager.s.get_ids(deleted=False)[0]

        unrelated_id = manager.s.batch_insert(
            [CacheData("unrelated", "answer", embedding_data=_vector(-1, 0))]
        )[0]
        manager.s.mark_deleted([unrelated_id])

        original_delete = manager.v.delete
        delete_calls = []

        def fail_once(ids):
            delete_calls.append(list(ids))
            if len(delete_calls) == 1:
                raise RuntimeError("transient vector failure")
            return original_delete(ids)

        manager.v.delete = fail_once
        with pytest.raises(RuntimeError, match="transient vector failure"):
            manager.save("new", "answer", _vector(0, 1))

        assert set(delete_calls[0]) == {unrelated_id, old_id}
        assert old_id in delete_calls[1]
        assert unrelated_id not in delete_calls[1]
        assert len(delete_calls[1]) == 2
        assert manager.s.get_ids(deleted=False) == []
        assert manager.s.get_ids(deleted=True) == [unrelated_id]
        assert manager.v.count() == 0
        assert manager.eviction_base.stats()["healthy"] is True
        assert manager.eviction_base.stats()["size"] == 0
        manager.close()


def test_persistent_vector_failure_keeps_policy_unhealthy_and_scalar_recoverable():
    with TemporaryDirectory() as root:
        manager = manager_factory(
            "sqlite,faiss",
            data_dir=root,
            vector_params={"dimension": 2, "top_k": 2},
            eviction_params={
                "eviction": "CARMA",
                "max_size": 1,
                "clean_size": 1,
                "policy_params": {
                    "admission_enabled": False,
                    "quota_enabled": False,
                    "max_topics": 2,
                    "max_cells": 2,
                },
            },
        )
        manager.save("old", "answer", _vector(1, 0))

        def fail_delete(_ids):
            raise RuntimeError("vector unavailable")

        manager.v.delete = fail_delete
        with pytest.raises(RuntimeError, match="vector unavailable"):
            manager.save("new", "answer", _vector(0, 1))

        assert manager.s.get_ids(deleted=False) == []
        assert len(manager.s.get_ids(deleted=True)) == 2
        assert manager.s.count(is_all=True) == 2
        assert manager.v.count() == 2
        assert manager.eviction_base.stats()["healthy"] is False
        with pytest.raises(RuntimeError, match="unhealthy"):
            manager.eviction_base.get(1)
        manager.close()
