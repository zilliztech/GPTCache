"""SQLite/FAISS integration checks for CARMA's storage cleanup path."""

from tempfile import TemporaryDirectory

import numpy as np

from gptcache.manager.factory import manager_factory


def _vector(*values):
    return np.asarray(values, dtype=np.float32)


def _manager(root):
    return manager_factory(
        "sqlite,faiss",
        data_dir=root,
        vector_params={"dimension": 2, "top_k": 1},
        eviction_params={
            "eviction": "CARMA",
            "max_size": 1,
            "clean_size": 1,
            "policy_params": {
                "topic_threshold": 0.0,
                "cell_threshold": 0.99,
                "demand_half_life": float("inf"),
                "quota_strength": 0.5,
                "admission_margin": 1.05,
                "centroid_alpha": 0.05,
                "entry_hit_weight": 0.25,
                "max_topics": 1,
                "max_cells": 4,
                "admission_enabled": True,
                "quota_enabled": False,
                "one_topic": True,
            },
        },
    )


def _assert_no_soft_deleted_rows(manager):
    assert manager.s.get_ids(deleted=True) == []
    assert manager.s.count(state=-1) == 0
    assert manager.s.count(is_all=True) == manager.s.count()


def test_carma_import_preserves_vector_store_keyword_arguments():
    with TemporaryDirectory() as root:
        manager = _manager(root)
        original_mul_add = manager.v.mul_add
        observed = {}

        def capture_mul_add(datas, **kwargs):
            observed.update(kwargs)
            return original_mul_add(datas)

        manager.v.mul_add = capture_mul_add
        manager.import_data(
            ["question"],
            ["answer"],
            [_vector(1, 0)],
            [None],
            caller_marker="preserved",
        )

        assert observed == {"caller_marker": "preserved"}
        manager.close()


def test_factory_wires_carma_and_physically_deletes_rejections_and_victims():
    with TemporaryDirectory() as root:
        manager = _manager(root)
        assert manager.eviction_base.policy == "CARMA"
        assert manager.eviction_base.requires_immediate_cleanup is True

        manager.save("incumbent", "answer-1", _vector(1, 0))
        first_id = manager.s.get_ids(deleted=False)[0]
        assert manager.s.count() == 1
        assert manager.v.count() == 1

        # At capacity, a first observation in a new cell is rejected.  Its row
        # and vector must be deleted immediately, not left soft-deleted where
        # FAISS top_k=1 could return it.
        manager.save("first-candidate", "answer-2", _vector(0, 1))
        assert manager.s.get_ids(deleted=False) == [first_id]
        assert manager.s.count() == 1
        assert manager.v.count() == 1
        _assert_no_soft_deleted_rows(manager)

        # The ghost cell records that miss.  The second occurrence should be
        # admitted and the old incumbent must disappear from both stores.
        manager.save("second-candidate", "answer-3", _vector(0, 1))
        remaining_ids = manager.s.get_ids(deleted=False)
        assert len(remaining_ids) == 1
        assert remaining_ids[0] != first_id
        assert manager.s.get_data_by_id(first_id) is None
        assert manager.s.get_data_by_id(remaining_ids[0]).question == "second-candidate"
        assert manager.s.count() == 1
        assert manager.v.count() == 1
        _assert_no_soft_deleted_rows(manager)

        search_result = manager.search(_vector(0, 1), top_k=1)
        assert search_result[0][1] == remaining_ids[0]
        manager.close()


def test_factory_restart_restores_embeddings_without_manufacturing_demand():
    with TemporaryDirectory() as root:
        first = manager_factory(
            "sqlite,faiss",
            data_dir=root,
            vector_params={"dimension": 2, "top_k": 1},
            eviction_params={
                "eviction": "CARMA",
                "max_size": 3,
                "clean_size": 1,
                "policy_params": {
                    "max_topics": 3,
                    "max_cells": 6,
                    "quota_enabled": False,
                },
            },
        )
        first.save("east", "answer-east", _vector(1, 0))
        first.save("north", "answer-north", _vector(0, 1))
        first.save("west", "answer-west", _vector(-1, 0))
        original_ids = first.s.get_ids(deleted=False)
        first.close()

        reopened = manager_factory(
            "sqlite,faiss",
            data_dir=root,
            vector_params={"dimension": 2, "top_k": 1},
            eviction_params={
                "eviction": "CARMA",
                "max_size": 3,
                "clean_size": 1,
                "policy_params": {
                    "max_topics": 3,
                    "max_cells": 6,
                    "quota_enabled": False,
                },
            },
        )
        snapshot = reopened.eviction_base.snapshot()

        assert snapshot["tick"] == 0
        assert set(snapshot["entries"]) == set(original_ids)
        assert all(entry["hit_mass"] == 0 for entry in snapshot["entries"].values())
        assert all(topic["demand"] == 0 for topic in snapshot["topics"].values())
        assert all(topic["miss_mass"] == 0 for topic in snapshot["topics"].values())
        assert reopened.eviction_base.stats()["restores"] == 3
        assert reopened.s.count() == 3
        assert reopened.v.count() == 3

        hit = reopened.search(_vector(1, 0), top_k=1)[0]
        reopened.hit_cache_callback(hit)
        assert reopened.eviction_base.stats()["hits"] == 1
        assert reopened.get_scalar_data(hit).question == "east"
        reopened.close()
