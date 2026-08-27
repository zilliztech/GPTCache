"""Focused behavioral tests for the CARMA eviction policy.

The tests intentionally use only CARMA's public methods.  Embeddings are tiny,
hand-written vectors so clustering and tie decisions remain deterministic.
"""

import math

import numpy as np
import pytest

from gptcache.manager.eviction.carma import ClusterAdaptiveEviction


def _vector(*values):
    return np.asarray(values, dtype=np.float32)


def _policy(evicted, **overrides):
    params = {
        "maxsize": 4,
        "clean_size": 1,
        "on_evict": lambda keys: evicted.append(list(keys)),
        "topic_threshold": 0.70,
        "cell_threshold": 0.95,
        "demand_half_life": math.inf,
        "quota_strength": 0.5,
        "admission_margin": 1.05,
        "centroid_alpha": 0.05,
        "entry_hit_weight": 0.25,
        "max_topics": 4,
        "max_cells": 8,
        "admission_enabled": True,
        "quota_enabled": True,
        "one_topic": False,
    }
    params.update(overrides)
    return ClusterAdaptiveEviction(**params)


def _entry_ids(policy):
    return set(policy.snapshot()["entries"])


def test_policy_identity_capacity_callback_and_stats():
    evicted = []
    policy = _policy(
        evicted,
        maxsize=2,
        max_topics=1,
        max_cells=4,
        one_topic=True,
        quota_enabled=False,
        admission_enabled=False,
    )

    policy.put_with_metadata([1, 2], [_vector(1, 0), _vector(0, 1)])
    assert policy.get(1) is not None
    policy.put_with_metadata([3], [_vector(1, 1)])

    assert policy.policy == "CARMA"
    assert policy.requires_immediate_cleanup is True
    assert _entry_ids(policy) == {1, 3}
    assert evicted == [[2]]

    stats = policy.stats()
    assert stats["size"] == 2
    assert stats["hits"] == 1
    assert stats["misses"] == 3
    assert stats["evictions"] == 1
    assert stats["rejections"] == 0


def test_second_occurrence_uses_ghost_evidence_to_win_admission():
    evicted = []
    policy = _policy(
        evicted,
        maxsize=1,
        max_topics=1,
        max_cells=4,
        one_topic=True,
        quota_enabled=False,
    )

    policy.put_with_metadata([1], [_vector(1, 0)])

    # The first observation of a new semantic cell is rejected at capacity.
    policy.put_with_metadata([2], [_vector(0, 1)])
    assert _entry_ids(policy) == {1}
    assert evicted == [[2]]
    assert policy.stats()["rejections"] == 1

    # Its ghost cell retained support, so a second observation is admissible.
    policy.put_with_metadata([3], [_vector(0, 1)])
    assert _entry_ids(policy) == {3}
    assert evicted == [[2], [1]]
    assert policy.stats()["admissions"] == 2


def test_logical_tick_decay_halves_hit_mass_after_two_events():
    evicted = []
    policy = _policy(
        evicted,
        maxsize=4,
        max_topics=1,
        max_cells=8,
        one_topic=True,
        quota_enabled=False,
        admission_enabled=False,
        demand_half_life=2,
    )

    policy.put_with_metadata([1], [_vector(1, 0)])
    policy.get(1)
    policy.put_with_metadata([2], [_vector(0, 1)])
    policy.put_with_metadata([3], [_vector(-1, 0)])

    snapshot = policy.snapshot()
    assert snapshot["tick"] == 4
    assert snapshot["entries"][1]["hit_mass"] == pytest.approx(0.5)


def test_clustering_and_equal_similarity_tie_are_deterministic():
    evicted = []
    policy = _policy(
        evicted,
        maxsize=5,
        topic_threshold=0.70,
        cell_threshold=0.95,
        max_topics=5,
        max_cells=10,
        quota_enabled=False,
        admission_enabled=False,
    )

    # The first two vectors create orthogonal topics.  Vector 30 is equally
    # similar to both and must choose the lower topic ID.  Vector 40 joins the
    # first vector's semantic cell.
    policy.put_with_metadata([10], [_vector(1, 0)])
    policy.put_with_metadata([20], [_vector(0, 1)])
    policy.put_with_metadata([30], [_vector(1, 1)])
    policy.put_with_metadata([40], [_vector(1, 0.01)])

    entries = policy.snapshot()["entries"]
    assert entries[10]["topic_id"] != entries[20]["topic_id"]
    assert entries[30]["topic_id"] == entries[10]["topic_id"]
    assert entries[40]["topic_id"] == entries[10]["topic_id"]
    assert entries[40]["cell_id"] == entries[10]["cell_id"]
    assert entries[30]["cell_id"] != entries[10]["cell_id"]
    assert evicted == []


@pytest.mark.parametrize(
    "bad_embedding",
    [
        None,
        _vector(),
        _vector(0, 0),
        _vector(np.nan, 0),
        _vector(1, 0, 0),
    ],
)
def test_invalid_vector_is_rejected_safely_at_capacity(bad_embedding):
    evicted = []
    policy = _policy(
        evicted,
        maxsize=1,
        max_topics=1,
        max_cells=2,
        one_topic=True,
        quota_enabled=False,
    )
    policy.put_with_metadata([1], [_vector(1, 0)])

    policy.put_with_metadata([2], [bad_embedding])

    assert _entry_ids(policy) == {1}
    assert evicted == [[2]]


def test_cell_history_is_bounded_even_under_unique_scan():
    evicted = []
    policy = _policy(
        evicted,
        maxsize=1,
        max_topics=1,
        max_cells=3,
        one_topic=True,
        cell_threshold=0.9999,
        quota_enabled=False,
    )
    policy.put_with_metadata([0], [_vector(1, 0)])

    for key in range(1, 21):
        angle = key * 0.13
        policy.put_with_metadata(
            [key], [_vector(math.cos(angle), math.sin(angle))]
        )

    snapshot = policy.snapshot()
    assert len(snapshot["cells"]) <= 3
    assert len(snapshot["topics"]) <= 1
    assert len(snapshot["entries"]) <= 1


def test_custom_id_order_registry_is_bounded_by_residents():
    evicted = []
    policy = _policy(
        evicted,
        maxsize=1,
        max_topics=1,
        max_cells=1,
        one_topic=True,
        quota_enabled=False,
        admission_enabled=False,
    )

    for index in range(1000):
        policy.put_with_metadata([("opaque", index)], [_vector(1, 0)])
        stats = policy.stats()
        assert stats["id_ordinals"] <= stats["size"] <= policy.maxsize

    assert policy.stats()["id_ordinals"] == 1
    assert _entry_ids(policy) == {("opaque", 999)}


def test_below_threshold_candidate_is_rejected_when_cell_bound_is_full():
    evicted = []
    policy = _policy(
        evicted,
        maxsize=1,
        max_topics=1,
        max_cells=1,
        one_topic=True,
        cell_threshold=0.95,
        quota_enabled=False,
        admission_margin=1.0,
    )
    policy.put_with_metadata([1], [_vector(1, 0)])

    policy.put_with_metadata([2], [_vector(0, 1)])

    assert _entry_ids(policy) == {1}
    assert evicted == [[2]]
    assert policy.snapshot()["last_event"]["action"] == "reject_cell_capacity"


def test_batch_alignment_error_is_atomic():
    evicted = []
    policy = _policy(evicted, maxsize=3)

    with pytest.raises(ValueError, match="same length|align|embedding"):
        policy.put_with_metadata([10, 20], [_vector(1, 0)])

    assert _entry_ids(policy) == set()
    assert evicted == []
    assert policy.stats()["misses"] == 0


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"maxsize": 0}, "maxsize"),
        ({"clean_size": 0}, "clean_size"),
        ({"topic_threshold": 1.1}, "topic_threshold"),
        ({"cell_threshold": 0.60}, "cell_threshold"),
        ({"demand_half_life": 0}, "demand_half_life"),
        ({"quota_strength": -1}, "quota_strength"),
        ({"admission_margin": math.inf}, "admission_margin"),
        ({"ghost_support_threshold": 1}, "ghost_support_threshold"),
        ({"centroid_alpha": 0}, "centroid_alpha"),
        ({"entry_hit_weight": -1}, "entry_hit_weight"),
        ({"max_topics": 0}, "max_topics"),
        ({"max_cells": 3}, "max_cells"),
    ],
)
def test_invalid_configuration_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        _policy([], **overrides)


def test_batch_preserves_id_to_embedding_alignment():
    evicted = []
    policy = _policy(
        evicted,
        maxsize=4,
        topic_threshold=0.80,
        cell_threshold=0.95,
        quota_enabled=False,
        admission_enabled=False,
    )
    policy.put_with_metadata(
        [10, 20],
        [_vector(1, 0), _vector(0, 1)],
    )
    policy.put_with_metadata([30], [_vector(0.01, 1)])

    entries = policy.snapshot()["entries"]
    assert entries[10]["topic_id"] != entries[20]["topic_id"]
    assert entries[30]["topic_id"] == entries[20]["topic_id"]
    assert entries[30]["cell_id"] == entries[20]["cell_id"]


def test_restore_rebuilds_structure_without_live_demand():
    evicted = []
    policy = _policy(evicted, maxsize=4)

    policy.restore([7, 8], [_vector(1, 0), _vector(0, 1)])

    snapshot = policy.snapshot()
    assert snapshot["tick"] == 0
    assert set(snapshot["entries"]) == {7, 8}
    assert all(entry["hit_mass"] == 0 for entry in snapshot["entries"].values())
    assert all(topic["demand"] == 0 for topic in snapshot["topics"].values())
    assert all(topic["miss_mass"] == 0 for topic in snapshot["topics"].values())
    assert policy.stats()["hits"] == 0
    assert policy.stats()["misses"] == 0
    assert evicted == []


def test_legacy_put_accepts_opaque_ids_and_rejects_unknown_at_capacity():
    evicted = []
    policy = _policy(
        evicted,
        maxsize=2,
        max_topics=1,
        max_cells=2,
        one_topic=True,
        quota_enabled=False,
    )

    policy.put([1, 2])
    assert _entry_ids(policy) == {1, 2}
    assert policy.get(1) is not None

    policy.put([3])
    assert _entry_ids(policy) == {1, 2}
    assert evicted == [[3]]


def test_quota_sum_is_integer_capacity_and_under_quota_topic_uses_donor():
    evicted = []
    policy = _policy(
        evicted,
        maxsize=4,
        max_topics=4,
        max_cells=8,
        topic_threshold=0.80,
        cell_threshold=0.95,
        quota_enabled=True,
        admission_enabled=False,
    )

    # Topic A initially occupies three slots while topic B occupies one.  The
    # next B candidate makes the balanced quota 2/2, so A must donate a slot.
    policy.put_with_metadata([1, 2, 3], [_vector(1, 0)] * 3)
    policy.put_with_metadata([4], [_vector(0, 1)])
    policy.put_with_metadata([5], [_vector(0, 1)])

    snapshot = policy.snapshot()
    quotas = snapshot["quotas"]
    assert all(isinstance(value, int) and value >= 0 for value in quotas.values())
    assert sum(quotas.values()) == policy.maxsize
    assert _entry_ids(policy) == {2, 3, 4, 5}
    assert evicted == [[1]]

    topic_a = snapshot["entries"][2]["topic_id"]
    topic_b = snapshot["entries"][5]["topic_id"]
    assert topic_a != topic_b
    assert quotas[topic_a] == 2
    assert quotas[topic_b] == 2


def test_admission_margin_boundary_is_exact():
    admitted_evictions = []
    admitted = _policy(
        admitted_evictions,
        maxsize=1,
        max_topics=1,
        max_cells=2,
        one_topic=True,
        quota_enabled=False,
        admission_margin=1.0,
    )
    admitted.put_with_metadata([1], [_vector(1, 0)])
    admitted.put_with_metadata([2], [_vector(1, 0)])

    # Candidate and incumbent have equal post-update value.  Equality passes
    # when the required ratio is exactly one.
    assert _entry_ids(admitted) == {2}
    assert admitted_evictions == [[1]]

    rejected_evictions = []
    rejected = _policy(
        rejected_evictions,
        maxsize=1,
        max_topics=1,
        max_cells=2,
        one_topic=True,
        quota_enabled=False,
        admission_margin=1.0001,
    )
    rejected.put_with_metadata([1], [_vector(1, 0)])
    rejected.put_with_metadata([2], [_vector(1, 0)])

    assert _entry_ids(rejected) == {1}
    assert rejected_evictions == [[2]]
    assert rejected.snapshot()["last_event"]["action"] == "reject_margin"


def test_equal_value_victim_tie_uses_oldest_entry_deterministically():
    results = []
    for _ in range(5):
        evicted = []
        policy = _policy(
            evicted,
            maxsize=2,
            max_topics=1,
            max_cells=4,
            one_topic=True,
            quota_enabled=False,
            admission_enabled=False,
        )
        policy.put_with_metadata([20], [_vector(1, 0)])
        policy.put_with_metadata([10], [_vector(0, 1)])
        policy.put_with_metadata([30], [_vector(-1, 0)])
        results.append((evicted, sorted(_entry_ids(policy))))

    assert results == [([[20]], [10, 30])] * 5


def test_seeded_reference_model_matches_victim_decisions():
    rng = np.random.default_rng(20260826)
    evicted = []
    policy = _policy(
        evicted,
        maxsize=4,
        max_topics=1,
        max_cells=4,
        one_topic=True,
        quota_enabled=False,
        admission_enabled=False,
    )
    policy.put_with_metadata([0, 1, 2, 3], [_vector(1, 0)] * 4)

    for candidate in range(4, 24):
        resident_ids = sorted(_entry_ids(policy))
        for _ in range(int(rng.integers(0, 6))):
            policy.get(resident_ids[int(rng.integers(0, len(resident_ids)))])

        snapshot = policy.snapshot()
        entries = snapshot["entries"]
        only_cell = next(iter(snapshot["cells"].values()))
        support_share = only_cell["support"] / len(only_cell["residents"])
        expected = min(
            entries.values(),
            key=lambda entry: (
                support_share + policy.entry_hit_weight * entry["hit_mass"],
                entry["last_hit_tick"],
                entry["insert_tick"],
                entry["id"],
            ),
        )["id"]

        policy.put_with_metadata([candidate], [_vector(1, 0)])
        assert evicted[-1] == [expected]


def test_duplicate_put_and_unknown_get_do_not_advance_live_tick():
    evicted = []
    policy = _policy(evicted, maxsize=2)
    policy.put_with_metadata([1], [_vector(1, 0)])
    before = policy.snapshot()

    policy.put_with_metadata([1], [_vector(0, 1)])
    assert policy.get(999) is None
    after_noops = policy.snapshot()

    assert after_noops["tick"] == before["tick"]
    assert after_noops["entries"] == before["entries"]
    assert policy.stats()["duplicate_puts"] == 1
    assert policy.stats()["unknown_hits"] == 1
    assert policy.stats()["misses"] == 1
    assert evicted == []

    assert policy.get(1) is True
    assert policy.snapshot()["tick"] == before["tick"] + 1
    assert policy.stats()["hits"] == 1


@pytest.mark.parametrize("seed", [0, 7, 31])
def test_random_event_sequence_preserves_all_state_invariants(seed):
    rng = np.random.default_rng(seed)
    evicted = []
    policy = _policy(
        evicted,
        maxsize=7,
        max_topics=4,
        max_cells=14,
        topic_threshold=0.40,
        cell_threshold=0.90,
        demand_half_life=17,
    )
    next_key = 0

    for _ in range(400):
        if next_key and rng.random() < 0.30:
            policy.get(int(rng.integers(0, next_key + 4)))
        else:
            key = next_key
            next_key += 1
            vector = rng.normal(size=4).astype(np.float32)
            if key % 29 == 0:
                vector[:] = 0  # Exercise safe invalid-candidate rejection.
            policy.put_with_metadata([key], [vector])

        snapshot = policy.snapshot()
        entries = snapshot["entries"]
        topics = snapshot["topics"]
        cells = snapshot["cells"]

        assert snapshot["size"] == len(entries) <= policy.maxsize
        assert len(topics) <= policy.max_topics
        assert len(cells) <= policy.max_cells
        assert policy.stats()["size"] == len(entries)

        for key, entry in entries.items():
            assert key in topics[entry["topic_id"]]["residents"]
            assert key in cells[entry["cell_id"]]["residents"]
        for topic_id, topic in topics.items():
            for key in topic["residents"]:
                assert key in entries
                assert entries[key]["topic_id"] == topic_id
        for cell_id, cell in cells.items():
            assert cell["topic_id"] in topics
            assert cell_id in topics[cell["topic_id"]]["cell_ids"]
            for key in cell["residents"]:
                assert key in entries
                assert entries[key]["cell_id"] == cell_id

    assert all(keys and len(keys) == len(set(keys)) for keys in evicted)
