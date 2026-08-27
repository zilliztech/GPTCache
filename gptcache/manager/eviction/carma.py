"""CARMA: online cluster-adaptive admission and eviction.

The policy is intentionally self-contained and NumPy-only.  It learns bounded
semantic topics and cells from insertion-time embeddings while retaining the
legacy ``put(ids)`` / ``get(id)`` eviction interface.
"""

import copy
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from gptcache.manager.eviction.base import EvictionBase


@dataclass
class _EntryState:
    key: Any
    topic_id: Optional[int]
    cell_id: Optional[int]
    hit_mass: float
    stat_tick: int
    insert_tick: int
    last_hit_tick: int


@dataclass
class _TopicState:
    topic_id: int
    centroid: np.ndarray
    seen_count: int = 1
    residents: Set[Any] = field(default_factory=set)
    cell_ids: Set[int] = field(default_factory=set)
    demand: float = 0.0
    miss_mass: float = 0.0
    stat_tick: int = 0


@dataclass
class _CellState:
    cell_id: int
    topic_id: int
    centroid: np.ndarray
    seen_count: int = 1
    residents: Set[Any] = field(default_factory=set)
    support: float = 0.0
    stat_tick: int = 0
    last_seen_tick: int = 0


class ClusterAdaptiveEviction(EvictionBase):
    """Cluster-Adaptive Reuse and Miss-pressure Admission (CARMA).

    Args:
        maxsize: Maximum number of resident cache entries.
        clean_size: Accepted for compatibility. CARMA removes exactly the
            overflow required to preserve capacity.
        on_evict: Storage callback receiving row IDs to delete.
        topic_threshold: Minimum cosine similarity for a coarse topic match.
        cell_threshold: Minimum cosine similarity for a fine cell match.
        demand_half_life: Logical request events required to halve adaptive
            statistics. ``None`` means ``10 * maxsize``; infinity disables
            decay.
        quota_strength: Exponent applied to demand/miss-pressure quota weights.
        admission_margin: Candidate-to-victim value ratio required to admit.
        ghost_support_threshold: Decayed cell support required to challenge an
            incumbent. The default means two observations within one half-life.
        centroid_alpha: Exponential moving-average centroid step.
        entry_hit_weight: Contribution of entry hit mass to retention value.
        max_topics: Bound on coarse topics; defaults to ``min(64, maxsize)``.
        max_cells: Bound on fine cells including ghosts; defaults to
            ``2 * maxsize``.
        admission_enabled: Disable for the eviction-only ablation.
        quota_enabled: Disable for the admission-only ablation.
        one_topic: Force all valid embeddings into one coarse topic.
        seed: Recorded for experiment identity. The algorithm has no random
            tie-breaking.
    """

    _QUOTA_REFRESH_INTERVAL = 32
    _EPSILON = 1e-12

    def __init__(
        self,
        maxsize: int = 1000,
        clean_size: int = 1,
        on_evict: Optional[Callable[[List[Any]], None]] = None,
        topic_threshold: float = 0.70,
        cell_threshold: float = 0.88,
        demand_half_life: Optional[float] = None,
        quota_strength: float = 0.5,
        admission_margin: float = 1.05,
        ghost_support_threshold: float = 1.5,
        centroid_alpha: float = 0.05,
        entry_hit_weight: float = 0.25,
        max_topics: Optional[int] = None,
        max_cells: Optional[int] = None,
        admission_enabled: bool = True,
        quota_enabled: bool = True,
        one_topic: bool = False,
        seed: int = 0,
        **kwargs: Any,
    ):
        del kwargs
        if not isinstance(maxsize, int) or isinstance(maxsize, bool) or maxsize <= 0:
            raise ValueError("maxsize must be a positive integer")
        if clean_size is None:
            clean_size = 1
        if (
            not isinstance(clean_size, int)
            or isinstance(clean_size, bool)
            or clean_size < 1
            or clean_size > maxsize
        ):
            raise ValueError("clean_size must be between 1 and maxsize")
        if on_evict is None or not callable(on_evict):
            raise ValueError("CARMA requires a callable on_evict callback")
        self._validate_threshold("topic_threshold", topic_threshold)
        self._validate_threshold("cell_threshold", cell_threshold)
        if cell_threshold < topic_threshold:
            raise ValueError("cell_threshold must be at least topic_threshold")

        if demand_half_life is None:
            demand_half_life = float(10 * maxsize)
        demand_half_life = float(demand_half_life)
        if math.isnan(demand_half_life) or demand_half_life <= 0:
            raise ValueError("demand_half_life must be positive or infinity")
        if not math.isfinite(float(quota_strength)) or quota_strength < 0:
            raise ValueError("quota_strength must be finite and non-negative")
        if not math.isfinite(float(admission_margin)) or admission_margin < 0:
            raise ValueError("admission_margin must be finite and non-negative")
        if (
            not math.isfinite(float(ghost_support_threshold))
            or ghost_support_threshold <= 1
        ):
            raise ValueError("ghost_support_threshold must be finite and greater than 1")
        if (
            not math.isfinite(float(centroid_alpha))
            or centroid_alpha <= 0
            or centroid_alpha > 1
        ):
            raise ValueError("centroid_alpha must be in (0, 1]")
        if not math.isfinite(float(entry_hit_weight)) or entry_hit_weight < 0:
            raise ValueError("entry_hit_weight must be finite and non-negative")

        if max_topics is None:
            max_topics = max(1, min(64, maxsize))
        if max_cells is None:
            max_cells = max(1, 2 * maxsize)
        if not isinstance(max_topics, int) or max_topics < 1:
            raise ValueError("max_topics must be a positive integer")
        if not isinstance(max_cells, int) or max_cells < maxsize:
            raise ValueError("max_cells must be an integer at least maxsize")

        self.maxsize = maxsize
        self.clean_size = clean_size
        self.topic_threshold = float(topic_threshold)
        self.cell_threshold = float(cell_threshold)
        self.demand_half_life = demand_half_life
        self.quota_strength = float(quota_strength)
        self.admission_margin = float(admission_margin)
        self.ghost_support_threshold = float(ghost_support_threshold)
        self.centroid_alpha = float(centroid_alpha)
        self.entry_hit_weight = float(entry_hit_weight)
        self.max_topics = max_topics
        self.max_cells = max_cells
        self.admission_enabled = bool(admission_enabled)
        self.quota_enabled = bool(quota_enabled)
        self.one_topic = bool(one_topic)
        self.seed = int(seed)

        self._on_evict = on_evict
        self._operation_lock = threading.RLock()
        self._lock = threading.RLock()
        self._healthy = True
        self._failure_reason: Optional[str] = None
        self._tick = 0
        self._dimension: Optional[int] = None
        self._entries: Dict[Any, _EntryState] = {}
        self._topics: Dict[int, _TopicState] = {}
        self._cells: Dict[int, _CellState] = {}
        self._next_topic_id = 0
        self._next_cell_id = 0
        self._quotas: Dict[int, int] = {}
        self._last_quota_tick = -self._QUOTA_REFRESH_INTERVAL
        self._last_event: Optional[Dict[str, Any]] = None
        self._id_ordinals: Dict[Any, int] = {}
        self._next_id_ordinal = 0
        self._counters: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "admissions": 0,
            "rejections": 0,
            "evictions": 0,
            "restores": 0,
            "duplicate_puts": 0,
            "unknown_hits": 0,
            "invalid_embeddings": 0,
            "topics_created": 0,
            "cells_created": 0,
            "ghost_cells_pruned": 0,
            "quota_refreshes": 0,
            "topics_recycled": 0,
            "callback_failures": 0,
        }

    @staticmethod
    def _validate_threshold(name: str, value: float) -> None:
        if not math.isfinite(float(value)) or value < -1 or value > 1:
            raise ValueError("%s must be a finite cosine threshold in [-1, 1]" % name)

    @property
    def policy(self) -> str:
        return "CARMA"

    @property
    def requires_immediate_cleanup(self) -> bool:
        return True

    @property
    def requires_embedding_restore(self) -> bool:
        return True

    @property
    def accepts_embedding_metadata(self) -> bool:
        return True

    def put(self, objs: List[Any]):
        """Track legacy IDs without embedding metadata.

        This compatibility path admits unknown IDs during cold fill. Once full,
        an unknown live candidate is rejected unless admission is disabled.
        """

        return self.put_with_metadata(objs, embeddings=None)

    def put_with_metadata(
        self,
        objs: Iterable[Any],
        embeddings: Optional[Sequence[Any]] = None,
        restoring: bool = False,
        last_accesses: Optional[Sequence[Any]] = None,
    ) -> List[Dict[str, Any]]:
        objects = list(objs)
        vectors = self._aligned_optional("embeddings", embeddings, len(objects))
        accesses = self._aligned_optional(
            "last_accesses", last_accesses, len(objects)
        )
        explicit_embeddings = embeddings is not None

        # Validate the whole batch before the first state transition. Without
        # this gate, a bad later ID could suppress cleanup for an earlier item.
        for key in objects:
            try:
                hash(key)
            except (TypeError, ValueError) as exc:
                raise ValueError("cache entry IDs must be hashable") from exc

        outcomes: List[Dict[str, Any]] = []
        victims: List[Any] = []
        with self._operation_lock:
            with self._lock:
                self._assert_healthy()
                backup = self._capture_state()
                for key in objects:
                    self._register_id(key)
                if restoring:
                    order = sorted(
                        range(len(objects)),
                        key=lambda i: (
                            self._timestamp_key(accesses[i]),
                            self._stable_key(objects[i]),
                        ),
                    )
                    restore_ranks = {
                        index: index_rank - len(order)
                        for index_rank, index in enumerate(order)
                    }
                    indices = order
                else:
                    restore_ranks = {}
                    indices = list(range(len(objects)))

                for index in indices:
                    outcome, removed = self._put_one(
                        objects[index],
                        vectors[index],
                        explicit_embeddings,
                        restoring,
                        restore_ranks.get(index, 0),
                    )
                    outcomes.append(outcome)
                    victims.extend(removed)

                if restoring:
                    self._refresh_quotas(force=True)

            # A later item in the same batch can re-admit an ID that an
            # earlier item displaced.  Storage cleanup is deferred until the
            # whole batch is settled, so only IDs that are non-resident in the
            # final state may be sent to the callback.
            unique_victims = [
                victim
                for victim in self._unique(victims)
                if victim not in self._entries
            ]
            if unique_victims:
                try:
                    # The operation lock serializes hits/inserts while the
                    # storage callback runs; the state lock is deliberately
                    # released to avoid callback re-entrancy deadlocks.
                    self._on_evict(unique_victims)
                except Exception as exc:
                    with self._lock:
                        self._restore_state(backup)
                        self._healthy = False
                        self._failure_reason = repr(exc)
                        self._counters["callback_failures"] += 1
                    raise
            with self._lock:
                # Custom-ID ordinals exist only to make ties deterministic
                # while an ID is resident (or while the current batch is being
                # settled).  Retaining ordinals for every historical rejected
                # or evicted ID would make policy memory grow with trace length.
                self._prune_id_ordinals()
            return outcomes

    def restore(
        self,
        objs: Iterable[Any],
        embeddings: Optional[Sequence[Any]] = None,
        last_accesses: Optional[Sequence[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Reconstruct resident structure without counting live demand."""

        return self.put_with_metadata(
            objs,
            embeddings=embeddings,
            restoring=True,
            last_accesses=last_accesses,
        )

    def rebuild(
        self,
        objs: Iterable[Any],
        embeddings: Optional[Sequence[Any]] = None,
        last_accesses: Optional[Sequence[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Discard adaptive state and reconstruct from authoritative storage."""

        with self._operation_lock:
            with self._lock:
                self._reset_state()
            return self.put_with_metadata(
                objs,
                embeddings=embeddings,
                restoring=True,
                last_accesses=last_accesses,
            )

    def mark_unhealthy(self, reason: Any) -> None:
        """Enter fail-stop mode after an external recovery failure."""

        with self._operation_lock, self._lock:
            self._healthy = False
            self._failure_reason = repr(reason)

    def get(self, obj: Any):
        """Record a cache hit and return ``True`` when ``obj`` is resident."""

        with self._operation_lock:
            with self._lock:
                self._assert_healthy()
                try:
                    entry = self._entries.get(obj)
                except TypeError:
                    entry = None
                if entry is None:
                    self._counters["unknown_hits"] += 1
                    return None

                self._tick += 1
                self._counters["hits"] += 1
                entry.hit_mass = self._decayed(entry.hit_mass, entry.stat_tick, self._tick)
                entry.hit_mass += 1.0
                entry.stat_tick = self._tick
                entry.last_hit_tick = self._tick

                if entry.topic_id is not None and entry.topic_id in self._topics:
                    topic = self._topics[entry.topic_id]
                    self._touch_topic(topic, self._tick, demand=1.0, miss=0.0)
                if entry.cell_id is not None and entry.cell_id in self._cells:
                    cell = self._cells[entry.cell_id]
                    self._touch_cell(cell, self._tick, support=1.0)

                if self._tick - self._last_quota_tick >= self._QUOTA_REFRESH_INTERVAL:
                    self._refresh_quotas(force=True)
                self._last_event = {
                    "tick": self._tick,
                    "key": obj,
                    "action": "hit",
                    "admitted": False,
                    "evicted_ids": [],
                }
                return True

    def snapshot(self) -> Dict[str, Any]:
        """Return a deterministic, read-only snapshot for tests and telemetry."""

        with self._operation_lock, self._lock:
            tick = self._tick
            entries: Dict[Any, Dict[str, Any]] = {}
            for key in sorted(self._entries, key=self._stable_key):
                entry = self._entries[key]
                entries[key] = {
                    "id": key,
                    "topic_id": entry.topic_id,
                    "cell_id": entry.cell_id,
                    "hit_mass": self._decayed(
                        entry.hit_mass, entry.stat_tick, tick
                    ),
                    "insert_tick": entry.insert_tick,
                    "last_hit_tick": entry.last_hit_tick,
                }

            topics: Dict[int, Dict[str, Any]] = {}
            for topic_id in sorted(self._topics):
                topic = self._topics[topic_id]
                topics[topic_id] = {
                    "topic_id": topic_id,
                    "residents": sorted(topic.residents, key=self._stable_key),
                    "cell_ids": sorted(topic.cell_ids),
                    "demand": self._decayed(topic.demand, topic.stat_tick, tick),
                    "miss_mass": self._decayed(
                        topic.miss_mass, topic.stat_tick, tick
                    ),
                    "quota": self._quotas.get(topic_id, 0),
                    "centroid": topic.centroid.astype(float).tolist(),
                }

            cells: Dict[int, Dict[str, Any]] = {}
            for cell_id in sorted(self._cells):
                cell = self._cells[cell_id]
                cells[cell_id] = {
                    "cell_id": cell_id,
                    "topic_id": cell.topic_id,
                    "residents": sorted(cell.residents, key=self._stable_key),
                    "support": self._decayed(cell.support, cell.stat_tick, tick),
                    "last_seen_tick": cell.last_seen_tick,
                    "centroid": cell.centroid.astype(float).tolist(),
                }

            return {
                "policy": self.policy,
                "tick": tick,
                "maxsize": self.maxsize,
                "size": len(self._entries),
                "dimension": self._dimension,
                "entries": entries,
                "entry_ids": list(entries),
                "entry_states": list(entries.values()),
                "topics": topics,
                "topic_states": list(topics.values()),
                "cells": cells,
                "cell_states": list(cells.values()),
                "ghost_cells": sum(1 for cell in self._cells.values() if not cell.residents),
                "quotas": dict(sorted(self._quotas.items())),
                "last_event": None if self._last_event is None else dict(self._last_event),
                "stats": self.stats(),
            }

    def stats(self) -> Dict[str, Any]:
        with self._operation_lock, self._lock:
            result: Dict[str, Any] = dict(self._counters)
            result.update(
                {
                    "tick": self._tick,
                    "size": len(self._entries),
                    "topics": len(self._topics),
                    "cells": len(self._cells),
                    "ghost_cells": sum(
                        1 for cell in self._cells.values() if not cell.residents
                    ),
                    "id_ordinals": len(self._id_ordinals),
                    "healthy": self._healthy,
                    "failure_reason": self._failure_reason,
                }
            )
            return result

    def _assert_healthy(self) -> None:
        if not self._healthy:
            raise RuntimeError(
                "CARMA is unhealthy after a storage callback failure; rebuild "
                "it from authoritative storage before further use"
            )

    def _capture_state(self) -> Dict[str, Any]:
        """Copy mutable policy state without copying arbitrary ID objects."""

        entries = {key: copy.copy(entry) for key, entry in self._entries.items()}
        topics = {
            topic_id: _TopicState(
                topic_id=topic.topic_id,
                # Centroid updates replace arrays rather than mutating them, so
                # sharing the immutable snapshot avoids an O(cells * dimension)
                # copy on every transactional insertion.
                centroid=topic.centroid,
                seen_count=topic.seen_count,
                residents=set(topic.residents),
                cell_ids=set(topic.cell_ids),
                demand=topic.demand,
                miss_mass=topic.miss_mass,
                stat_tick=topic.stat_tick,
            )
            for topic_id, topic in self._topics.items()
        }
        cells = {
            cell_id: _CellState(
                cell_id=cell.cell_id,
                topic_id=cell.topic_id,
                centroid=cell.centroid,
                seen_count=cell.seen_count,
                residents=set(cell.residents),
                support=cell.support,
                stat_tick=cell.stat_tick,
                last_seen_tick=cell.last_seen_tick,
            )
            for cell_id, cell in self._cells.items()
        }
        last_event = None
        if self._last_event is not None:
            last_event = dict(self._last_event)
            last_event["evicted_ids"] = list(self._last_event.get("evicted_ids", []))
        return {
            "healthy": self._healthy,
            "failure_reason": self._failure_reason,
            "tick": self._tick,
            "dimension": self._dimension,
            "entries": entries,
            "topics": topics,
            "cells": cells,
            "next_topic_id": self._next_topic_id,
            "next_cell_id": self._next_cell_id,
            "quotas": dict(self._quotas),
            "last_quota_tick": self._last_quota_tick,
            "last_event": last_event,
            "id_ordinals": dict(self._id_ordinals),
            "next_id_ordinal": self._next_id_ordinal,
            "counters": dict(self._counters),
        }

    def _restore_state(self, state: Dict[str, Any]) -> None:
        self._healthy = state["healthy"]
        self._failure_reason = state["failure_reason"]
        self._tick = state["tick"]
        self._dimension = state["dimension"]
        self._entries = state["entries"]
        self._topics = state["topics"]
        self._cells = state["cells"]
        self._next_topic_id = state["next_topic_id"]
        self._next_cell_id = state["next_cell_id"]
        self._quotas = state["quotas"]
        self._last_quota_tick = state["last_quota_tick"]
        self._last_event = state["last_event"]
        self._id_ordinals = state["id_ordinals"]
        self._next_id_ordinal = state["next_id_ordinal"]
        self._counters = state["counters"]

    def _reset_state(self) -> None:
        self._healthy = True
        self._failure_reason = None
        self._tick = 0
        self._dimension = None
        self._entries = {}
        self._topics = {}
        self._cells = {}
        self._next_topic_id = 0
        self._next_cell_id = 0
        self._quotas = {}
        self._last_quota_tick = -self._QUOTA_REFRESH_INTERVAL
        self._last_event = None
        self._id_ordinals = {}
        self._next_id_ordinal = 0
        self._counters = {key: 0 for key in self._counters}

    def _put_one(
        self,
        key: Any,
        raw_vector: Any,
        explicit_embedding: bool,
        restoring: bool,
        restore_rank: int,
    ) -> Tuple[Dict[str, Any], List[Any]]:
        if key in self._entries:
            self._counters["duplicate_puts"] += 1
            outcome = self._outcome(key, "duplicate", True, [])
            self._last_event = outcome
            return outcome, []

        if restoring:
            event_tick = restore_rank
            self._counters["restores"] += 1
        else:
            self._tick += 1
            event_tick = self._tick
            self._counters["misses"] += 1

        vector = self._coerce_embedding(raw_vector)
        if explicit_embedding and vector is None:
            self._counters["invalid_embeddings"] += 1
            if not restoring:
                self._counters["rejections"] += 1
                outcome = self._outcome(key, "reject_invalid", False, [key])
                self._last_event = outcome
                return outcome, [key]

        restore_victim: Optional[_EntryState] = None
        if restoring and len(self._entries) >= self.maxsize:
            # Restore is a strict stored-LRU reconstruction. Semantic support
            # from an incoming row must not change which persisted row is old.
            restore_victim = min(
                self._entries.values(),
                key=lambda entry: (
                    entry.insert_tick,
                    self._stable_key(entry.key),
                ),
            )
            self._remove_entry(restore_victim.key)

        topic_id: Optional[int] = None
        cell_id: Optional[int] = None
        if vector is not None:
            topic_id, cell_id = self._assign(vector, restoring, event_tick)
            if cell_id is None and not restoring:
                self._counters["rejections"] += 1
                outcome = self._outcome(key, "reject_cell_capacity", False, [key])
                self._last_event = outcome
                return outcome, [key]

        if len(self._entries) < self.maxsize:
            self._add_entry(key, topic_id, cell_id, event_tick)
            if not restoring:
                self._counters["admissions"] += 1
            removed = [] if restore_victim is None else [restore_victim.key]
            if restore_victim is not None:
                self._counters["evictions"] += 1
            outcome = self._outcome(
                key,
                "restore_replace" if restore_victim is not None else (
                    "restore" if restoring else "admit_cold"
                ),
                True,
                removed,
            )
            self._last_event = outcome
            return outcome, removed

        if vector is None:
            if self.admission_enabled:
                self._counters["rejections"] += 1
                outcome = self._outcome(key, "reject_missing_embedding", False, [key])
                self._last_event = outcome
                return outcome, [key]
            victim = self._select_victim(list(self._entries.values()), self._tick)
            self._remove_entry(victim.key)
            self._add_entry(key, None, None, event_tick)
            self._counters["admissions"] += 1
            self._counters["evictions"] += 1
            outcome = self._outcome(key, "replace_unknown", True, [victim.key])
            self._last_event = outcome
            return outcome, [victim.key]

        self._refresh_quotas(force=True)
        cell = self._cells[cell_id]
        cell_support = self._decayed(cell.support, cell.stat_tick, self._tick)
        if (
            self.admission_enabled
            and cell_support < self.ghost_support_threshold - self._EPSILON
        ):
            self._counters["rejections"] += 1
            outcome = self._outcome(key, "reject_first_occurrence", False, [key])
            self._last_event = outcome
            return outcome, [key]

        if (
            self.admission_enabled
            and self.quota_enabled
            and self._quotas.get(topic_id, 0) <= 0
        ):
            self._counters["rejections"] += 1
            outcome = self._outcome(key, "reject_zero_quota", False, [key])
            self._last_event = outcome
            return outcome, [key]

        candidates = self._eligible_victims(topic_id)
        victim = self._select_victim(candidates, self._tick)
        if self.admission_enabled:
            victim_value = self._entry_value(victim, self._tick)
            post_occupancy = len(cell.residents) + 1
            if victim.cell_id == cell_id:
                post_occupancy -= 1
            candidate_value = cell_support / max(1, post_occupancy)
            if candidate_value + self._EPSILON < self.admission_margin * victim_value:
                self._counters["rejections"] += 1
                outcome = self._outcome(key, "reject_margin", False, [key])
                self._last_event = outcome
                return outcome, [key]

        self._remove_entry(victim.key)
        self._add_entry(key, topic_id, cell_id, event_tick)
        self._counters["admissions"] += 1
        self._counters["evictions"] += 1
        outcome = self._outcome(key, "replace", True, [victim.key])
        self._last_event = outcome
        return outcome, [victim.key]

    def _assign(
        self, vector: np.ndarray, restoring: bool, event_tick: int
    ) -> Tuple[Optional[int], Optional[int]]:
        topic = self._match_or_create_topic(vector)
        cell = self._match_or_create_cell(topic, vector, self._tick)

        if cell is None:
            if not topic.residents and not topic.cell_ids and topic.demand == 0:
                self._topics.pop(topic.topic_id, None)
                self._quotas.pop(topic.topic_id, None)
            return None, None

        if restoring:
            self._touch_cell(cell, self._tick, support=1.0)
        else:
            self._touch_topic(topic, self._tick, demand=1.0, miss=1.0)
            self._touch_cell(cell, self._tick, support=1.0)
        cell.last_seen_tick = self._tick if not restoring else event_tick
        return topic.topic_id, cell.cell_id

    def _match_or_create_topic(self, vector: np.ndarray) -> _TopicState:
        if self.one_topic and self._topics:
            topic = self._topics[min(self._topics)]
            self._update_centroid(topic, vector)
            return topic

        best_topic: Optional[_TopicState] = None
        best_similarity = -math.inf
        for topic_id in sorted(self._topics):
            topic = self._topics[topic_id]
            similarity = float(np.dot(topic.centroid, vector))
            if similarity > best_similarity:
                best_topic = topic
                best_similarity = similarity

        threshold_miss = best_topic is None or best_similarity < self.topic_threshold
        if (
            threshold_miss
            and not self.one_topic
            and len(self._topics) >= self.max_topics
        ):
            recyclable = [topic for topic in self._topics.values() if not topic.residents]
            if recyclable:
                recycled = min(
                    recyclable,
                    key=lambda topic: (
                        self._topic_weight(topic, self._tick),
                        max(
                            [self._cells[cell_id].last_seen_tick for cell_id in topic.cell_ids]
                            or [topic.stat_tick]
                        ),
                        topic.topic_id,
                    ),
                )
                for cell_id in list(recycled.cell_ids):
                    self._cells.pop(cell_id, None)
                    self._counters["ghost_cells_pruned"] += 1
                self._topics.pop(recycled.topic_id, None)
                self._quotas.pop(recycled.topic_id, None)
                self._counters["topics_recycled"] += 1

        should_create = (
            best_topic is None
            or (
                not self.one_topic
                and best_similarity < self.topic_threshold
                and len(self._topics) < self.max_topics
            )
        )
        if should_create:
            topic_id = self._next_topic_id
            self._next_topic_id += 1
            topic = _TopicState(topic_id=topic_id, centroid=vector.copy())
            self._topics[topic_id] = topic
            self._counters["topics_created"] += 1
            return topic

        assert best_topic is not None
        self._update_centroid(best_topic, vector)
        return best_topic

    def _match_or_create_cell(
        self, topic: _TopicState, vector: np.ndarray, tick: int
    ) -> Optional[_CellState]:
        best_cell: Optional[_CellState] = None
        best_similarity = -math.inf
        for cell_id in sorted(topic.cell_ids):
            cell = self._cells[cell_id]
            similarity = float(np.dot(cell.centroid, vector))
            if similarity > best_similarity:
                best_cell = cell
                best_similarity = similarity

        if best_cell is None or best_similarity < self.cell_threshold:
            if len(self._cells) >= self.max_cells:
                self._prune_one_ghost(tick)
            if len(self._cells) < self.max_cells:
                cell_id = self._next_cell_id
                self._next_cell_id += 1
                cell = _CellState(
                    cell_id=cell_id,
                    topic_id=topic.topic_id,
                    centroid=vector.copy(),
                    stat_tick=tick,
                    last_seen_tick=tick,
                )
                self._cells[cell_id] = cell
                topic.cell_ids.add(cell_id)
                self._counters["cells_created"] += 1
                return cell
            # The candidate did not satisfy the cell threshold and no empty
            # ghost could be reclaimed.  Falling through to ``best_cell`` here
            # would silently merge a below-threshold vector into an unrelated
            # resident cell and could bypass first-occurrence admission.
            return None

        if best_cell is None:
            return None
        self._update_centroid(best_cell, vector)
        return best_cell

    def _prune_one_ghost(self, tick: int) -> None:
        ghosts = [cell for cell in self._cells.values() if not cell.residents]
        if not ghosts:
            return
        victim = min(
            ghosts,
            key=lambda cell: (
                self._decayed(cell.support, cell.stat_tick, tick),
                cell.last_seen_tick,
                cell.cell_id,
            ),
        )
        topic = self._topics.get(victim.topic_id)
        if topic is not None:
            topic.cell_ids.discard(victim.cell_id)
        del self._cells[victim.cell_id]
        self._counters["ghost_cells_pruned"] += 1

    def _eligible_victims(self, candidate_topic_id: int) -> List[_EntryState]:
        all_entries = list(self._entries.values())
        if not self.quota_enabled:
            return all_entries

        topic = self._topics[candidate_topic_id]
        topic_occupancy = len(topic.residents)
        topic_quota = self._quotas.get(candidate_topic_id, 0)
        if topic_occupancy >= topic_quota:
            own = [self._entries[key] for key in topic.residents]
            return own or all_entries

        unknown = [entry for entry in all_entries if entry.topic_id is None]
        if unknown:
            return unknown

        donors: List[Tuple[int, float, int]] = []
        for topic_id, state in self._topics.items():
            occupancy = len(state.residents)
            excess = occupancy - self._quotas.get(topic_id, 0)
            if excess > 0:
                weight = self._topic_weight(state, self._tick)
                donors.append((topic_id, weight / max(1, occupancy), excess))
        if not donors:
            return all_entries
        donor_id = min(donors, key=lambda item: (-item[2], item[1], item[0]))[0]
        return [self._entries[key] for key in self._topics[donor_id].residents]

    def _select_victim(
        self, candidates: Sequence[_EntryState], tick: int
    ) -> _EntryState:
        if not candidates:
            raise RuntimeError("CARMA has no resident victim at capacity")
        return min(
            candidates,
            key=lambda entry: (
                self._entry_value(entry, tick),
                entry.last_hit_tick,
                entry.insert_tick,
                self._stable_key(entry.key),
            ),
        )

    def _entry_value(self, entry: _EntryState, tick: int) -> float:
        hit_mass = self._decayed(entry.hit_mass, entry.stat_tick, tick)
        if entry.cell_id is None or entry.cell_id not in self._cells:
            support_share = 1.0
        else:
            cell = self._cells[entry.cell_id]
            support = self._decayed(cell.support, cell.stat_tick, tick)
            support_share = support / max(1, len(cell.residents))
        return support_share + self.entry_hit_weight * hit_mass

    def _refresh_quotas(self, force: bool = False) -> None:
        if (
            not force
            and self._tick - self._last_quota_tick < self._QUOTA_REFRESH_INTERVAL
        ):
            return
        tick = self._tick
        active = []
        for topic_id, topic in self._topics.items():
            ghost_supported = any(
                self._decayed(
                    self._cells[cell_id].support,
                    self._cells[cell_id].stat_tick,
                    tick,
                )
                >= self.ghost_support_threshold - self._EPSILON
                for cell_id in topic.cell_ids
            )
            if topic.residents or ghost_supported:
                active.append(topic_id)

        weights = {
            topic_id: self._topic_weight(self._topics[topic_id], tick)
            for topic_id in active
        }
        selected = sorted(active, key=lambda tid: (-weights[tid], tid))[: self.maxsize]
        quotas = {topic_id: 0 for topic_id in self._topics}
        for topic_id in selected:
            quotas[topic_id] = 1

        remaining = self.maxsize - len(selected)
        if remaining > 0 and selected:
            total_weight = sum(weights[topic_id] for topic_id in selected)
            raw = {
                topic_id: remaining * weights[topic_id] / total_weight
                for topic_id in selected
            }
            floors = {topic_id: int(math.floor(raw[topic_id])) for topic_id in selected}
            for topic_id, amount in floors.items():
                quotas[topic_id] += amount
            leftover = remaining - sum(floors.values())
            remainder_order = sorted(
                selected,
                key=lambda tid: (
                    -(raw[tid] - floors[tid]),
                    -weights[tid],
                    tid,
                ),
            )
            for topic_id in remainder_order[:leftover]:
                quotas[topic_id] += 1

        self._quotas = quotas
        self._last_quota_tick = tick
        self._counters["quota_refreshes"] += 1

    def _topic_weight(self, topic: _TopicState, tick: int) -> float:
        demand = self._decayed(topic.demand, topic.stat_tick, tick)
        miss_mass = self._decayed(topic.miss_mass, topic.stat_tick, tick)
        pressure = (miss_mass + 1.0) / (demand + 2.0)
        base = max(self._EPSILON, demand * pressure)
        if self.quota_strength == 0:
            return 1.0
        return base ** self.quota_strength

    def _add_entry(
        self,
        key: Any,
        topic_id: Optional[int],
        cell_id: Optional[int],
        event_tick: int,
    ) -> None:
        if cell_id is not None:
            if topic_id is None or self._cells[cell_id].topic_id != topic_id:
                raise RuntimeError("CARMA topic/cell ownership invariant violated")
        self._entries[key] = _EntryState(
            key=key,
            topic_id=topic_id,
            cell_id=cell_id,
            hit_mass=0.0,
            stat_tick=self._tick,
            insert_tick=event_tick,
            last_hit_tick=event_tick,
        )
        if topic_id is not None:
            self._topics[topic_id].residents.add(key)
        if cell_id is not None:
            self._cells[cell_id].residents.add(key)

    def _remove_entry(self, key: Any) -> None:
        entry = self._entries.pop(key)
        if entry.topic_id is not None and entry.topic_id in self._topics:
            self._topics[entry.topic_id].residents.discard(key)
        if entry.cell_id is not None and entry.cell_id in self._cells:
            self._cells[entry.cell_id].residents.discard(key)

    def _touch_topic(
        self, topic: _TopicState, tick: int, demand: float, miss: float
    ) -> None:
        topic.demand = self._decayed(topic.demand, topic.stat_tick, tick) + demand
        topic.miss_mass = (
            self._decayed(topic.miss_mass, topic.stat_tick, tick) + miss
        )
        topic.stat_tick = tick

    def _touch_cell(self, cell: _CellState, tick: int, support: float) -> None:
        cell.support = self._decayed(cell.support, cell.stat_tick, tick) + support
        cell.stat_tick = tick
        cell.last_seen_tick = tick

    def _update_centroid(self, state: Any, vector: np.ndarray) -> None:
        alpha = self.centroid_alpha
        centroid = (1.0 - alpha) * state.centroid + alpha * vector
        norm = float(np.linalg.norm(centroid))
        if norm > 0 and math.isfinite(norm):
            state.centroid = (centroid / norm).astype(np.float32)
        state.seen_count += 1

    def _coerce_embedding(self, value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        try:
            vector = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            return None
        if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
            return None
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0:
            return None
        if self._dimension is None:
            self._dimension = int(vector.size)
        elif vector.size != self._dimension:
            return None
        return (vector / norm).astype(np.float32)

    def _decayed(self, value: float, then: int, now: int) -> float:
        elapsed = max(0, now - then)
        if elapsed == 0 or math.isinf(self.demand_half_life):
            return float(value)
        return float(value) * (2.0 ** (-elapsed / self.demand_half_life))

    @staticmethod
    def _aligned_optional(
        name: str, values: Optional[Sequence[Any]], expected: int
    ) -> List[Any]:
        if values is None:
            return [None] * expected
        result = list(values)
        if len(result) != expected:
            raise ValueError("%s must have the same length as objs" % name)
        return result

    def _register_id(self, value: Any) -> None:
        if self._has_intrinsic_stable_key(value):
            return
        if value not in self._id_ordinals:
            self._id_ordinals[value] = self._next_id_ordinal
            self._next_id_ordinal += 1

    def _prune_id_ordinals(self) -> None:
        """Retain and compact encounter order only for resident custom IDs."""

        retained = [
            (value, ordinal)
            for value, ordinal in self._id_ordinals.items()
            if value in self._entries
        ]
        retained.sort(key=lambda item: item[1])
        self._id_ordinals = {
            value: ordinal for ordinal, (value, _) in enumerate(retained)
        }
        self._next_id_ordinal = len(self._id_ordinals)

    @staticmethod
    def _has_intrinsic_stable_key(value: Any) -> bool:
        return (
            isinstance(value, (bool, int, str, bytes))
            or isinstance(value, float)
            and math.isfinite(value)
        )

    def _stable_key(self, value: Any) -> Tuple[Any, ...]:
        """Order built-in IDs by value and custom IDs by internal ordinal."""

        if isinstance(value, bool):
            return (0, int(value))
        if isinstance(value, int):
            return (1, value)
        if isinstance(value, float) and math.isfinite(value):
            return (2, value)
        if isinstance(value, str):
            return (3, value)
        if isinstance(value, bytes):
            return (4, value)
        self._register_id(value)
        return (
            5,
            type(value).__module__,
            type(value).__qualname__,
            self._id_ordinals[value],
        )

    @staticmethod
    def _timestamp_key(value: Any) -> Tuple[int, float, str]:
        if isinstance(value, datetime):
            return (0, value.timestamp(), "")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return (0, float(value), "")
        return (1, 0.0, repr(value))

    @staticmethod
    def _unique(values: Sequence[Any]) -> List[Any]:
        result = []
        seen = set()
        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)
        return result

    def _outcome(
        self, key: Any, action: str, admitted: bool, evicted_ids: List[Any]
    ) -> Dict[str, Any]:
        return {
            "tick": self._tick,
            "key": key,
            "action": action,
            "admitted": admitted,
            "evicted_ids": list(evicted_ids),
            "size": len(self._entries),
        }
