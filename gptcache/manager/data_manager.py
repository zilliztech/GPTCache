import pickle
import threading
from abc import abstractmethod, ABCMeta
from typing import List, Any, Optional, Union

import cachetools
import numpy as np
import requests

from gptcache.manager.eviction import EvictionBase
from gptcache.manager.eviction.distributed_cache import NoOpEviction
from gptcache.manager.eviction_manager import EvictionManager
from gptcache.manager.object_data.base import ObjectBase
from gptcache.manager.scalar_data.base import (
    CacheStorage,
    CacheData,
    DataType,
    Answer,
    Question,
)
from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.utils.error import CacheError, ParamError
from gptcache.utils.log import gptcache_log


class DataManager(metaclass=ABCMeta):
    """DataManager manage the cache data, including save and search"""

    @abstractmethod
    def save(self, question, answer, embedding_data, **kwargs):
        pass

    @abstractmethod
    def import_data(
            self,
            questions: List[Any],
            answers: List[Any],
            embedding_datas: List[Any],
            session_ids: List[Optional[str]],
            **kwargs,
    ):
        pass

    @abstractmethod
    def get_scalar_data(self, res_data, **kwargs) -> CacheData:
        pass

    def hit_cache_callback(self, res_data, **kwargs):
        pass

    @abstractmethod
    def search(self, embedding_data, **kwargs):
        """search the data in the cache store accrodding to the embedding data

        :return: a list of search result, [[score, id], [score, id], ...]
        """
        pass

    def flush(self):
        pass

    @abstractmethod
    def add_session(self, res_data, session_id, pre_embedding_data):
        pass

    @abstractmethod
    def list_sessions(self, session_id, key):
        pass

    @abstractmethod
    def delete_session(self, session_id):
        pass

    def report_cache(
        self,
        user_question,
        cache_question,
        cache_question_id,
        cache_answer,
        similarity_value,
        cache_delta_time,
    ):
        pass

    @abstractmethod
    def close(self):
        pass


class MapDataManager(DataManager):
    """MapDataManager, store all data in a map data structure.

    :param data_path: the path to save the map data, defaults to 'data_map.txt'.
    :type data_path:  str
    :param max_size: the max size for the cache, defaults to 1000.
    :type max_size: int
    :param get_data_container: a Callable to get the data container, defaults to None.
    :type get_data_container:  Callable


    Example:
        .. code-block:: python

            from gptcache.manager import get_data_manager

            data_manager = get_data_manager("data_map.txt", 1000)
    """

    def __init__(self, data_path, max_size, get_data_container=None):
        if get_data_container is None:
            self.data = cachetools.LRUCache(max_size)
        else:
            self.data = get_data_container(max_size)
        self.data_path = data_path
        self.init()

    def init(self):
        try:
            with open(self.data_path, "rb") as f:
                self.data = pickle.load(f)
        except FileNotFoundError:
            return
        except PermissionError:
            raise CacheError(  # pylint: disable=W0707
                f"You don't have permission to access this file <{self.data_path}>."
            )

    def save(self, question, answer, embedding_data, **kwargs):
        if isinstance(question, Question):
            question = question.content
        session = kwargs.get("session", None)
        session_id = {session.name} if session else set()
        self.data[embedding_data] = (question, answer, embedding_data, session_id)

    def import_data(
        self,
        questions: List[Any],
        answers: List[Any],
        embedding_datas: List[Any],
        session_ids: List[Optional[str]],
        **_,
    ):
        if (
            len(questions) != len(answers)
            or len(questions) != len(embedding_datas)
            or len(questions) != len(session_ids)
        ):
            raise ParamError("Make sure that all parameters have the same length")
        for i, embedding_data in enumerate(embedding_datas):
            self.data[embedding_data] = (
                questions[i],
                answers[i],
                embedding_datas[i],
                {session_ids[i]} if session_ids[i] else set(),
            )

    def get_scalar_data(self, res_data, **kwargs) -> CacheData:
        session = kwargs.get("session", None)
        if session:
            answer = (
                res_data[1].answer if isinstance(res_data[1], Answer) else res_data[1]
            )
            if not session.check_hit_func(
                session.name, list(res_data[3]), [res_data[0]], answer
            ):
                return None
        return CacheData(question=res_data[0], answers=res_data[1])

    def search(self, embedding_data, **kwargs):
        try:
            return [self.data[embedding_data]]
        except KeyError:
            return []

    def flush(self):
        try:
            with open(self.data_path, "wb") as f:
                pickle.dump(self.data, f)
        except PermissionError:
            gptcache_log.error(
                "You don't have permission to access this file %s.", self.data_path
            )

    def add_session(self, res_data, session_id, pre_embedding_data):
        res_data[3].add(session_id)

    def list_sessions(self, session_id=None, key=None):
        session_ids = set()
        for k in self.data:
            if session_id and session_id in self.data[k][3]:
                session_ids.add(k)
            elif len(self.data[k][3]) > 0:
                session_ids.update(self.data[k][3])
        return list(session_ids)

    def delete_session(self, session_id):
        keys = self.list_sessions(session_id=session_id)
        for k in keys:
            self.data[k][3].remove(session_id)
            if len(self.data[k][3]) == 0:
                del self.data[k]

    def close(self):
        self.flush()


def normalize(vec):
    try:
        array = np.asarray(vec, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ParamError("embedding data must be a finite one-dimensional vector") from exc
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ParamError("embedding data must be a finite one-dimensional vector")
    magnitude = float(np.linalg.norm(array))
    if not np.isfinite(magnitude) or magnitude <= 0:
        raise ParamError("embedding data must have a positive finite norm")
    return array / magnitude


class SSDataManager(DataManager):
    """Generate SSDataManage to manager the data.

    :param s: CacheStorage to manager the scalar data, it can be generated with :meth:`gptcache.manager.CacheBase`.
    :type s: CacheStorage
    :param v: VectorBase to manager the vector data, it can be generated with :meth:`gptcache.manager.VectorBase`.
    :type v:  VectorBase
    :param o: ObjectBase to manager the object data, it can be generated with :meth:`gptcache.manager.ObjectBase`.
    :type o:  ObjectBase
    :param e: EvictionBase to manager the eviction data, it can be generated with :meth:`gptcache.manager.EvictionBase`.
    :type e:  EvictionBase
    """

    def __init__(
        self,
        s: CacheStorage,
        v: VectorBase,
        o: Optional[ObjectBase],
        e: Optional[EvictionBase],
        max_size,
        clean_size,
        policy="LRU",
        policy_params=None,
    ):
        self.s = s
        self.v = v
        self.o = o
        self._operation_lock = threading.RLock()
        self.eviction_manager = EvictionManager(self.s, self.v)
        if policy_params is None:
            policy_params = {}
        if e is None:
            e = EvictionBase(name="memory",
                             maxsize=max_size,
                             clean_size=clean_size,
                             policy=policy,
                             on_evict=self._clear,
                             **dict(policy_params))
        self.eviction_base = e

        if not isinstance(self.eviction_base, NoOpEviction):
            # if eviction manager is no op redis, we don't need to put data into eviction base
            ids = self.s.get_ids(deleted=False)
            if getattr(self.eviction_base, "requires_embedding_restore", False):
                peek = getattr(self.s, "peek_data_by_id", self.s.get_data_by_id)
                cache_datas = [peek(cache_id) for cache_id in ids]
                embeddings = [
                    None if cache_data is None else cache_data.embedding_data
                    for cache_data in cache_datas
                ]
                last_accesses = [
                    None if cache_data is None else cache_data.last_access
                    for cache_data in cache_datas
                ]
                self.eviction_base.restore(
                    ids, embeddings=embeddings, last_accesses=last_accesses
                )
            else:
                self.eviction_base.put(ids)

    def _clear(self, marked_keys):
        if not marked_keys:
            return
        self.eviction_manager.soft_evict(marked_keys)
        if getattr(self.eviction_base, "requires_immediate_cleanup", False):
            self.eviction_manager.delete()
        elif self.eviction_manager.check_evict():
            self.eviction_manager.delete()

    def _rebuild_eviction_state(self):
        if not getattr(self.eviction_base, "requires_embedding_restore", False):
            return
        ids = self.s.get_ids(deleted=False)
        peek = getattr(self.s, "peek_data_by_id", self.s.get_data_by_id)
        cache_datas = [peek(cache_id) for cache_id in ids]
        self.eviction_base.rebuild(
            ids,
            embeddings=[
                None if cache_data is None else cache_data.embedding_data
                for cache_data in cache_datas
            ],
            last_accesses=[
                None if cache_data is None else cache_data.last_access
                for cache_data in cache_datas
            ],
        )

    @staticmethod
    def _unique_ids(values):
        result = []
        seen = set()
        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)
        return result

    def _mark_eviction_unhealthy(self, reason):
        marker = getattr(self.eviction_base, "mark_unhealthy", None)
        if callable(marker):
            try:
                marker(reason)
            # pylint: disable-next=broad-except
            except Exception as marker_error:  # pragma: no cover - defensive
                # Defensive: a custom eviction implementation must not mask the
                # storage error that caused this fail-stop path.
                gptcache_log.error(
                    "Failed to mark eviction policy unhealthy: %s", marker_error
                )

    def _recover_failed_import(self, ids, marked_before):
        """Roll back this import without clearing pre-existing tombstones."""
        try:
            self.s.mark_deleted(ids)
            before = set(marked_before)
            marked_after = self.s.get_ids(deleted=True)
            recovery_ids = self._unique_ids(
                list(ids) + [key for key in marked_after if key not in before]
            )

            if recovery_ids:
                delete_result = self.v.delete(recovery_ids)
                if delete_result is False:
                    raise RuntimeError("vector store reported unsuccessful deletion")

            live_ids = set(self.s.get_ids(deleted=False))
            if any(key in live_ids for key in ids):
                raise RuntimeError("scalar rollback left imported rows live")

            targeted_clear = getattr(self.s, "clear_deleted_data_by_ids", None)
            if callable(targeted_clear):
                targeted_clear(recovery_ids)
            return True
        # pylint: disable-next=broad-except
        except Exception as recovery_error:  # pragma: no cover - backend-specific
            # Storage plugins expose different exception types. Recovery must
            # fail closed for any backend-specific error.
            gptcache_log.error(
                "Failed to restore cache consistency after cache import: %s",
                recovery_error,
            )
            self._mark_eviction_unhealthy(recovery_error)
            return False

    def save(self, question, answer, embedding_data, **kwargs):
        """Save the data and vectors to cache and vector storage.

        :param question: question data.
        :type question: str
        :param answer: answer data.
        :type answer: str, Answer or (Any, DataType)
        :param embedding_data: vector data.
        :type embedding_data: np.ndarray

        Example:
            .. code-block:: python

                import numpy as np
                from gptcache.manager import get_data_manager, CacheBase, VectorBase

                data_manager = get_data_manager(CacheBase('sqlite'), VectorBase('faiss', dimension=128))
                data_manager.save('hello', 'hi', np.random.random((128, )).astype('float32'))
        """
        session = kwargs.get("session", None)
        session_id = session.name if session else None
        self.import_data([question], [answer], [embedding_data], [session_id], **kwargs)

    def _process_answer_data(self, answers: Union[Answer, List[Answer]]):
        if isinstance(answers, Answer):
            answers = [answers]
        new_ans = []
        for ans in answers:
            if ans.answer_type != DataType.STR:
                new_ans.append(Answer(self.o.put(ans.answer), ans.answer_type))
            else:
                new_ans.append(ans)
        return new_ans

    def _process_question_data(self, question: Union[str, Question]):
        if isinstance(question, Question):
            if question.deps is None:
                return question

            for dep in question.deps:
                if dep.dep_type == DataType.IMAGE_URL:
                    dep.dep_type.data = self.o.put(requests.get(dep.data).content)
            return question

        return Question(question)

    def import_data(
        self,
        questions: List[Any],
        answers: List[Answer],
        embedding_datas: List[Any],
        session_ids: List[Optional[str]],
        **kwargs,
    ):
        if (
            len(questions) != len(answers)
            or len(questions) != len(embedding_datas)
            or len(questions) != len(session_ids)
        ):
            raise ParamError("Make sure that all parameters have the same length")
        cache_datas = []
        embedding_datas = [
            normalize(embedding_data) for embedding_data in embedding_datas
        ]
        expected_dimension = getattr(
            self.v, "_dimension", getattr(self.v, "dimension", None)
        )
        if expected_dimension is not None and any(
            embedding_data.size != expected_dimension
            for embedding_data in embedding_datas
        ):
            raise ParamError(
                "embedding dimension does not match the configured vector store"
            )
        for i, embedding_data in enumerate(embedding_datas):
            if self.o is not None and not isinstance(answers[i], str):
                ans = self._process_answer_data(answers[i])
            else:
                ans = answers[i]

            cache_datas.append(
                CacheData(
                    question=self._process_question_data(questions[i]),
                    answers=ans,
                    embedding_data=embedding_data.astype("float32"),
                    session_id=session_ids[i],
                )
            )
        with self._operation_lock:
            marked_before = self.s.get_ids(deleted=True)
            ids = []
            try:
                ids = self.s.batch_insert(cache_datas)
                self.v.mul_add(
                    [
                        VectorData(id=ids[i], data=embedding_data)
                        for i, embedding_data in enumerate(embedding_datas)
                    ],
                    **kwargs,
                )
                if getattr(self.eviction_base, "accepts_embedding_metadata", False):
                    self.eviction_base.put_with_metadata(
                        ids,
                        embeddings=[
                            cache_data.embedding_data for cache_data in cache_datas
                        ],
                    )
                else:
                    self.eviction_base.put(ids)
            except Exception:
                recovered = not ids or self._recover_failed_import(
                    ids, marked_before
                )
                if recovered:
                    try:
                        self._rebuild_eviction_state()
                    # pylint: disable-next=broad-except
                    except Exception as rebuild_error:  # pragma: no cover
                        # Rebuild is the last recovery boundary; any plugin
                        # failure leaves the policy explicitly unhealthy.
                        gptcache_log.error(
                            "Failed to rebuild eviction state after cache import: %s",
                            rebuild_error,
                        )
                        self._mark_eviction_unhealthy(rebuild_error)
                raise

    def get_scalar_data(self, res_data, **kwargs) -> Optional[CacheData]:
        session = kwargs.get("session", None)
        cache_data = self.s.get_data_by_id(res_data[1])
        if cache_data is None:
            return None

        if session:
            cache_answer = (
                cache_data.answers[0].answer
                if isinstance(cache_data.answers[0], Answer)
                else cache_data.answers[0]
            )
            res_list = self.list_sessions(key=res_data[1])
            cache_session_ids, cache_questions = [r.session_id for r in res_list], [
                r.session_question for r in res_list
            ]
            if not session.check_hit_func(
                session.name, cache_session_ids, cache_questions, cache_answer
            ):
                return None

        for ans in cache_data.answers:
            if ans.answer_type != DataType.STR:
                ans.answer = self.o.get(ans.answer)
        return cache_data

    def hit_cache_callback(self, res_data, **kwargs):
        self.eviction_base.get(res_data[1])

    def search(self, embedding_data, **kwargs):
        embedding_data = normalize(embedding_data)
        top_k = kwargs.pop("top_k", -1)
        return self.v.search(data=embedding_data, top_k=top_k, **kwargs)

    def flush(self):
        self.s.flush()
        self.v.flush()

    def add_session(self, res_data, session_id, pre_embedding_data):
        self.s.add_session(res_data[1], session_id, pre_embedding_data)

    def list_sessions(self, session_id=None, key=None):
        res = self.s.list_sessions(session_id, key)
        if key:
            return res
        if session_id:
            return list(r.id for r in res)
        return list(set(r.session_id for r in res))

    def delete_session(self, session_id):
        keys = self.list_sessions(session_id=session_id)
        self.s.delete_session(keys)

    def report_cache(
        self,
        user_question,
        cache_question,
        cache_question_id,
        cache_answer,
        similarity_value,
        cache_delta_time,
    ):
        self.s.report_cache(
            user_question,
            cache_question,
            cache_question_id,
            cache_answer,
            similarity_value,
            cache_delta_time,
        )

    def close(self):
        self.s.close()
        self.v.close()
