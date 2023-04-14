from abc import abstractmethod, ABCMeta
import pickle
from typing import List, Any

import cachetools
import numpy as np

from gptcache.utils.error import CacheError, ParamError
from gptcache.manager.scalar_data.base import CacheStorage, CacheData
from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.manager.eviction import EvictionManager


class DataManager(metaclass=ABCMeta):
    """DataManager manage the cache data, including save and search"""

    @abstractmethod
    def save(self, question, answer, embedding_data, **kwargs):
        pass

    @abstractmethod
    def import_data(
        self, questions: List[Any], answers: List[Any], embedding_datas: List[Any]
    ):
        pass

    # should return the tuple, (question, answer, embedding)
    @abstractmethod
    def get_scalar_data(self, res_data, **kwargs):
        pass

    def update_access_time(self, res_data, **kwargs):
        pass

    @abstractmethod
    def search(self, embedding_data, **kwargs):
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
        self.data[embedding_data] = (question, answer, embedding_data)

    def import_data(
        self, questions: List[Any], answers: List[Any], embedding_datas: List[Any]
    ):
        if len(questions) != len(answers) or len(questions) != len(embedding_datas):
            raise ParamError("Make sure that all parameters have the same length")
        for i, embedding_data in enumerate(embedding_datas):
            self.data[embedding_data] = (questions[i], answers[i], embedding_datas[i])

    def get_scalar_data(self, res_data, **kwargs):
        return res_data

    def search(self, embedding_data, **kwargs):
        try:
            return [self.data[embedding_data]]
        except KeyError:
            return []

    def close(self):
        try:
            with open(self.data_path, "wb") as f:
                pickle.dump(self.data, f)
        except PermissionError:
            print(f"You don't have permission to access this file <{self.data_path}>.")


def normalize(vec):
    magnitude = np.linalg.norm(vec)
    normalized_v = vec / magnitude
    return normalized_v


class SSDataManager(DataManager):
    """Generate SSDataManage to manager the data.

    :param s: CacheStorage to manager the scalar data, it can be generated with :meth:`gptcache.manager.CacheBase`.
    :type s: CacheStorage
    :param v: VectorBase to manager the vector data, it can be generated with :meth:`gptcache.manager.VectorBase`.
    :type v:  VectorBase
    :param max_size: the max size for the cache, defaults to 1000.
    :type max_size: int
    :param clean_size: the size to clean up, defaults to `max_size * 0.2`.
    :type clean_size: int
    :param eviction: The eviction policy, it is support "LRU" and "FIFO" now, and defaults to "LRU".
    :type eviction:  str
    """

    s: CacheStorage
    v: VectorBase

    def __init__(self, s: CacheStorage, v: VectorBase, max_size, clean_size, eviction="LRU"):
        self.max_size = max_size
        self.cur_size = 0
        self.clean_size = clean_size
        self.s = s
        self.v = v
        self.eviction = EvictionManager(self.s, self.v, eviction)
        self.cur_size = self.s.count()

    def _clear(self):
        self.eviction.soft_evict(self.clean_size)
        if self.eviction.check_evict():
            self.eviction.delete()
        self.cur_size = self.s.count()

    def save(self, question, answer, embedding_data, **kwargs):
        """Save the data and vectors to cache and vector storage.

        :param question: question data.
        :type question: str
        :param answer: answer data.
        :type answer: str
        :param embedding_data: vector data.
        :type embedding_data: np.ndarray

        Example:
            .. code-block:: python

                import numpy as np
                from gptcache.manager import get_data_manager, CacheBase, VectorBase

                data_manager = get_data_manager(CacheBase('sqlite'), VectorBase('faiss', dimension=128))
                data_manager.save('hello', 'hi', np.random.random((128, )).astype('float32'))
        """

        if self.cur_size >= self.max_size:
            self._clear()

        self.import_data([question], [answer], [embedding_data])

    def import_data(
        self, questions: List[Any], answers: List[Any], embedding_datas: List[Any]
    ):
        if len(questions) != len(answers) or len(questions) != len(embedding_datas):
            raise ParamError("Make sure that all parameters have the same length")
        cache_datas = []
        embedding_datas = [
            normalize(embedding_data) for embedding_data in embedding_datas
        ]
        for i, embedding_data in enumerate(embedding_datas):
            cache_datas.append(
                CacheData(
                    question=questions[i],
                    answer=answers[i],
                    embedding_data=embedding_data.astype("float32"),
                )
            )
        ids = self.s.batch_insert(cache_datas)
        self.v.mul_add(
            [
                VectorData(id=ids[i], data=embedding_data)
                for i, embedding_data in enumerate(embedding_datas)
            ]
        )
        self.cur_size += len(questions)

    def get_scalar_data(self, res_data, **kwargs):
        return self.s.get_data_by_id(res_data[1])

    def update_access_time(self, res_data, **kwargs):
        return self.s.update_access_time(res_data[1])

    def search(self, embedding_data, **kwargs):
        embedding_data = normalize(embedding_data)
        return self.v.search(embedding_data)

    def close(self):
        self.s.close()
        self.v.close()
