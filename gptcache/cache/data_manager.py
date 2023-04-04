import hashlib
from abc import abstractmethod, ABCMeta
import pickle
import cachetools
import numpy as np

from .scalar_data.base import ScalarStorage
from .vector_data.base import VectorBase, ClearStrategy
from .eviction import EvictionManager
from ..utils.error import CacheError


class DataManager(metaclass=ABCMeta):
    @abstractmethod
    def init(self, **kwargs): pass

    @abstractmethod
    def save(self, question, answer, embedding_data, **kwargs): pass

    # should return the tuple, (question, answer)
    @abstractmethod
    def get_scalar_data(self, vector_data, **kwargs): pass

    @abstractmethod
    def search(self, embedding_data, **kwargs): pass

    @abstractmethod
    def close(self): pass


class MapDataManager(DataManager):
    def __init__(self, data_path, max_size, get_data_container=None):
        if get_data_container is None:
            self.data = cachetools.LRUCache(max_size)
        else:
            self.data = get_data_container(max_size)
        self.data_path = data_path

    def init(self, **kwargs):
        try:
            f = open(self.data_path, 'rb')
            self.data = pickle.load(f)
            f.close()
        except FileNotFoundError:
            # print(f'File <${self.data_path}> is not found.')
            return
        except PermissionError:
            raise CacheError(f'You don\'t have permission to access this file <${self.data_path}>.')

    def save(self, question, answer, embedding_data, **kwargs):
        self.data[embedding_data] = (question, answer)

    def get_scalar_data(self, vector_data, **kwargs):
        return vector_data

    def search(self, embedding_data, **kwargs):
        try:
            return [self.data[embedding_data]]
        except KeyError:
            return []

    def close(self):
        try:
            f = open(self.data_path, 'wb')
            pickle.dump(self.data, f)
            f.close()
        except PermissionError:
            print(f'You don\'t have permission to access this file <${self.data_path}>.')


def sha_data(data):
    if isinstance(data, list):
        data = np.array(data)
    m = hashlib.sha1()
    m.update(data.astype('float32').tobytes())
    return m.hexdigest()


def normalize(vec):
    magnitude = np.linalg.norm(vec)
    normalized_v = vec / magnitude
    return normalized_v


class SSDataManager(DataManager):
    s: ScalarStorage
    v: VectorBase

    def __init__(self, max_size, clean_size, s, v):
        self.max_size = max_size
        self.cur_size = 0
        self.clean_size = clean_size
        self.s = s
        self.v = v
        self.eviction = EvictionManager(s, v)

    def init(self, **kwargs):
        self.s.init(**kwargs)
        self.v.init(**kwargs)
        self.cur_size = self.s.count()

    def _clear(self):
        self.eviction.soft_evict(self.clean_size)
        if not self.eviction.check_evict():
            pass
        elif self.v.clear_strategy() == ClearStrategy.DELETE:
            self.eviction.delete()
        elif self.v.clear_strategy() == ClearStrategy.REBUILD:
            self.eviction.rebuild()
        else:
            raise RuntimeError('Unknown clear strategy')
        self.cur_size = self.s.count()

    def save(self, question, answer, embedding_data, **kwargs):
        if self.cur_size >= self.max_size:
            self._clear()
        embedding_data = normalize(embedding_data)
        key = sha_data(embedding_data)
        self.s.insert(key, question, answer, embedding_data.astype('float32'))
        self.v.add(key, embedding_data)
        self.cur_size += 1

    def get_scalar_data(self, search_data, **kwargs):
        distance, vector_data = search_data
        key = sha_data(vector_data)
        self.s.update_access_time(key)
        return self.s.get_data_by_id(key)

    def search(self, embedding_data, **kwargs):
        embedding_data = normalize(embedding_data)
        return self.v.search(embedding_data)

    def close(self):
        self.s.close()
        self.v.close()
