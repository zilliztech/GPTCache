import hashlib
from abc import abstractmethod, ABCMeta
import pickle
import numpy as np
from typing import Callable

import cachetools

from .scalar_data.scalar_store import ScalarStore
from .vector_data.vector_store import VectorStore
from .vector_data.vector_index import VectorIndex


class DataManager(metaclass=ABCMeta):
    @abstractmethod
    def init(self, **kwargs): pass

    @abstractmethod
    def save(self, question: str, answer: str, embedding_data: np.ndarray, **kwargs): pass

    # should return the tuple, (question, answer)
    @abstractmethod
    def get_scalar_data(self, search_data, **kwargs): pass

    @abstractmethod
    def search(self, embedding_data: np.ndarray, **kwargs): pass

    @abstractmethod
    def close(self): pass


class MapDataManager(DataManager):
    def __init__(self, data_path: str, max_size: int, get_data_container: Callable = None):
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
            print(f'File <${self.data_path}> is not found.')
        except PermissionError:
            print(f'You don\'t have permission to access this file <${self.data_path}>.')

    def save(self, question: str, answer: str, embedding_data: np.ndarray, **kwargs):
        self.data[embedding_data] = (question, answer)

    def get_scalar_data(self, search_data, **kwargs):
        return search_data

    def search(self, embedding_data: np.ndarray, **kwargs):
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


def sha_data(data: np.ndarray):
    m = hashlib.sha1()
    m.update(data.astype('float32').tobytes())
    return m.hexdigest()


# SSDataManager scalar store and vector store
class SSDataManager(DataManager):
    s: ScalarStore
    v: VectorStore

    def __init__(self, max_size: int, clean_size: int, s: ScalarStore, v: VectorStore):
        self.max_size = max_size
        self.cur_size = 0
        self.clean_size = clean_size
        self.s = s
        self.v = v

    def init(self, **kwargs):
        self.s.init(**kwargs)
        self.v.init(**kwargs)
        self.cur_size = self.s.count()

    def save(self, question: str, answer: str, embedding_data: np.ndarray, **kwargs):
        if self.cur_size >= self.max_size:
            ids = self.s.eviction(self.clean_size)
            self.cur_size = self.s.count()
            self.v.delete(ids)
        key = sha_data(embedding_data)
        self.s.insert(key, question, answer, embedding_data)
        self.v.add(key, embedding_data)
        self.cur_size += 1

    def get_scalar_data(self, search_data, **kwargs):
        distance, vector_data = search_data
        key = sha_data(vector_data)
        return self.s.select_data(key)

    def search(self, embedding_data: np.ndarray, **kwargs):
        return self.v.search(embedding_data)

    def close(self):
        self.s.close()
        self.v.close()


# SIDataManager scalar store and vector index
class SIDataManager(DataManager):
    s: ScalarStore
    v: VectorIndex

    def __init__(self, max_size: int, clean_size: int, s: ScalarStore, v: VectorIndex):
        self.max_size = max_size
        self.cur_size = 0
        self.clean_size = clean_size
        self.s = s
        self.v = v

    def init(self, **kwargs):
        self.s.init(**kwargs)
        self.v.init(**kwargs)
        self.cur_size = self.s.count()

    def save(self, question: str, answer: str, embedding_data: np.ndarray, **kwargs):
        if self.cur_size >= self.max_size:
            self.s.eviction(self.clean_size)
            all_data = self.s.select_all_embedding_data()
            self.cur_size = len(all_data)
            self.v = self.v.rebuild_index(all_data)
        key = sha_data(embedding_data)
        self.s.insert(key, question, answer, embedding_data)
        self.v.add(key, embedding_data)
        self.cur_size += 1

    def get_scalar_data(self, search_data, **kwargs):
        distance, vector_data = search_data
        key = sha_data(vector_data)
        return self.s.select_data(key)

    def search(self, embedding_data: np.ndarray, **kwargs):
        return self.v.search(embedding_data)

    def close(self):
        self.s.close()
        self.v.close()
