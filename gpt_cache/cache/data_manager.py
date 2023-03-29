import hashlib
from abc import abstractmethod, ABCMeta
import pickle

import cachetools

from .scalar_data.scalar_store import ScalarStore
from .vector_data.vector_store import VectorStore
from .vector_data.vector_index import VectorIndex


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
            print(f'File <${self.data_path}> is not found.')
        except PermissionError:
            print(f'You don\'t have permission to access this file <${self.data_path}>.')

    def save(self, question, answer, embedding_data, **kwargs):
        self.data[embedding_data] = (question, answer)

    def get_scalar_data(self, vector_data, **kwargs):
        return vector_data

    def search(self, embedding_data, **kwargs):
        return [self.data[embedding_data]]

    def close(self):
        try:
            f = open(self.data_path, 'wb')
            pickle.dump(self.data, f)
            f.close()
        except PermissionError:
            print(f'You don\'t have permission to access this file <${self.data_path}>.')


def sha_data(data):
    m = hashlib.sha1()
    m.update(data.astype('float32').tobytes())
    return m.hexdigest()


# SVDataManager scalar store and vector store
class SSDataManager(DataManager):
    s: ScalarStore
    v: VectorStore

    def __init__(self, max_size, clean_size, s, v):
        self.max_size = max_size
        self.cur_size = 0
        self.clean_size = clean_size
        self.s = s
        self.v = v

    def init(self, **kwargs):
        self.s.init(**kwargs)
        self.v.init(**kwargs)
        self.cur_size = self.s.count()

    def save(self, question, answer, embedding_data, **kwargs):
        if self.cur_size >= self.max_size:
            ids = self.s.clean_cache(self.clean_size)
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

    def search(self, embedding_data, **kwargs):
        return self.v.search(embedding_data)

    def close(self):
        self.s.close()
        self.v.close()


# SIDataManager scalar store and vector index
class SIDataManager(DataManager):
    s: ScalarStore
    v: VectorIndex

    def __init__(self, max_size, clean_size, s, v):
        self.max_size = max_size
        self.cur_size = 0
        self.clean_size = clean_size
        self.s = s
        self.v = v

    def init(self, **kwargs):
        self.s.init(**kwargs)
        self.v.init(**kwargs)
        self.cur_size = self.s.count()

    def save(self, question, answer, embedding_data, **kwargs):
        if self.cur_size >= self.max_size:
            self.s.clean_cache(self.clean_size)
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

    def search(self, embedding_data, **kwargs):
        return self.v.search(embedding_data)

    def close(self):
        self.s.close()
        self.v.close()
