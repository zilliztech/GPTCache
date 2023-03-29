import hashlib
from abc import abstractmethod, ABCMeta
import pickle

import cachetools

from .scalar_data.sqllite3 import SQLite
from .scalar_data.scalar_store import ScalarStore
from .vector_data.faiss import Faiss
from .vector_data.vector_store import VectorStore
from .vector_data.vector_index import VectorIndex


class DataManager(metaclass=ABCMeta):
    @abstractmethod
    def init(self, **kwargs): pass

    @abstractmethod
    def save(self, data, embedding_data, **kwargs): pass

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

    def save(self, data, embedding_data, **kwargs):
        self.data[embedding_data] = (embedding_data, data)

    def get_scalar_data(self, vector_data, **kwargs):
        return vector_data[1]

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


# SFDataManager sqlite3 + faiss
class SFDataManager(DataManager):
    s: SQLite
    f: Faiss

    # when the size of data reach the max_size, it will remove the clean_size amount of data
    def __init__(self, sqlite_path, index_path, dimension, top_k,
                 max_size, clean_size, clean_cache_strategy):
        self.sqlite_path = sqlite_path
        self.index_path = index_path
        self.dimension = dimension
        self.top_k = top_k
        self.max_size = max_size
        self.cur_size = 0
        self.clean_size = clean_size
        self.clean_cache_strategy = clean_cache_strategy
        self.clean_cache_thread = None

    def init(self, **kwargs):
        self.s = SQLite(self.sqlite_path, self.clean_cache_strategy)
        self.cur_size = self.s.count()
        self.f = Faiss(self.index_path, self.dimension, self.top_k)

    def rebuild_index(self, all_data, top_k=1):
        bak = Faiss(self.index_path, self.dimension, top_k=top_k, skip_file=True)
        bak.mult_add(all_data)
        self.f = bak
        self.clean_cache_thread = None

    def save(self, data, embedding_data, **kwargs):
        if self.cur_size >= self.max_size and self.clean_cache_thread is None:
            self.s.clean_cache(self.clean_size)
            all_data = self.s.select_all_embedding_data()
            self.cur_size = len(all_data)
            self.rebuild_index(all_data, self.top_k)
            # TODO async
            # self.clean_cache_thread = threading.Thread(target=self.rebuild_index,
            #                                            args=(all_data, self.top_k),
            #                                            daemon=True)
            # self.clean_cache_thread.start()

        key = sha_data(embedding_data)
        self.s.insert(key, data, embedding_data)
        self.f.add(embedding_data)
        self.cur_size += 1

    def get_scalar_data(self, search_data, **kwargs):
        distance, vector_data = search_data
        key = sha_data(vector_data)
        return self.s.select_data(key)

    def search(self, embedding_data, **kwargs):
        return self.f.search(embedding_data)

    def close(self):
        self.s.close()
        self.f.close()


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

    def save(self, data, embedding_data, **kwargs):
        if self.cur_size >= self.max_size:
            ids = self.s.clean_cache(self.clean_size)
            self.cur_size = self.s.count()
            self.v.delete(ids)
        key = sha_data(embedding_data)
        self.s.insert(key, data, embedding_data)
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

    def save(self, data, embedding_data, **kwargs):
        if self.cur_size >= self.max_size:
            self.s.clean_cache(self.clean_size)
            all_data = self.s.select_all_embedding_data()
            self.cur_size = len(all_data)
            self.v = self.v.rebuild_index(all_data)
        key = sha_data(embedding_data)
        self.s.insert(key, data, embedding_data)
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
