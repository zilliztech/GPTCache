import hashlib
from abc import abstractmethod, ABCMeta
import pickle
from .scalar_data.sqllite3 import SQLite
from .vector_data.faiss import Faiss


class DataManager(metaclass=ABCMeta):
    @abstractmethod
    def init(self): pass

    @abstractmethod
    def save(self, data, embedding_data, **kwargs): pass

    @abstractmethod
    def get_scalar_data(self, vector_data, **kwargs): pass

    @abstractmethod
    def search(self, embedding_data, **kwargs): pass

    @abstractmethod
    def close(self): pass


class MapDataManager(DataManager):
    def __init__(self, data_path):
        self.data = {}
        self.data_path = data_path

    def init(self):
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
        return self.data[embedding_data]

    def close(self):
        try:
            f = open(self.data_path, 'wb')
            pickle.dump(self.data, f)
            f.close()
        except PermissionError:
            print(f'You don\'t have permission to access this file <${self.data_path}>.')


def sha_data(data):
    m = hashlib.sha1()
    m.update(data.tobytes())
    return m.hexdigest()


# SFDataManager sqlite3 + knowhere
class SFDataManager(DataManager):
    s: SQLite
    f: Faiss

    def __init__(self, sqlite_path, index_path, dimension):
        self.sqlite_path = sqlite_path
        self.index_path = index_path
        self.dimension = dimension

    def init(self):
        self.s = SQLite(self.sqlite_path)
        self.f = Faiss(self.index_path, self.dimension)

    def save(self, data, embedding_data, **kwargs):
        key = sha_data(embedding_data)
        self.s.insert(key, data)
        self.f.add(embedding_data)

    def get_scalar_data(self, search_data, **kwargs):
        distance, vector_data = search_data
        key = sha_data(vector_data)
        return self.s.select(key)

    def search(self, embedding_data, **kwargs):
        return self.f.search(embedding_data)

    def close(self):
        self.s.close()
        self.f.close()

