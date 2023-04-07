from abc import ABCMeta, abstractmethod

import numpy as np


class CacheStorage(metaclass=ABCMeta):
    """
    BaseStorage for scalar data.
    """
    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def insert(self, key, data, reply, embedding_data: np.ndarray = None):
        pass

    @abstractmethod
    def get_data_by_id(self, key):
        pass

    @abstractmethod
    def get_embedding_data(self, offset, size):
        pass

    @abstractmethod
    def remove_by_state(self):
        pass

    @abstractmethod
    def update_access_time(self, key):
        pass

    @abstractmethod
    def update_state(self, keys):
        pass

    @abstractmethod
    def count(self):
        pass

    @abstractmethod
    def get_old_access(self, count):
        pass

    @abstractmethod
    def get_old_create(self, count):
        pass

    @abstractmethod
    def close(self):
        pass
