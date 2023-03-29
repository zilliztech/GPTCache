from abc import ABCMeta, abstractmethod

import numpy as np


class ScalarStore(metaclass=ABCMeta):
    @abstractmethod
    def init(self, **kwargs): pass

    @abstractmethod
    def insert(self, key, data, embedding_data: np.ndarray): pass

    @abstractmethod
    def select_data(self, key): pass

    @abstractmethod
    def select_all_embedding_data(self): pass

    # clean_cache: should return the id list
    @abstractmethod
    def clean_cache(self, count): pass

    @abstractmethod
    def count(self): pass

    @abstractmethod
    def close(self): pass
