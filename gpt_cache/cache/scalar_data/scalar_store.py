from abc import ABCMeta, abstractmethod

import numpy as np


class ScalarStore(metaclass=ABCMeta):
    @abstractmethod
    def init(self, **kwargs): pass

    @abstractmethod
    def insert(self, key, question, answer, embedding_data: np.ndarray): pass

    @abstractmethod
    def select_data(self, key): pass

    @abstractmethod
    def select_all_embedding_data(self): pass

    # eviction: should return the id list
    @abstractmethod
    def eviction(self, count): pass

    @abstractmethod
    def count(self): pass

    @abstractmethod
    def close(self): pass
