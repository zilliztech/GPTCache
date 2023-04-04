from abc import ABCMeta, abstractmethod

import numpy as np


class ScalarStore(metaclass=ABCMeta):
    @abstractmethod
    def init(self, **kwargs): pass

    @abstractmethod
    def insert(self, key: str, question: str, answer, embedding_data: np.ndarray): pass

    @abstractmethod
    def select_data(self, key: str): pass

    @abstractmethod
    def select_all_embedding_data(self): pass

    # eviction: should return the id list
    @abstractmethod
    def eviction(self, count: int): pass

    @abstractmethod
    def count(self): pass

    @abstractmethod
    def close(self): pass
