from abc import ABCMeta, abstractmethod

import numpy as np


class VectorIndex(metaclass=ABCMeta):
    @abstractmethod
    def init(self, **kwargs): pass

    @abstractmethod
    def add(self, key, data: np.ndarray): pass

    @abstractmethod
    def search(self, data: np.ndarray): pass

    @abstractmethod
    def rebuild_index(self, all_data): pass

    @abstractmethod
    def close(self): pass
