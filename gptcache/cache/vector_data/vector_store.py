from abc import ABCMeta, abstractmethod

import numpy as np


class VectorStore(metaclass=ABCMeta):
    @abstractmethod
    def init(self, **kwargs): pass

    @abstractmethod
    def add(self, key: str, data: np.ndarray): pass

    @abstractmethod
    def search(self, data: np.ndarray): pass

    @abstractmethod
    def delete(self, ids: list[str]): pass

    @abstractmethod
    def close(self): pass
