from abc import ABC, abstractmethod
from typing import List
from enum import Enum


class ClearStrategy(Enum):
    REBUILD = 0
    DELETE = 1


class VectorBase(ABC):

    def init(self, **kwargs):
        pass

    @abstractmethod
    def add(self, key: str, data: 'ndarray'):
        pass

    @abstractmethod
    def search(self, data: 'ndarray'):
        pass

    @abstractmethod
    def clear_strategy(self):
        pass    

    def rebuild(self) -> bool:
        raise NotImplementedError

    def delete(self, ids) -> bool:
        raise NotImplementedError

    @abstractmethod
    def close(self):
        pass
