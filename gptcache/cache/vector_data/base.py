from abc import ABC, abstractmethod
from typing import List


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
    def delete(self) -> bool:
        pass

    @abstractmethod
    def close(self):
        pass
