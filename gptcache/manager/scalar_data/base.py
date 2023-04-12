from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any, List

import numpy as np


@dataclass
class CacheData:
    question: Any
    answer: Any
    embedding_data: Optional[np.ndarray] = None


class CacheStorage(metaclass=ABCMeta):
    """
    BaseStorage for scalar data.
    """

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def batch_insert(self, datas: List[CacheData]):
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
