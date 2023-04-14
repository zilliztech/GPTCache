from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Any, List, Union

import numpy as np


class AnswerType(IntEnum):
    STR = 0
    IMAGE_BASE64 = 1
    IMAGE_URL = 2


@dataclass
class CacheData:
    """
    answer_type:
        0: str
        1: base64 image
    """
    question: Any
    answer: Union[Any, List[Any]]
    embedding_data: Optional[np.ndarray] = None
    answer_type: int = AnswerType.STR


class CacheStorage(metaclass=ABCMeta):
    """
    BaseStorage for scalar data.
    """

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def batch_insert(self, all_data: List[CacheData]):
        pass

    @abstractmethod
    def get_data_by_id(self, key):
        pass

    @abstractmethod
    def clear_deleted_data(self):
        pass

    @abstractmethod
    def get_ids(self, deleted=True):
        pass

    @abstractmethod
    def mark_deleted(self, keys):
        pass

    @abstractmethod
    def update_access_time(self, key):
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
