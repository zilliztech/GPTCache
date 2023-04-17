from abc import ABCMeta, abstractmethod
from typing import Any, List


class EvictionBase(metaclass=ABCMeta):
    """
    Eviction base.
    """

    @abstractmethod
    def put(self, objs: List[Any]):
        pass

    @abstractmethod
    def get(self, obj: Any):
        pass

    @property
    @abstractmethod
    def policy(self) -> str:
        pass
