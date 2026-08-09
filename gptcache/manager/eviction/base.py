from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional


class EvictionBase(metaclass=ABCMeta):
    """
    Eviction base.
    """

    @abstractmethod
    def put(self, objs: List[Any], embeddings: Optional[List[Any]] = None):
        """Register cache entries with the eviction policy.

        :param objs: the entry ids.
        :param embeddings: optional unit-norm embeddings, one per id. Policies
            that make eviction decisions from the vectors themselves (such as
            ``ARC`` with semantic ghost lists) use these; every other policy
            ignores the argument. Optional and defaulted so that existing
            implementations and callers are unaffected.
        """

    @abstractmethod
    def get(self, obj: Any):
        pass

    @property
    @abstractmethod
    def policy(self) -> str:
        pass
