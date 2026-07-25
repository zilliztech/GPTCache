from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional


class EvictionBase(metaclass=ABCMeta):
    """
    Eviction base.
    """

    @abstractmethod
    def put(self, objs: List[Any], costs: Optional[List[float]] = None):
        """Insert entries, optionally with a per-entry cost.

        :param costs: optional list matching ``objs``, giving how expensive
            each entry was to produce (e.g. generated tokens or answer
            length). Only cost-aware policies (GDSF) use it; all other
            implementations ignore it.
        """
        pass

    @abstractmethod
    def get(self, obj: Any):
        pass

    @property
    @abstractmethod
    def policy(self) -> str:
        pass
