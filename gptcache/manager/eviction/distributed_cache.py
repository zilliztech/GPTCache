# pylint: disable=wrong-import-position
from abc import ABC, abstractmethod
from typing import List

from gptcache.manager.eviction.base import EvictionBase


class DistributedEviction(EvictionBase, ABC):
    """
    Base class for Distributed Eviction Strategy.
    """

    @abstractmethod
    def put(self, objs: List[str]):
        pass

    @abstractmethod
    def get(self, obj: str):
        pass

    @property
    @abstractmethod
    def policy(self) -> str:
        pass


class NoOpEviction(EvictionBase):
    """eviction: No Op Eviction Strategy. This is used when Eviction is managed internally
    by the Databases such as Redis or memcached and no eviction is required to perform.

    """

    @property
    def policy(self) -> str:
        return ""

    def __init__(self, **kwargs):
        pass

    def put(self, objs: List[str]):
        pass

    def get(self, obj: str):
        pass
