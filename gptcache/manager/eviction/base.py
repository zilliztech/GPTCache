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

    def put_with_metadata(self, objs: List[Any], **metadata):
        """Optional metadata-aware insertion hook.

        Existing third-party policies remain compatible because the default
        implementation ignores metadata and delegates to ``put``.
        """
        del metadata
        return self.put(objs)

    def restore(self, objs: List[Any], **metadata):
        """Optional cold-start hook that defaults to ordinary insertion."""
        del metadata
        return self.put(objs)

    @property
    def requires_immediate_cleanup(self) -> bool:
        return False

    @property
    def requires_embedding_restore(self) -> bool:
        return False

    @property
    def accepts_embedding_metadata(self) -> bool:
        return False

    @property
    @abstractmethod
    def policy(self) -> str:
        pass
