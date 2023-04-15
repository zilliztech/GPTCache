from abc import ABC, abstractmethod
from typing import Any, List


class ObjectBase(ABC):
    """
    Object storage base.
    """

    @abstractmethod
    def put(self, obj: Any) -> str:
        pass

    @abstractmethod
    def get(self, obj: str) -> Any:
        pass

    @abstractmethod
    def get_access_link(self, obj: str) -> str:
        pass

    @abstractmethod
    def delete(self, to_delete: List[str]):
        pass
