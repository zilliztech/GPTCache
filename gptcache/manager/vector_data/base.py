from abc import ABC, abstractmethod


class VectorBase(ABC):
    """VectorBase: base vector store interface"""

    @abstractmethod
    def add(self, key: str, data: "ndarray"):
        pass

    @abstractmethod
    def search(self, data: "ndarray"):
        pass

    @abstractmethod
    def rebuild(self, ids=None) -> bool:
        pass

    @abstractmethod
    def delete(self, ids) -> bool:
        pass

    @abstractmethod
    def close(self):
        pass
