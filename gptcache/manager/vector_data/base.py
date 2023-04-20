from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class VectorData:
    id: int
    data: np.ndarray


class VectorBase(ABC):
    """VectorBase: base vector store interface"""

    @abstractmethod
    def mul_add(self, datas: List[VectorData]):
        pass

    @abstractmethod
    def search(self, data: np.ndarray, top_k: int):
        pass

    @abstractmethod
    def rebuild(self, ids=None) -> bool:
        pass

    @abstractmethod
    def delete(self, ids) -> bool:
        pass

    @abstractmethod
    def close(self) -> bool:
        pass
