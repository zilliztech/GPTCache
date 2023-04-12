from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np


class ClearStrategy(Enum):
    REBUILD = 0
    DELETE = 1


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
    def search(self, data: np.ndarray):
        pass

    @abstractmethod
    def clear_strategy(self):
        pass

    def rebuild(self, all_data, keys) -> bool:
        raise NotImplementedError

    def delete(self, ids) -> bool:
        raise NotImplementedError

    @abstractmethod
    def close(self):
        pass
