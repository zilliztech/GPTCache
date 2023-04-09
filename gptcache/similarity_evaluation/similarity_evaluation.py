from abc import ABCMeta, abstractmethod
from typing import Tuple, Dict, Any


class SimilarityEvaluation(metaclass=ABCMeta):
    @abstractmethod
    def evaluation(
        self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **kwargs
    ) -> float:
        pass

    @abstractmethod
    def range(self) -> Tuple[float, float]:
        pass
