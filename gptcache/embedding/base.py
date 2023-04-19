from abc import ABCMeta, abstractmethod


class BaseEmbedding(metaclass=ABCMeta):
    """
    _Embedding base.
    """

    @abstractmethod
    def to_embeddings(self, data, **kwargs):
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        return 0
