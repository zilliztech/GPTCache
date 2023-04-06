from abc import ABCMeta, abstractmethod


class BaseEmbedding(metaclass=ABCMeta):
    """
    _Embedding base.
    """
    @abstractmethod
    def to_embeddings(self, data):
        pass

    @abstractmethod
    def dimension(self):
        pass
