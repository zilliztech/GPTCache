from abc import ABCMeta, abstractmethod


class SimilarityEvaluation(metaclass=ABCMeta):
    @abstractmethod
    def evaluation(self, src_dict, cache_dict, **kwargs): pass

    @abstractmethod
    def range(self): pass
