import numpy as np

from .similarity_evaluation import SimilarityEvaluation


class NumpyNormEvaluation(SimilarityEvaluation):

    def evaluation(self, src_dict, cache_dict, **kwargs):
        src_embedding = src_dict["embedding"]
        _, cache_embedding = cache_dict["search_result"]
        return 1.0 - np.linalg.norm(src_embedding - cache_embedding)

    def range(self):
        return 0.0, 1.0
