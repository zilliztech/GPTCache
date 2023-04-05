import numpy as np

from .similarity_evaluation import SimilarityEvaluation


class NumpyNormEvaluation(SimilarityEvaluation):

    def __init__(self, enable_normal: bool = False):
        self.enable_normal = enable_normal

    @staticmethod
    def normalize(vec):
        magnitude = np.linalg.norm(vec)
        normalized_v = vec / magnitude
        return normalized_v

    def evaluation(self, src_dict, cache_dict, **kwargs):
        src_embedding = self.normalize(src_dict["embedding"]) if self.enable_normal else src_dict["embedding"]
        _, cache_embedding = cache_dict["search_result"]
        cache_embedding = self.normalize(cache_embedding) if self.enable_normal \
            else cache_embedding
        return np.linalg.norm(src_embedding - cache_embedding)

    def range(self):
        return 0.0, 2.0
