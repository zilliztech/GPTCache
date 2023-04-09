from typing import Dict, Tuple, Any

import numpy as np

from gptcache.similarity_evaluation import SimilarityEvaluation


class NumpyNormEvaluation(SimilarityEvaluation):
    """Using Numpy norm to evaluate sentences pair similarity.

    :param enable_normal: whether to normalize the embedding, defaults to False.
    :type enable_normal: bool

    Example:
        .. code-block:: python

            from gptcache.similarity_evaluation import NumpyNormEvaluation
            import numpy as np

            evaluation = NumpyNormEvaluation()
            score = evaluation.evaluation(
                {
                    'embedding': np.array([-0.5, -0.5])
                },
                {
                    'search_result': (0, np.array([1, 1]))
                }
            )
    """

    def __init__(self, enable_normal: bool = True):
        self.enable_normal = enable_normal

    @staticmethod
    def normalize(vec: np.ndarray):
        """Normalize the input vector.

        :param vec: numpy vector needs to normalize.
        :type vec: numpy.array

        :return: normalized vector.
        """
        magnitude = np.linalg.norm(vec)
        normalized_v = vec / magnitude
        return normalized_v

    def evaluation(
        self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **_
    ) -> float:
        """Evaluate the similarity score of pair.

        :param src_dict: the query dictionary to evaluate with cache.
        :type src_dict: Dict
        :param cache_dict: the cache dictionary.
        :type cache_dict: Dict

        :return: evaluation score.
        """
        src_embedding = (
            self.normalize(src_dict["embedding"])
            if self.enable_normal
            else src_dict["embedding"]
        )
        _, cache_embedding = cache_dict["search_result"]
        cache_embedding = (
            self.normalize(cache_embedding) if self.enable_normal else cache_embedding
        )
        return np.linalg.norm(src_embedding - cache_embedding)

    def range(self) -> Tuple[float, float]:
        """Range of similarity score.

        :return: minimum and maximum of similarity score.
        """
        return 0.0, 2.0
