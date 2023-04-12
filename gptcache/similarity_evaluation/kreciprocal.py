import numpy as np
from typing import Tuple, Dict, Any

from gptcache.similarity_evaluation import SimilarityEvaluation


def euclidean_distance_calculate(vec_l: np.array, vec_r: np.array):
    return np.sum((vec_l - vec_r)**2)

class KReciprocalEvaluation(SimilarityEvaluation):
    """Using K Reciprocal to evaluate sentences pair similarity.

    :vectordb: vector database to retrieval embeddings to test k-reciprocal relationship.
    :type model: gptcache.manager.vector_data.faiss

    Example:
        .. code-block:: python

            from gptcache.similarity_evaluation import KReciprocalEvaluation
            from gptcache.manager.vector_data.faiss import Faiss
            import numpy as np

            faiss = Faiss('./none', 3, 10)
            cached_data = np.array([1.0, 2.0, 3.0])
            faiss.add(0, cached_data)
            evaluation = KReciprocalEvaluation(vectordb=faiss)
            query = np.array([1.1, 2.1, 3.1])
            score = evaluation.evaluation(
                {
                    'question': 'question1',
                    'embedding_data': query
                },
                {
                    'question': 'question2',
                    'embedding_data': cached_data
                }
            )
    """


    def __init__(self, vectordb: 'gptcache.manager.vector_data.base.VectorBase', top_k: int = 3):
        self.vectordb = vectordb
        self.top_k = top_k

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
        src_question = src_dict['question']
        cache_question = cache_dict['question']
        if src_question == cache_question:
            return 1
        candidates = list(self.vectordb.search(cache_dict['embedding_data'], self.top_k))
        top_k_candidates = candidates[: self.top_k]
        euc_dist = euclidean_distance_calculate(src_dict['embedding_data'], cache_dict['embedding_data'])
        if euc_dist > top_k_candidates[-1][0]:
            euc_dist = 0
        return euc_dist

    def range(self) -> Tuple[float, float]:
        """Range of similarity score.

        :return: minimum and maximum of similarity score.
        """
        return 0.0, 1.0
