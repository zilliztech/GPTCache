import numpy as np
from typing import Dict, Any

from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.manager.vector_data.base import VectorBase


def euclidean_distance_calculate(vec_l: np.array, vec_r: np.array):
    return np.sum((vec_l - vec_r)**2)

class KReciprocalEvaluation(SearchDistanceEvaluation):
    """Using K Reciprocal to evaluate sentences pair similarity.

    This evaluator borrows popular reranking method K-reprocical reranking for similarity evaluation. K-reciprocal relation refers to the mutual
    nearest neighbor relationship between two embeddings, where each embedding is the K nearest neighbor of the other based on a given distance
    metric.  This evaluator checks whether the query embedding is in candidate cache embedding's `top_k` nearest neighbors. If query embedding
    is not candidate's `top_k` neighbors, the pair will be considered as dissimilar pair. Otherwise, their distance will be kept and continue
    for a `SearchDistanceEvaluation` check.  `max_distance` is used to bound this distance to make it between [0-`max_distance`]. `positive` is
    used to indicate this distance is directly proportional to the similarity of two entites. If `positive` is set `False`,
    `max_distance` will be used to substract this distance to get the final score.

    :param vectordb: vector database to retrieval embeddings to test k-reciprocal relationship.
    :type vectordb: gptcache.manager.vector_data.base.VectorBase
    :param top_k: for each retievaled candidates, this method need to test if the query is top-k of candidate.
    :type top_k: int
    :param max_distance: the bound of maximum distance.
    :type max_distance: float
    :param positive: if the larger distance indicates more similar of two entities, It is True. Otherwise it is False.
    :type positive: bool

    Example:
        .. code-block:: python

            from gptcache.similarity_evaluation import KReciprocalEvaluation
            from gptcache.manager.vector_data.faiss import Faiss
            from gptcache.manager.vector_data.base import VectorData
            import numpy as np

            faiss = Faiss('./none', 3, 10)
            cached_data = np.array([0.57735027, 0.57735027, 0.57735027])
            faiss.mul_add([VectorData(id=0, data=cached_data)])
            evaluation = KReciprocalEvaluation(vectordb=faiss, top_k=2, max_distance = 4.0, positive=False)
            query = np.array([0.61396013, 0.55814557, 0.55814557])
            score = evaluation.evaluation(
                {
                    'question': 'question1',
                    'embedding': query
                },
                {
                    'question': 'question2',
                    'embedding': cached_data
                }
            )
    """


    def __init__(self, vectordb: VectorBase, top_k: int = 3, max_distance: float = 4.0, positive: bool=False):
        super().__init__(max_distance, positive)
        self.vectordb = vectordb
        self.top_k = top_k

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
        src_question = src_dict['question']
        cache_question = cache_dict['question']
        if src_question == cache_question:
            return 1
        query_emb = self.normalize(src_dict['embedding'])
        candidates = self.vectordb.search(cache_dict['embedding'], self.top_k + 1)
        euc_dist = euclidean_distance_calculate(query_emb, cache_dict['embedding'])
        if euc_dist > candidates[-1][0]:
            euc_dist = self.range()[1]

        result_dict = {}
        result_dict['search_result'] = (euc_dist, None)
        return super().evaluation(None, result_dict)


