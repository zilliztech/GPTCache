from typing import Tuple, Dict, Any
import numpy as np
from gptcache.similarity_evaluation import SimilarityEvaluation
from gptcache.manager.vector_data.base import VectorBase
from gptcache.utils.log import gptcache_log


class SearchDistanceEvaluation(SimilarityEvaluation):
    """Using search distance to evaluate sentences pair similarity.

    This is the evaluator to compare two embeddings according to their distance computed in embedding retrieval stage.
    In the retrieval stage, `search_result` is the distance used for approximate nearest neighbor search and have been
    put into `cache_dict`. `max_distance` is used to bound this distance to make it between [0-`max_distance`]. `positive` is
    used to indicate this distance is directly proportional to the similarity of two entites. If `positive` is set `False`,
    `max_distance` will be used to substract this distance to get the final score.

    :param max_distance: the bound of maximum distance.
    :type max_distance: float
    :param positive: if the larger distance indicates more similar of two entities, It is True. Otherwise it is False.
    :type positive: bool

    Example:
        .. code-block:: python

            from gptcache.similarity_evaluation import SearchDistanceEvaluation

            evaluation = SearchDistanceEvaluation()
            score = evaluation.evaluation(
                {},
                {
                    "search_result": (1, None)
                }
            )
    """

    def __init__(self, max_distance=4.0, positive=False, vectordb: VectorBase = None, cache_check: bool = False):
        self.max_distance = max_distance
        self.positive = positive
        self.vectordb = vectordb
        self.cache_check = cache_check

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
        self.cache_health_check(cache_dict)
        distance, _ = cache_dict["search_result"]
        if distance < 0:
            distance = 0
        elif distance > self.max_distance:
            distance = self.max_distance
        if self.positive:
            return distance
        return self.max_distance - distance

    def range(self) -> Tuple[float, float]:
        """Range of similarity score.

        :return: minimum and maximum of similarity score.
        """
        return 0.0, self.max_distance

    def cache_health_check(self, cache_dict: Dict[str, Any]):
        """This function checks if the embedding
           from vector store matches one in cache store.
           If cache store and vector store are out of
           sync with each other, cache retrieval can
           be incorrect.
           If this happens, force the similary score
           to the lowerest possible value.
        """
        if self.cache_check:
            emb_in_cache = cache_dict["embedding"]
            _, data_id = cache_dict["search_result"]
            emb_in_vec = self.vectordb.get_embeddings(data_id)
            flag = np.all(emb_in_cache == emb_in_vec)
            if not flag:
                gptcache_log.critical("Cache Store and Vector Store are out of sync!!!")
                # 0: identical, inf: different
                cache_dict["search_result"] = (
                    np.inf,
                    data_id,
                )
                # self-healing by replacing entry
                # in the vec store with the one
                # from cache store by the same
                # entry_id.
                self.vectordb.update_embeddings(
                    id=data_id,
                    emb=cache_dict["embedding"],
                )
