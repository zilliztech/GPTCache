from typing import Tuple, Dict, Any, List

import numpy as np

from gptcache.adapter.api import _get_model
from gptcache.similarity_evaluation import SimilarityEvaluation


def euclidean_distance_calculate(vec_l: np.array, vec_r: np.array):
    return np.sum((vec_l - vec_r) ** 2)


def reweight(weights, length):
    if length >= len(weights):
        return weights
    else:
        reweighted_ws = []
        sum_ws = 0
        for i in range(length):
            sum_ws += weights[i]
        for i in range(length):
            reweighted_ws.append(weights[i] * (1 / sum_ws))
        return reweighted_ws


class SequenceMatchEvaluation(SimilarityEvaluation):
    """
    Evaluate sentence pair similarity using SequenceMatchEvaluation.

    :param weights: List of weights corresponding to each sequence element for calculating the weighted distance.
    :type weights: List[float]
    :param embedding_extractor: The embedding extractor used to obtain embeddings from the text content.
    :type embedding_extractor: gptcache.embedding.base.BaseEmbedding


    Example:
        .. code-block:: python

            from gptcache.similarity_evaluation import SequenceMatchEvaluation
            from gptcache.embedding import Onnx

            weights = [0.5, 0.3, 0.2]
            evaluation = SequenceMatchEvaluation(weights, 'onnx')

            query = {
                'question': 'USER: "foo2" USER: "foo4"',
            }

            cache = {
                'question': 'USER: "foo6" USER: "foo8"',
            }

            score = evaluation.evaluation(query, cache)
    """

    def __init__(
        self, weights: List[float], embedding_extractor: str, embedding_config=None
    ):
        self.embedding_extractor = _get_model(embedding_extractor, embedding_config)
        self.weights = weights

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
        src_question = src_dict["question"]
        cache_question = cache_dict["question"]
        src_contents = src_question.split("USER: ")
        cache_contents = cache_question.split("USER: ")
        src_contents = [content for content in src_contents if len(content) > 0]
        cache_contents = [content for content in cache_contents if len(content) > 0]
        src_embs = []
        cache_embs = []
        for content in src_contents:
            src_embs.append(
                self.normalize(self.embedding_extractor.to_embeddings(content))
            )
        for content in cache_contents:
            cache_embs.append(
                self.normalize(self.embedding_extractor.to_embeddings(content))
            )
        length = min([len(src_contents), len(cache_contents), len(self.weights)])
        assert length > 0
        ws = self.weights
        ws = ws[::-1]
        ws = reweight(ws, length)
        src_embs = src_embs[::-1]
        cache_embs = cache_embs[::-1]
        weighted_distance = 0
        for i in range(length):
            weighted_distance += (
                4 - euclidean_distance_calculate(src_embs[i], cache_embs[i])
            ) * ws[i]
        return weighted_distance

    def range(self) -> Tuple[float, float]:
        """Range of similarity score.

        :return: minimum and maximum of similarity score.
        """
        return 0.0, 4.0
