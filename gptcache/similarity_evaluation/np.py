from typing import Dict, Tuple, Any

import numpy as np

from gptcache.similarity_evaluation import SimilarityEvaluation


class NumpyNormEvaluation(SimilarityEvaluation):
    """Using Numpy norm to evaluate sentences pair similarity.

    This evaluator calculate the L2 distance of two embeddings for similarity check. if `enable_normal` is True,
    both query embedding and cache embedding will be normalized. Note normalized distance will substracted by
    maximum distance so it will be positively correlated with the similarity.

    :param enable_normal: whether to normalize the embedding, defaults to False.
    :type enable_normal: bool
    :param question_embedding_function: optional, a function to generate question embedding
    :type question_embedding_function: function

    Example:
        .. code-block:: python

            from gptcache.similarity_evaluation import NumpyNormEvaluation
            import numpy as np

            evaluation = NumpyNormEvaluation()
            score = evaluation.evaluation(
                {
                    'question': 'What is color of sky?'
                    'embedding': np.array([-0.5, -0.5])
                },
                {
                    'question': 'What is the color of sky?'
                    'embedding': np.array([-0.49, -0.51])
                }
            )
    """

    def __init__(self, enable_normal: bool = True, question_embedding_function=None):
        self.enable_normal = enable_normal
        self.question_encoder = question_embedding_function

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
        if 'question' in src_dict and 'question' in cache_dict:
            if src_dict['question'].lower() == cache_dict['question'].lower():
                return self.range()[1]
            if 'embedding' not in src_dict or 'embedding' not in cache_dict or src_dict['embedding'] is None or cache_dict['embedding'] is None:
                assert self.question_encoder, 'You need to a valid question_embedding_function to generate question embedding in the evaluator.'
                src_dict['embedding'] = self.question_encoder(src_dict['question'])
                cache_dict['embedding'] = self.question_encoder(cache_dict['question'])
        src_embedding = (
            self.normalize(src_dict['embedding'])
            if self.enable_normal
            else src_dict['embedding']
        )
        cache_embedding = cache_dict['embedding']
        cache_embedding = (
            self.normalize(cache_embedding) if self.enable_normal else cache_embedding
        )
        return self.range()[1] - np.linalg.norm(src_embedding - cache_embedding)

    def range(self) -> Tuple[float, float]:
        """Range of similarity score.

        :return: minimum and maximum of similarity score.
        """
        return 0.0, 2.0
