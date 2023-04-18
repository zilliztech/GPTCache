from typing import Tuple, Dict, Any

from gptcache.similarity_evaluation.similarity_evaluation import SimilarityEvaluation


class ExactMatchEvaluation(SimilarityEvaluation):
    """Using exact metric to evaluate sentences pair similarity.

    This evaluator is used to directly compare two `question` from text. If every single character in two questions can match, then this evaluator
    will return 1 else 0.

    Example:
        .. code-block:: python

            from gptcache.similarity_evaluation import ExactMatchEvaluation

            evaluation = ExactMatchEvaluation()
            score = evaluation.evaluation(
                {
                    "question": "What is the color of sky?"
                },
                {
                    "question": "What is the color of sky?"
                }
            )
    """

    def __init__(self):
        pass

    def evaluation(
        self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **_
    ) -> float:
        """Evaluate the similarity score of pair.

        :param src_dict: the query dictionary to evaluate with cache_dict.
        :type src_dict: Dict
        :param cache_dict: the cache dictionary.
        :type cache_dict: Dict

        :return: evaluation score.
        """
        return 1 if cache_dict["question"] == src_dict["question"] else 0

    def range(self) -> Tuple[float, float]:
        """Range of similarity score.

        :return: minimum and maximum of similarity score.
        """
        return 0, 1
