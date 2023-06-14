from datetime import datetime
from typing import Tuple, Dict, Any

from gptcache.adapter.api import _get_eval
from gptcache.similarity_evaluation import SimilarityEvaluation


class TimeEvaluation(SimilarityEvaluation):
    """Add time dimension restrictions on the basis of other Evaluation,
    for example, only use the cache within 1 day from the current time,
    and filter out the previous cache.

    :param evaluation: Similarity evaluation, like distance/onnx.
    :param evaluation_config: Similarity evaluation config.
    :param time_range: Time range, time unit: s

    Example:
        .. code-block:: python

            import datetime

            from gptcache.manager.scalar_data.base import CacheData
            from gptcache.similarity_evaluation import TimeEvaluation

            evaluation = TimeEvaluation(evaluation="distance", time_range=86400)

            similarity = eval.evaluation(
                {},
                {
                    "search_result": (3.5, None),
                    "cache_data": CacheData("a", "b", create_on=datetime.datetime.now()),
                },
            )
            # 0.5

    """

    def __init__(self, evaluation: str, evaluation_config=None, time_range: float = 86400.0):
        if evaluation_config is None:
            evaluation_config = {}
        self._eval = _get_eval(evaluation, evaluation_config)
        self._time_range = time_range

    def evaluation(self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **kwargs) -> float:
        cache_data = cache_dict.get("cache_data", None)
        if not cache_data or not cache_data.create_on:
            return self.range()[0]
        delta_time = datetime.now().timestamp() - cache_data.create_on.timestamp()
        if delta_time > self._time_range:
            return self.range()[0]
        return self._eval.evaluation(src_dict, cache_dict, **kwargs)

    def range(self) -> Tuple[float, float]:
        return self._eval.range()

