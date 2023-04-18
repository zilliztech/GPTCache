from typing import Optional, Callable, List

from gptcache.utils.error import CacheError


class Config:
    """Pass configuration.

    :param log_time_func: optional, customized log time function
    :param similarity_threshold: a threshold ranged from 0 to 1 to filter search results with similarity score higher than the threshold.
                                 When it is 0, there is no hits. When it is 1, all search results will be returned as hits.
    :type similarity_threshold: float

    Example:
        .. code-block:: python

            from gptcache import Config

            configs = Config(similarity_threshold=0.6)
    """

    def __init__(
            self,
            log_time_func: Optional[Callable[[str, float], None]] = None,
            similarity_threshold: float = 0.8,
            prompts: Optional[List[str]] = None
    ):
        if similarity_threshold < 0 or similarity_threshold > 1:
            raise CacheError(
                "Invalid the similarity threshold param, reasonable range: 0-1"
            )
        self.log_time_func = log_time_func
        self.similarity_threshold = similarity_threshold
        self.prompts = prompts
