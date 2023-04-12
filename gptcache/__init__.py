import atexit
import os
import time
from typing import List, Any, Optional, Callable

import openai

from gptcache.embedding.string import to_embeddings as string_embedding
from gptcache.manager.data_manager import DataManager
from gptcache.manager.factory import get_data_manager
from gptcache.processor.post import first
from gptcache.processor.pre import last_content
from gptcache.similarity_evaluation.similarity_evaluation import SimilarityEvaluation
from gptcache.similarity_evaluation.exact_match import ExactMatchEvaluation
from gptcache.utils.error import CacheError


def cache_all(*_, **__):
    return True


def time_cal(func, func_name=None, report_func=None):
    def inner(*args, **kwargs):
        time_start = time.time()
        res = func(*args, **kwargs)
        delta_time = time.time() - time_start
        if cache.config.log_time_func:
            cache.config.log_time_func(
                func.__name__ if func_name is None else func_name, delta_time
            )
        if report_func is not None:
            report_func(delta_time)
        return res

    return inner


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


class Report:
    """Get GPTCache report including time and counts for different operations."""

    def __init__(self):
        self.embedding_all_time = 0
        self.embedding_count = 0
        self.search_all_time = 0
        self.search_count = 0
        self.hint_cache_count = 0

    def embedding(self, delta_time):
        """Embedding counts and time.

        :param delta_time: additional runtime.
        """
        self.embedding_all_time += delta_time
        self.embedding_count += 1

    def search(self, delta_time):
        """Search counts and time.

        :param delta_time: additional runtime.
        """
        self.search_all_time += delta_time
        self.search_count += 1

    def average_embedding_time(self):
        """Average embedding time.

        :param delta_time: delta time.
        """
        return round(
            self.embedding_all_time / self.embedding_count
            if self.embedding_count != 0
            else 0,
            4,
        )

    def average_search_time(self):
        return round(
            self.search_all_time / self.search_count
            if self.embedding_count != 0
            else 0,
            4,
        )

    def hint_cache(self):
        self.hint_cache_count += 1


class Cache:
    """Initialize GPTCache.

    :param similarity_evaluation: SimilarityEvaluation module
    :type similarity_evaluation: gptcache.similarity_evaluation.similarity_evaluation.SimilarityEvaluation

    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.adapter import openai

            cache.init()
            cache.set_openai_key()
    """

    similarity_evaluation: SimilarityEvaluation

    # it should be called when start the cache system
    def __init__(self):
        self.has_init = False
        self.cache_enable_func = None
        self.pre_embedding_func = None
        self.embedding_func = None
        self.data_manager: Optional[DataManager] = None
        self.post_process_messages_func = None
        self.config = Config()
        self.report = Report()
        self.next_cache = None

    def init(
            self,
            cache_enable_func=cache_all,
            pre_embedding_func=last_content,
            embedding_func=string_embedding,
            data_manager: DataManager = get_data_manager(),
            similarity_evaluation=ExactMatchEvaluation(),
            post_process_messages_func=first,
            config=Config(),
            next_cache=None,
    ):
        """Pass parameters to initialize GPTCache.

        :param cache_enable_func: a function to enable cache, defaults to ``cache_all``
        :param pre_embedding_func: a function to preprocess embedding, defaults to ``last_content``
        :param embedding_func: a function to extract embeddings from requests for similarity search, defaults to ``string_embedding``
        :param data_manager: a ``DataManager`` module, defaults to ``get_data_manager()``
        :param similarity_evaluation: a module to calculate embedding similarity, defaults to ``ExactMatchEvaluation()``
        :param post_process_messages_func: a function to post-process messages, defaults to ``first``
        :param config: a module to pass configurations, defaults to ``Config()``
        :param next_cache: customized method for next cache
        """
        self.has_init = True
        self.cache_enable_func = cache_enable_func
        self.pre_embedding_func = pre_embedding_func
        self.embedding_func = embedding_func
        self.data_manager: DataManager = data_manager
        self.similarity_evaluation = similarity_evaluation
        self.post_process_messages_func = post_process_messages_func
        self.config = config
        self.next_cache = next_cache

        @atexit.register
        def close():
            try:
                self.data_manager.close()
            except Exception as e:  # pylint: disable=W0703
                print(e)

    def import_data(self, questions: List[Any], answers: List[Any]) -> None:
        """ Import data to GPTCache

        :param questions: preprocessed question Data
        :param answers: list of answers to questions
        :return: None
        """
        self.data_manager.import_data(
            questions=questions,
            answers=answers,
            embedding_datas=[self.embedding_func(question) for question in questions],
        )

    @staticmethod
    def set_openai_key():
        openai.api_key = os.getenv("OPENAI_API_KEY")


cache = Cache()
