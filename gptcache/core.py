import atexit
import os
import time

import openai
from .embedding.string import to_embeddings as string_embedding
from .cache.data_manager import DataManager
from .cache.factory import get_data_manager
from .processor.post import first
from .processor.pre import last_content
from .similarity_evaluation.similarity_evaluation import SimilarityEvaluation
from .similarity_evaluation.string import AbsoluteEvaluation
from .utils.error import CacheError


def cache_all(*args, **kwargs):
    return True


def time_cal(func, func_name=None, report_func=None):
    def inner(*args, **kwargs):
        time_start = time.time()
        res = func(*args, **kwargs)
        delta_time = time.time() - time_start
        if cache.config.log_time_func:
            cache.config.log_time_func(func.__name__ if func_name is None else func_name, delta_time)
        if report_func is not None:
            report_func(delta_time)
        return res

    return inner


class Config:

    def __init__(self,
                 log_time_func=None,
                 similarity_threshold=0.8,
                 ):
        if similarity_threshold < 0 or similarity_threshold > 1:
            raise CacheError("Invalid the similarity threshold param")
        self.log_time_func = log_time_func
        self.similarity_threshold = similarity_threshold


class Report:
    def __init__(self):
        self.embedding_all_time = 0
        self.embedding_count = 0
        self.search_all_time = 0
        self.search_count = 0
        self.hint_cache_count = 0

    def embedding(self, delta_time):
        self.embedding_all_time += delta_time
        self.embedding_count += 1

    def search(self, delta_time):
        self.search_all_time += delta_time
        self.search_count += 1

    def average_embedding_time(self):
        return round(self.embedding_all_time / self.embedding_count if self.embedding_count != 0 else 0, 4)

    def average_search_time(self):
        return round(self.search_all_time / self.search_count if self.embedding_count != 0 else 0, 4)

    def hint_cache(self):
        self.hint_cache_count += 1


class Cache:
    similarity_evaluation: SimilarityEvaluation

    # it should be called when start the cache system
    def __init__(self):
        self.has_init = False
        self.cache_enable_func = None
        self.pre_embedding_func = None
        self.embedding_func = None
        self.data_manager = None
        self.post_process_messages_func = None
        self.config = Config()
        self.report = Report()
        self.next_cache = None

    def init(self,
             cache_enable_func=cache_all,
             pre_embedding_func=last_content,
             embedding_func=string_embedding,
             data_manager: DataManager = get_data_manager("map"),
             similarity_evaluation=AbsoluteEvaluation(),
             post_process_messages_func=first,
             config=Config(),
             next_cache=None,
             **kwargs
             ):
        self.has_init = True
        self.cache_enable_func = cache_enable_func
        self.pre_embedding_func = pre_embedding_func
        self.embedding_func = embedding_func
        self.data_manager: DataManager = data_manager
        self.similarity_evaluation = similarity_evaluation
        self.post_process_messages_func = post_process_messages_func
        self.data_manager.init(**kwargs)
        self.config = config
        self.next_cache = next_cache

        @atexit.register
        def close():
            try:
                self.data_manager.close()
            except Exception as e:
                print(e)

    @staticmethod
    def set_openai_key():
        openai.api_key = os.getenv("OPENAI_API_KEY")


cache = Cache()
