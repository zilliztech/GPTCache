import atexit
import os
import time

import openai
from .encoder.string import to_embeddings as string_embedding
from .cache.data_manager import DataManager
from .cache.factory import get_data_manager
from .processor.post import first
from .processor.pre import last_content
from .ranker.string import absolute_evaluation


def cache_all(*args, **kwargs):
    return True


def time_cal(func, func_name=None, report_func=None):
    def inner(*args, **kwargs):
        time_start = time.time()
        res = func(*args, **kwargs)
        delta_time = time.time() - time_start
        if cache.config.print_func_time:
            print("func `{}` consume time: {:.2f}s".format(
                func.__name__ if func_name is None else func_name, delta_time))
        if report_func is not None:
            report_func(delta_time)
        return res
    return inner


class Config:
    def __init__(self,
                 print_func_time=False,
                 report_func_time=True,
                 ):
        self.print_func_time = print_func_time
        self.report_func_time = report_func_time


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
    # it should be called when start the cache system
    def __init__(self):
        self.has_init = False
        self.cache_enable_func = None
        self.pre_embedding_func = None
        self.embedding_func = None
        self.data_manager = None
        self.evaluation_func = None
        self.post_process_messages_func = None
        self.similarity_threshold = None
        self.similarity_positive = True
        self.config = None
        self.report = Report()
        self.next_cache = None

    def init(self,
             cache_enable_func=cache_all,
             pre_embedding_func=last_content,
             embedding_func=string_embedding,
             data_manager: DataManager = get_data_manager("map"),
             evaluation_func=absolute_evaluation,
             post_process_messages_func=first,
             similarity_threshold=0.5,
             similarity_positive=True,
             config=Config(),
             next_cache=None,
             **kwargs
             ):
        self.has_init = True
        self.cache_enable_func = cache_enable_func
        self.pre_embedding_func = pre_embedding_func
        self.embedding_func = embedding_func
        self.data_manager: DataManager = data_manager
        self.evaluation_func = evaluation_func
        self.post_process_messages_func = post_process_messages_func
        self.similarity_threshold = similarity_threshold
        self.similarity_positive = similarity_positive
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
