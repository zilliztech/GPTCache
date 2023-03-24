import os
import openai
from .embedding.string import to_embeddings as string_embedding
from .cache.data_manager import DataManager, MapDataManager
from .similarity_evaluation.string import absolute_evaluation


def cache_all(*args, **kwargs):
    return True


class Cache:
    # it should be called when start the cache system
    def __init__(self):
        self.cache_enable_func = None
        self.embedding_func = None
        self.data_manager = None
        self.evaluation_func = None
        self.similarity_threshold = None
        self.similarity_positive = True

    def init(self,
             cache_enable_func=cache_all,
             embedding_func=string_embedding,
             data_manager: DataManager = MapDataManager("data_map.txt"),
             evaluation_func=absolute_evaluation,
             similarity_threshold=100,
             similarity_positive=True,
             ):
        self.cache_enable_func = cache_enable_func
        self.embedding_func = embedding_func
        self.data_manager: DataManager = data_manager
        self.evaluation_func = evaluation_func
        self.similarity_threshold = similarity_threshold
        self.similarity_positive = similarity_positive
        self.data_manager.init()

    @staticmethod
    def set_openai_key():
        openai.api_key = os.getenv("OPENAI_API_KEY")


cache = Cache()
