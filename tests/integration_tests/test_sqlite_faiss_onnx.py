import atexit
import os
import pytest
from base.client_base import Base
from utils.util_log import test_log as log
from common import common_func as cf

from gptcache.adapter import openai
from gptcache.cache.factory import get_data_manager
from gptcache.core import cache, Config
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation


class TestSqliteInvalid(Base):

    """
    ******************************************************************
    #  The followings are the exception cases
    ******************************************************************
    """

    @pytest.mark.parametrize("threshold", [-1, 2, 2.0, '0.5'])
    @pytest.mark.tags("L1")
    def test_invalid_similarity_threshold(self, threshold):
        """
        target: test init: invalid similarity threshold
        method: input non-num and num which is out of range [0, 1]
        expected: raise exception and report the error
        """
        onnx = Onnx()
        data_manager = get_data_manager("sqlite", "faiss",
                                            dimension=onnx.dimension, max_size=2000)
        is_exception = False
        try:
            cache.init(embedding_func=onnx.to_embeddings,
                       data_manager=data_manager,
                       similarity_evaluation=SearchDistanceEvaluation,
                       config=Config(
                           log_time_func=cf.log_time_func,
                           similarity_threshold=threshold,
                       ),
                       )
        except Exception as e:
            log.info(e)
            is_exception = True

        assert is_exception

    @pytest.mark.tags("L2")
    def test_no_openai_key(self):
        """
        target: test no openai key when could not hit in cache
        method: set similarity_threshold as 1 and no openai key
        expected: raise exception and report the error
        """
        onnx = Onnx()
        data_manager = get_data_manager("sqlite", "faiss",
                                            dimension=onnx.dimension, max_size=2000)
        cache.init(embedding_func=onnx.to_embeddings,
                   data_manager=data_manager,
                   similarity_evaluation=SearchDistanceEvaluation,
                   config=Config(
                       log_time_func=cf.log_time_func,
                       similarity_threshold=1,
                   ),
                   )

        is_exception = False
        try:
            openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "what do you feel like chatgpt"}
                ],
            )
        except Exception as e:
            log.info(e)
            is_exception = True

        assert is_exception


class TestSqliteFaiss(Base):

    """
    ******************************************************************
    #  The followings are general cases
    ******************************************************************
    """

    @pytest.mark.tags("L1")
    def test_hit_default(self):
        """
        target: test hit the cache function
        method: keep default similarity_threshold
        expected: hit successfully
        """

        onnx = Onnx()
        data_manager = get_data_manager("sqlite", "faiss",
                                            dimension=onnx.dimension, max_size=2000)
        cache.init(embedding_func=onnx.to_embeddings,
                   data_manager=data_manager,
                   similarity_evaluation=SearchDistanceEvaluation(),
                   config=Config(
                       log_time_func=cf.log_time_func,
                   ),
                   )

        question = "what do you think about chatgpt"
        answer = "chatgpt is a good application"
        cache.data_manager.save(question, answer, cache.embedding_func(question))

        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "what do you feel like chatgpt"}
            ],
        )

    @pytest.mark.tags("L1")
    def test_hit(self):
        """
        target: test hit the cache function
        method: set similarity_threshold as 1
        expected: hit successfully
        """

        onnx = Onnx()
        data_manager = get_data_manager("sqlite", "faiss",
                                            dimension=onnx.dimension, max_size=2000)
        cache.init(embedding_func=onnx.to_embeddings,
                   data_manager=data_manager,
                   similarity_evaluation=SearchDistanceEvaluation(),
                   config=Config(
                       log_time_func=cf.log_time_func,
                       similarity_threshold=0.8,
                   ),
                   )

        question = "what do you think about chatgpt"
        answer = "chatgpt is a good application"
        cache.data_manager.save(question, answer, cache.embedding_func(question))

        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "what do you feel like chatgpt"}
            ],
        )

    @pytest.mark.tags("L1")
    def test_miss(self):
        """
        target: test miss the cache function
        method: set similarity_threshold as 0
        expected: raise exception and report the error
        """
        onnx = Onnx()
        data_manager = get_data_manager("sqlite", "faiss",
                                            dimension=onnx.dimension, max_size=2000)
        cache.init(embedding_func=onnx.to_embeddings,
                   data_manager=data_manager,
                   similarity_evaluation=SearchDistanceEvaluation,
                   config=Config(
                       log_time_func=cf.log_time_func,
                       similarity_threshold=0,
                   ),
                   )

        question = "what do you think about chatgpt"
        answer = "chatgpt is a good application"
        cache.data_manager.save(question, answer, cache.embedding_func(question))

        is_exception = False
        try:
            openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "what do you feel like chatgpt"}
                ],
            )
        except Exception as e:
            log.info(e)
            is_exception = True

        assert is_exception

    @pytest.mark.tags("L1")
    def test_disable_cache(self):
        """
        target: test cache not enabled
        method: set cache enable as false
        expected: hit successfully
        """

        onnx = Onnx()
        data_manager = get_data_manager("sqlite", "faiss",
                                            dimension=onnx.dimension, max_size=2000)
        cache.init(cache_enable_func=cf.disable_cache,
                   embedding_func=onnx.to_embeddings,
                   data_manager=data_manager,
                   similarity_evaluation=SearchDistanceEvaluation(),
                   config=Config(
                       log_time_func=cf.log_time_func,
                   ),
                   )

        question = "what do you think about chatgpt"
        answer = "chatgpt is a good application"
        cache.data_manager.save(question, answer, cache.embedding_func(question))

        is_exception = False
        try:
            openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "what do you feel like chatgpt"}
                ],
            )
        except Exception as e:
            log.info(e)
            is_exception = True

        assert is_exception



