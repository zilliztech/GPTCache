import atexit
import os
import pytest

from gptcache.adapter import openai
from gptcache.cache.factory import get_ss_data_manager
from gptcache.core import cache, Config
from gptcache.embedding import Towhee
from gptcache.similarity_evaluation.simple import SearchDistanceEvaluation

sqlite_file = "gptcache.db"
faiss_file = "faiss.index"


@atexit.register
def remove_file():
    if os.path.isfile(sqlite_file):
        os.remove(sqlite_file)
    if os.path.isfile(faiss_file):
        os.remove(faiss_file)


class TestSqliteInvalid:

    """
    ******************************************************************
    #  The followings are the exception cases
    ******************************************************************
    """

    @pytest.mark.parametrize("threshold", [-1, 2, 2.0, '0.5'])
    def test_invalid_similarity_threshold(self, threshold):
        """
        target: test init: invalid similarity threshold
        method: input non-num and num which is out of range [0, 1]
        expected: raise exception and report the error
        """
        towhee = Towhee()
        if os.path.isfile(sqlite_file):
            os.remove(sqlite_file)
        if os.path.isfile(faiss_file):
            os.remove(faiss_file)
        data_manager = get_ss_data_manager("sqlite", "faiss",
                                           dimension=towhee.dimension(), max_size=2000)

        def log_time_func(func_name, delta_time):
            print("func `{}` consume time: {:.2f}s".format(func_name, delta_time))

        is_exception = False
        try:
            cache.init(embedding_func=towhee.to_embeddings,
                       data_manager=data_manager,
                       similarity_evaluation=SearchDistanceEvaluation,
                       config=Config(
                           log_time_func=log_time_func,
                           similarity_threshold=threshold,
                       ),
                       )
        except Exception as e:
            print(e)
            is_exception = True

        assert is_exception

    def test_no_openai_key(self):
        """
        target: test no openai key when could not hit in cache
        method: set similarity_threshold as 1 and no openai key
        expected: raise exception and report the error
        """
        towhee = Towhee()
        if os.path.isfile(sqlite_file):
            os.remove(sqlite_file)
        if os.path.isfile(faiss_file):
            os.remove(faiss_file)
        data_manager = get_ss_data_manager("sqlite", "faiss",
                                           dimension=towhee.dimension(), max_size=2000)

        def log_time_func(func_name, delta_time):
            print("func `{}` consume time: {:.2f}s".format(func_name, delta_time))

        cache.init(embedding_func=towhee.to_embeddings,
                   data_manager=data_manager,
                   similarity_evaluation=SearchDistanceEvaluation,
                   config=Config(
                       log_time_func=log_time_func,
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
            is_exception = True

        assert is_exception


class TestSqliteFaiss:

    """
    ******************************************************************
    #  The followings are general cases
    ******************************************************************
    """

    def test_hit_default(self):
        """
        target: test hit the cache function
        method: keep default similarity_threshold
        expected: hit successfully
        """

        towhee = Towhee()
        if os.path.isfile(sqlite_file):
            os.remove(sqlite_file)
        if os.path.isfile(faiss_file):
            os.remove(faiss_file)
        data_manager = get_ss_data_manager("sqlite", "faiss",
                                           dimension=towhee.dimension(), max_size=2000)

        def log_time_func(func_name, delta_time):
            print("func `{}` consume time: {:.2f}s".format(func_name, delta_time))

        cache.init(embedding_func=towhee.to_embeddings,
                   data_manager=data_manager,
                   similarity_evaluation=SearchDistanceEvaluation(),
                   config=Config(
                       log_time_func=log_time_func,
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

    def test_hit(self):
        """
        target: test hit the cache function
        method: set similarity_threshold as 1
        expected: hit successfully
        """

        towhee = Towhee()
        if os.path.isfile(sqlite_file):
            os.remove(sqlite_file)
        if os.path.isfile(faiss_file):
            os.remove(faiss_file)
        data_manager = get_ss_data_manager("sqlite", "faiss",
                                           dimension=towhee.dimension(), max_size=2000)

        def log_time_func(func_name, delta_time):
            print("func `{}` consume time: {:.2f}s".format(func_name, delta_time))

        cache.init(embedding_func=towhee.to_embeddings,
                   data_manager=data_manager,
                   similarity_evaluation=SearchDistanceEvaluation(),
                   config=Config(
                       log_time_func=log_time_func,
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

    def test_miss(self):
        """
        target: test miss the cache function
        method: set similarity_threshold as 0
        expected: raise exception and report the error
        """
        towhee = Towhee()
        if os.path.isfile(sqlite_file):
            os.remove(sqlite_file)
        if os.path.isfile(faiss_file):
            os.remove(faiss_file)
        data_manager = get_ss_data_manager("sqlite", "faiss",
                                           dimension=towhee.dimension(), max_size=2000)

        def log_time_func(func_name, delta_time):
            print("func `{}` consume time: {:.2f}s".format(func_name, delta_time))

        cache.init(embedding_func=towhee.to_embeddings,
                   data_manager=data_manager,
                   similarity_evaluation=SearchDistanceEvaluation,
                   config=Config(
                       log_time_func=log_time_func,
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
            is_exception = True

        assert is_exception




