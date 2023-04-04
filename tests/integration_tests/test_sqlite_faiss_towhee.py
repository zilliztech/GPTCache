import atexit
import os

from gptcache.adapter import openai
from gptcache.cache.factory import get_ss_data_manager
from gptcache.core import cache, Config
from gptcache.encoder import Towhee
from gptcache.ranker.simple import SearchDistanceEvaluation

sqlite_file = "sqlite.db"
faiss_file = "faiss.index"


@atexit.register
def remove_file():
    if os.path.isfile(sqlite_file):
        os.remove(sqlite_file)
    if os.path.isfile(faiss_file):
        os.remove(faiss_file)


def test_hint():
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


def test_miss():
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


test_hint()

test_miss()
