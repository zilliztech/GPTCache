import os
import time

from gptcache.adapter import openai
from gptcache.core import cache, Config
from gptcache.cache.factory import get_ss_data_manager
from gptcache.similarity_evaluation.simple import SearchDistanceEvaluation
from gptcache.embedding import Towhee


def run():
    towhee = Towhee()
    # chinese model
    # towhee = Towhee(model="uer/albert-base-chinese-cluecorpussmall")

    sqlite_file = "gptcache.db"
    faiss_file = "faiss.index"
    has_data = os.path.isfile(sqlite_file) and os.path.isfile(faiss_file)
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

    if not has_data:
        question = "what do you think about chatgpt"
        answer = "chatgpt is a good application"
        cache.data_manager.save(question, answer, cache.embedding_func(question))

    mock_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "what do you feel like chatgpt"}
    ]

    # mock_messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "what do you think chatgpt"}
    # ]

    start_time = time.time()
    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mock_messages,
    )
    end_time = time.time()
    print("cache hint time consuming: {:.2f}s".format(end_time - start_time))

    print(answer)


if __name__ == '__main__':
    run()
