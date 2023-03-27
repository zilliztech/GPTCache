import time

from gpt_cache.view import openai
from gpt_cache.core import cache
from gpt_cache.cache.data_manager import SFDataManager
from gpt_cache.similarity_evaluation.faiss import faiss_evaluation
from gpt_cache.embedding.towhee import Towhee


def run():
    towhee = Towhee()
    cache.init(embedding_func=towhee.to_embeddings,
               data_manager=SFDataManager("sqlite.db", "faiss.index", towhee.dimension()),
               evaluation_func=faiss_evaluation,
               similarity_threshold=10000,
               similarity_positive=False)

    # you should OPEN it if you FIRST run it
    cache.data_manager.save("chatgpt is a good application", cache.embedding_func("what do you think about chatgpt"))

    # distance 77
    mock_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "what do you feel like chatgpt"}
    ]

    # distance 21
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
