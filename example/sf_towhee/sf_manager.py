import time

from scenario_cache.view import openai
from scenario_cache.core import cache
from scenario_cache.cache.data_manager import SFDataManager
from scenario_cache.similarity_evaluation.faiss import faiss_evaluation
from scenario_cache.embedding.towhee import to_embeddings as towhee_embedding

d = 768


def run():
    cache.init(embedding_func=towhee_embedding,
               data_manager=SFDataManager("sqlite.db", "faiss.index", d),
               evaluation_func=faiss_evaluation,
               similarity_threshold=10000,
               similarity_positive=False)

    # you should OPEN it if you FIRST run it
    # source_messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "what do you think about chatgpt"}
    # ]
    # cache.data_manager.save("chatgpt is a good application", cache.embedding_func({"messages": source_messages}))

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
        cache_context={
            "search": {
                "user": "foo"
            }
        },
    )
    end_time = time.time()
    print("cache hint time consuming: {:.2f}s".format(end_time - start_time))

    print(answer)
    cache.data_manager.close()


if __name__ == '__main__':
    run()
