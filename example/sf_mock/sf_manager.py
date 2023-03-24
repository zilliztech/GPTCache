from scenario_cache.view import openai
from scenario_cache.core import cache
from scenario_cache.cache.data_manager import SFDataManager
from scenario_cache.similarity_evaluation.faiss import faiss_evaluation
import numpy as np


d = 8


def mock_embeddings(data, **kwargs):
    return np.random.random((1, d)).astype('float32')


def run():
    cache.init(embedding_func=mock_embeddings,
               data_manager=SFDataManager("sqlite.db", "faiss.index", d),
               evaluation_func=faiss_evaluation,
               similarity_threshold=10000,
               similarity_positive=False)

    mock_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "foo"}
    ]
    # you should open it if you first run it
    # cache.data_manager.save("receiver the foo", cache.embedding_func({"messages": mock_messages}))

    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mock_messages,
        cache_context={
            "search": {
                "user": "foo"
            }
        },
    )
    print(answer)
    cache.data_manager.close()


if __name__ == '__main__':
    run()
