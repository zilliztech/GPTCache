from gpt_cache.view import openai
from gpt_cache.core import cache, Config
from gpt_cache.cache.data_manager import SFDataManager
from gpt_cache.similarity_evaluation.faiss import faiss_evaluation
import numpy as np


d = 8


def mock_embeddings(data, **kwargs):
    return np.random.random((1, d)).astype('float32')


def run():
    cache.init(embedding_func=mock_embeddings,
               data_manager=SFDataManager("sqlite.db", "faiss.index", d),
               evaluation_func=faiss_evaluation,
               similarity_threshold=10000,
               similarity_positive=False,
               config=Config(top_k=3),
               )

    mock_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "foo"}
    ]
    # you should OPEN it if you FIRST run it
    # cache.data_manager.save("receiver the foo", cache.embedding_func("foo"))

    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mock_messages,
    )
    print(answer)


if __name__ == '__main__':
    run()
