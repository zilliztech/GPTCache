import os
from chromadb.config import Settings

from gptcache.adapter import openai
from gptcache.core import cache, Config
from gptcache.cache.factory import get_ss_data_manager
from gptcache.ranker.simple import pair_evaluation
import numpy as np


d = 8

def mock_embeddings(data, **kwargs):
    return np.random.random((d, )).astype('float32')


def run():
    sqlite_file = "sqlite.db"
    has_data = os.path.isfile(sqlite_file)
    # milvus
    data_manager = get_ss_data_manager("sqlite", "chromadb", max_size=8, clean_size=2, client_settings={})

    cache.init(embedding_func=mock_embeddings,
               data_manager=data_manager,
               evaluation_func=pair_evaluation,
               config=Config(
                       similarity_threshold=10000,
                       similarity_positive=False,
                   ),
               )

    mock_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "foo"}
    ]
    if not has_data:
        for i in range(10):
            question = f"foo{i}"
            answer = f"receiver the foo {i}"
            cache.data_manager.save(question, answer, cache.embedding_func(question))

    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mock_messages,
    )
    print(answer)


if __name__ == '__main__':
    run()
