import os

from gptcache.view import openai
from gptcache.core import cache, Config
from gptcache.cache.factory import get_ss_data_manager
from gptcache.similarity_evaluation.simple import pair_evaluation
import numpy as np


d = 8


def mock_embeddings(data, **kwargs):
    return np.random.random((d, )).astype('float32')


def run():
    sqlite_file = "sqlite.db"
    has_data = os.path.isfile(sqlite_file)
    # milvus
    data_manager = get_ss_data_manager("sqlite", "milvus", dimension=d, max_size=8, clean_size=2)
    # zilliz cloud
    # data_manager = get_ss_data_manager("sqlite", "milvus", dimension=d, max_size=8, clean_size=2,
    #                                    host="xxx.zillizcloud.com",
    #                                    port=19530,
    #                                    user="xxx", password="xxx", is_https=True,
    #                                    )
    cache.init(embedding_func=mock_embeddings,
               data_manager=data_manager,
               evaluation_func=pair_evaluation,
               similarity_threshold=10000,
               similarity_positive=False,
               config=Config(),
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
