from gpt_cache.view import openai
from gpt_cache.core import cache, Config
from gpt_cache.cache.factory import get_ss_data_manager
from gpt_cache.similarity_evaluation.simple import pair_evaluation
import numpy as np


d = 8


def mock_embeddings(data, **kwargs):
    return np.random.random((1, d)).astype('float32')


def run():
    # milvus
    data_manager = get_ss_data_manager("sqlite", "milvus", dimension=d, max_size=8, clean_size=2)
    # milvus cloud
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
    # you should OPEN it if you FIRST run it
    for i in range(10):
        cache.data_manager.save(f"receiver the foo {i}", cache.embedding_func("foo"))

    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mock_messages,
    )
    print(answer)


if __name__ == '__main__':
    run()
