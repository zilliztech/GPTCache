from gptcache.utils import import_cxoracle
import_cxoracle()

import cx_Oracle
import numpy as np

from gptcache.adapter import openai
from gptcache.core import cache, Config
from gptcache.cache.factory import get_data_manager
from gptcache.similarity_evaluation.simple import SearchDistanceEvaluation


d = 8
has_data = False


def mock_embeddings(data, **kwargs):
    return np.random.random((d, )).astype('float32')


def run():
    # init client first
    cx_Oracle.init_oracle_client(lib_dir="/Users/root/Downloads/instantclient_19_8")

    # `sql_url` defaults to 'oracle+cx_oracle://oracle:123456@127.0.0.1:1521/?service_name=helowin&encoding=UTF-8&nencoding=UTF-8'
    data_manager = get_data_manager("oracle", "milvus", dimension=d, max_size=8, clean_size=2)
    cache.init(embedding_func=mock_embeddings,
               data_manager=data_manager,
               similarity_evaluation=SearchDistanceEvaluation(),
               config=Config(
                       similarity_threshold=0,
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
