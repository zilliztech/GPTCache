import os
import numpy as np

from gptcache.adapter import openai
from gptcache import cache
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation


d = 8


def mock_embeddings(data, **kwargs):
    return np.random.random((d, )).astype('float32')


def run():
    scalar_stores = [
        CacheBase('sqlite', sql_url='sqlite:///./sqlite.db'),
        CacheBase('postgresql', sql_url='postgresql+psycopg2://postgres:123456@127.0.0.1:5432/postgres'),
        CacheBase('mysql', sql_url='mysql+pymysql://root:123456@127.0.0.1:3306/mysql'),
        CacheBase('mariadb', sql_url='mariadb+pymysql://root:123456@127.0.0.1:3307/mysql'),
        CacheBase('sqlserver', sql_url='ssql+pyodbc://sa:Strongpsw_123@127.0.0.1:1434/msdb?driver=ODBC+Driver+17+for+SQL+Server'),
        CacheBase('oracle', sql_url='oracle+cx_oracle://oracle:123456@127.0.0.1:1521/?service_name=helowin&encoding=UTF-8&nencoding=UTF-8'),
        CacheBase('dynamo'),
    ]

    for scalar_store in scalar_stores:
        if os.path.exists('faiss.index'):
            os.remove('faiss.index')
        vector_base = VectorBase('faiss', dimension=d)
        data_manager = get_data_manager(scalar_store, vector_base)
        cache.init(embedding_func=mock_embeddings,
                   data_manager=data_manager,
                   similarity_evaluation=SearchDistanceEvaluation(),
                   )
        cache.set_openai_key()

        answer = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'user', 'content': 'what is chatgpt'}
            ],
        )
        print('answer:', answer)

        answer = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'user', 'content': 'what is chatgpt'}
            ],
        )
        print('answer cached:', answer)


if __name__ == '__main__':
    run()
