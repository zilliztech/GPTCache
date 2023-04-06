from gptcache.adapter import openai
from gptcache import cache
from gptcache.cache.factory import get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
import numpy as np


d = 8


def mock_embeddings(data, **kwargs):
    return np.random.random((d, )).astype('float32')


def run():
    scalar_stores = [
        'sqlite',          # `sql_url` defaults to 'sqlite:///./sqlite.db'
        'postgresql',      # `sql_url` defaults to 'postgresql+psycopg2://postgres:123456@127.0.0.1:5432/postgres'
        'mysql',           # `sql_url` defaults to 'mysql+pymysql://root:123456@127.0.0.1:3306/mysql'
        'mariadb',         # `sql_url` defaults to 'mariadb+pymysql://root:123456@127.0.0.1:3307/mysql'
        'sqlserver',       # `sql_url` defaults to 'mssql+pyodbc://sa:Strongpsw_123@127.0.0.1:1434/msdb?driver=ODBC+Driver+17+for+SQL+Server'
        'oracle',          # `sql_url` defaults to 'oracle+cx_oracle://oracle:123456@127.0.0.1:1521/?service_name=helowin&encoding=UTF-8&nencoding=UTF-8'
    ]

    for scalar_store in scalar_stores:
        data_manager = get_data_manager(scalar_store, 'faiss', dimension=d)
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
        print(answer)


if __name__ == '__main__':
    run()
