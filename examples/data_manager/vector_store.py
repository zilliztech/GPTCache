from gptcache.adapter import openai
from gptcache import cache
from gptcache.manager.factory import get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
import numpy as np


d = 8


def mock_embeddings(data, **kwargs):
    return np.random.random((d, )).astype('float32')


def run():
    vector_stores = [
        'faiss',
        'milvus',
        'chromadb',
    ]
    for vector_store in vector_stores:
        data_manager = get_data_manager('sqlite', vector_store, dimension=d)

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
