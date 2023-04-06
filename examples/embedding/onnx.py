from gptcache.adapter import openai
from gptcache import cache
from gptcache.manager.factory import get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.embedding import Onnx


def run():
    onnx = Onnx()

    data_manager = get_data_manager('sqlite', 'faiss', dimension=onnx.dimension)

    cache.init(embedding_func=onnx.to_embeddings,
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
