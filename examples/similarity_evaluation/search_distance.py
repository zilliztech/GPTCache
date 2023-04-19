from gptcache.adapter import openai
from gptcache import cache
from gptcache.manager import get_data_manager, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.embedding import Onnx


def run():
    onnx = Onnx()

    vector_base = VectorBase('faiss', dimension=onnx.dimension)
    data_manager = get_data_manager('sqlite', vector_base)

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
