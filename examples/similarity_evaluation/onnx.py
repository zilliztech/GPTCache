from gptcache.adapter import openai
from gptcache import cache
from gptcache.manager.factory import get_data_manager
from gptcache.embedding import Onnx as EmbeddingOnnx
from gptcache.similarity_evaluation import Onnx as EvaluationOnnx


def run():
    onnx = EmbeddingOnnx()
    evaluation_onnx = EvaluationOnnx()

    data_manager = get_data_manager('sqlite', 'faiss', dimension=onnx.dimension)

    cache.init(embedding_func=onnx.to_embeddings,
               data_manager=data_manager,
               similarity_evaluation=evaluation_onnx,
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
