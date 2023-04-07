import os

from gptcache.adapter import openai
from gptcache import cache, Config
from gptcache.manager import get_data_manager, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
import numpy as np


d = 8


def mock_embeddings(data, **kwargs):  # pylint: disable=W0613
    return np.random.random((d,)).astype("float32")


def run():
    sqlite_file = "sqlite.db"
    faiss_file = "faiss.index"
    has_data = os.path.isfile(sqlite_file) and os.path.isfile(faiss_file)
    vector_base = VectorBase("faiss", dimension=d, top_k=3)
    data_manager = get_data_manager("sqlite", vector_base, max_size=8, clean_size=2)
    cache.init(
        embedding_func=mock_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
        config=Config(
            similarity_threshold=0,
        ),
    )

    mock_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "foo"},
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


if __name__ == "__main__":
    run()
