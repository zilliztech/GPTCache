import os

from gptcache.utils.response import get_message_from_openai_answer
from gptcache.adapter import openai
from gptcache import cache, Config
from gptcache.manager import get_data_manager, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
import numpy as np


d = 8


def mock_embeddings(data, **kwargs):  # pylint: disable=W0613
    return np.random.random((d,)).astype("float32")


def test_sqlite_faiss():
    sqlite_file = "sqlite.db"
    faiss_file = "faiss.index"

    if os.path.isfile(sqlite_file):
        os.remove(sqlite_file)
    if os.path.isfile(faiss_file):
        os.remove(faiss_file)

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
    cache.import_data(
        [f"foo{i}" for i in range(10)], [f"receiver the foo {i}" for i in range(10)]
    )

    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mock_messages,
    )
    assert get_message_from_openai_answer(answer)

    cache.flush()
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
    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mock_messages,
    )
    assert get_message_from_openai_answer(answer)
