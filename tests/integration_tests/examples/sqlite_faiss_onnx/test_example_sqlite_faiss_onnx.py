import os
import time

from gptcache.adapter import openai
from gptcache import cache, Config
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.embedding import Onnx


def test_sqlite_faiss_onnx():
    onnx = Onnx()

    sqlite_file = "sqlite.db"
    faiss_file = "faiss.index"
    has_data = os.path.isfile(sqlite_file) and os.path.isfile(faiss_file)
    cache_base = CacheBase("sqlite")
    vector_base = VectorBase("faiss", dimension=onnx.dimension)
    data_manager = get_data_manager(cache_base, vector_base, max_size=2000)

    def log_time_func(func_name, delta_time):
        print("func `{}` consume time: {:.2f}s".format(func_name, delta_time))

    cache.init(
        embedding_func=onnx.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
        config=Config(
            log_time_func=log_time_func,
            similarity_threshold=0.9
        ),
    )

    if not has_data:
        question = "what do you think about chatgpt"
        answer = "chatgpt is a good application"
        cache.data_manager.save(question, answer, cache.embedding_func(question))

    mock_messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'what do you think chatgpt'}
    ]

    start_time = time.time()
    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mock_messages,
    )
    end_time = time.time()
    print("cache hint time consuming: {:.2f}s".format(end_time - start_time))
    print(answer)

    is_exception = False
    try:
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=mock_messages,
            cache_factor=100,
        )
    except Exception:
        is_exception = True

    assert is_exception

    mock_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "what do you feel like chatgpt"},
    ]
    is_exception = False
    try:
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=mock_messages,
        )
    except Exception:
        is_exception = True

    assert is_exception

    is_exception = False
    try:
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=mock_messages,
            cache_factor=0.5,
        )
    except Exception:
        is_exception = True

    assert not is_exception
