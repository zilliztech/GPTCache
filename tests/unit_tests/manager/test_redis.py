import random
from unittest.mock import patch

import numpy as np

from gptcache import Cache
from gptcache.adapter import openai
from gptcache.adapter.api import init_similar_cache
from gptcache.embedding import Onnx
from gptcache.manager import VectorBase, manager_factory
from gptcache.manager.vector_data.base import VectorData
from gptcache.processor.pre import last_content
from gptcache.utils.response import get_message_from_openai_answer


def test_redis_vector_store():
    encoder = Onnx()
    dim = encoder.dimension
    vector_base = VectorBase("redis", dimension=dim)
    vector_base.mul_add([VectorData(id=i, data=np.random.rand(dim)) for i in range(10)])

    search_res = vector_base.search(np.random.rand(dim))
    print(search_res)
    assert len(search_res) == 1

    search_res = vector_base.search(np.random.rand(dim), top_k=10)
    print(search_res)
    assert len(search_res) == 10

    vector_base.delete([i for i in range(5)])

    search_res = vector_base.search(np.random.rand(dim), top_k=10)
    print(search_res)
    assert len(search_res) == 5


def test_redis_sqlite():
    redis_cache = Cache()
    encoder = Onnx()
    redis_data_manager = manager_factory(
        "sqlite,redis",
        data_dir=str(random.random()),
        vector_params={"dimension": encoder.dimension},
    )
    init_similar_cache(
        cache_obj=redis_cache,
        pre_func=last_content,
        embedding=encoder,
        data_manager=redis_data_manager,
    )
    question = "what's github"
    expect_answer = "GitHub is an online platform used primarily for version control and coding collaborations."
    with patch("openai.ChatCompletion.create") as mock_create:
        datas = {
            "choices": [
                {
                    "message": {"content": expect_answer, "role": "assistant"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "created": 1677825464,
            "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion.chunk",
        }
        mock_create.return_value = datas

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            cache_obj=redis_cache,
        )

        assert get_message_from_openai_answer(response) == expect_answer, response

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "can you explain what GitHub is"},
        ],
        cache_obj=redis_cache,
    )
    answer_text = get_message_from_openai_answer(response)
    assert answer_text == expect_answer, answer_text
