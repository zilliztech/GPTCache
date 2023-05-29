import time
from typing import Iterator

from gptcache.adapter.adapter import adapt
from gptcache.manager.scalar_data.base import DataType, Answer
from gptcache.utils import import_llama_cpp_python

import_llama_cpp_python()

import llama_cpp # pylint: disable=wrong-import-position


class Llama(llama_cpp.Llama):
    """llama.cpp wrapper

        You should have the llama-cpp-python library installed.
        https://github.com/abetlen/llama-cpp-python

    Example:
        .. code-block:: python

            onnx = Onnx()
            m = manager_factory('sqlite,faiss,local', data_dir=root, vector_params={"dimension": onnx.dimension})
            llm_cache = Cache()
            llm_cache.init(
                pre_embedding_func=get_prompt,
                data_manager=m,
                embedding_func=onnx.to_embeddings
            )
            llm = Llama('./models/7B/ggml-model.bin')
            answer = llm(prompt=question, cache_obj=llm_cache)
    """
    def __call__(
            self,
            prompt: str,
            **kwargs
    ):

        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs):  # pylint: disable=unused-argument
            if not isinstance(llm_data, Iterator):
                update_cache_func(Answer(llm_data["choices"][0]["text"], DataType.STR))
                return llm_data
            else:
                def stream_answer(it):
                    total_answer = ""
                    for item in it:
                        total_answer += item["choices"][0]["text"]
                        yield item
                    update_cache_func(Answer(total_answer, DataType.STR))

                return stream_answer(llm_data)

        def cache_data_convert(cache_data):
            if kwargs.get("stream", False):
                return _construct_stream_resp_from_cache(cache_data)
            return _construct_resp_from_cache(cache_data)

        return adapt(
            self.create_completion,
            cache_data_convert,
            update_cache_callback,
            prompt=prompt,
            **kwargs
        )


def _construct_resp_from_cache(return_message):
    return {
        "gptcache": True,
        "choices": [
            {
                "text": return_message,
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        "object": "chat.completion",
    }


def _construct_stream_resp_from_cache(return_message):
    return [
        {
            "gptcache": True,
            "choices": [
                {
                    "text": return_message,
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "created": int(time.time()),
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
            "object": "chat.completion",
        }
    ]
