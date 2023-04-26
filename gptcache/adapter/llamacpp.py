from typing import Optional, List, Iterator
import time

from gptcache.adapter.adapter import adapt
from gptcache.manager.scalar_data.base import DataType, Answer

from gptcache.utils import import_llama_cpp_python


import_llama_cpp_python()

from llama_cpp import Llama  # pylint: disable=wrong-import-position


class LlamaCpp(Llama):
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
        llm = LlamaCpp('./models/7B/ggml-model.bin')
        answer = llm(prompt=question, cache_obj=llm_cache)
    """
    def __call__(
            self,
            prompt: str,
            suffix: Optional[str] = None,
            max_tokens: int = 128,
            temperature: float = 0.8,
            top_p: float = 0.95,
            logprobs: Optional[int] = None,
            echo: bool = False,
            stop: Optional[List[str]] = None,
            repeat_penalty: float = 1.1,
            top_k: int = 40,
            stream: bool = False,
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
                    return total_answer

                return stream_answer(llm_data)

        def cache_data_convert(cache_data):
            if stream:
                return construct_stream_resp_from_cache(cache_data)
            return construct_resp_from_cache(cache_data)

        if stop is None:
            stop = []
        return adapt(
            self.create_completion,
            cache_data_convert,
            update_cache_callback,
            prompt=prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            stream=stream,
            **kwargs
        )


def construct_resp_from_cache(return_message):
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


def construct_stream_resp_from_cache(return_message):
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
