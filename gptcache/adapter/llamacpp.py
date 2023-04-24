from typing import Optional, List, Dict, Any, Iterator
from functools import partial
import time
import copy

from gptcache.adapter.adapter import adapt
from gptcache.manager.scalar_data.base import DataType, Answer

from gptcache.utils import import_llama_cpp_python


import_llama_cpp_python()

from llama_cpp import Llama  # pylint: disable=wrong-import-position


class LlamaCpp:
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

    def __init__(
            self,
            model_path: str,
            n_ctx: int = 512,
            n_parts: int = -1,
            seed: int = 1337,
            f16_kv: bool = True,
            logits_all: bool = False,
            vocab_only: bool = False,
            use_mmap: bool = True,
            use_mlock: bool = False,
            n_threads: Optional[int] = None,
            n_batch: int = 8,
            last_n_tokens_size: int = 64,
            suffix: Optional[str] = None,
            max_tokens: int = 128,
            temperature: float = 0.8,
            top_p: float = 0.95,
            logprobs: Optional[int] = None,
            echo: bool = False,
            stop: Optional[List[str]] = None,
            repeat_penalty: float = 1.1,
            top_k: int = 40
    ):
        """Load a llama.cpp model from `model_path`.

        :param model_path: Path to the model.
        :type model_path: str
        :param n_ctx: Maximum context size.
        :type n_ctx: int
        :param n_parts: Number of parts to split the model into. If -1, the number of parts is automatically determined.
        :type n_parts: int
        :param seed: Random seed. 0 for random.
        :type seed: int
        :param f16_kv: Use half-precision for key/value cache.
        :type f16_kv: bool
        :param logits_all: Return logits for all tokens, not just the last token.
        :type logits_all: bool
        :param vocab_only: Only load the vocabulary no weights.
        :type vocab_only: bool
        :param use_mmap: Use mmap if possible.
        :type use_mmap: bool
        :param use_mlock: Force the system to keep the model in RAM.
        :type use_mlock: bool
        :param n_threads: Number of threads to use. If None, the number of threads is automatically determined.
        :type n_threads: int
        :param n_batch: Maximum number of prompt tokens to batch together when calling llama_eval.
        :type n_batch: int
        :param last_n_tokens_size: Maximum number of tokens to keep in the last_n_tokens deque.
        :type last_n_tokens_size: int
        :param suffix: A suffix to append to the generated text. If None, no suffix is appended.
        :type suffix: str
        :param max_tokens: The maximum number of tokens to generate.
        :type max_tokens: int
        :param temperature: The temperature to use for sampling.
        :type temperature: float
        :param top_p: The top-p value to use for sampling.
        :type top_p: float
        :param logprobs: The number of logprobs to return. If None, no logprobs are returned.
        :type logprobs: int
        :param echo: Whether to echo the prompt.
        :type echo: bool
        :param stop: A list of strings to stop generation when encountered.
        :type stop: List[str]
        :param repeat_penalty: The penalty to apply to repeated tokens.
        :type repeat_penalty: float
        :param top_k: The top-k value to use for sampling.
        :type top_k: int
        """
        self._llm = Llama(model_path, n_ctx, n_parts, seed, f16_kv,
                          logits_all, vocab_only, use_mmap, use_mlock, False,
                          n_threads, n_batch, last_n_tokens_size)
        self.suffix = suffix
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.logprobs = logprobs
        self.echo = echo
        self.stop = stop if stop is not None else []
        self.repeat_penalty = repeat_penalty
        self.top_k = top_k

    @property
    def _params(self) -> Dict[str, Any]:
        """Get the default parameters for calling llama_cpp."""
        return {
            "suffix": self.suffix,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "logprobs": self.logprobs,
            "echo": self.echo,
            "stop": self.stop,
            "repeat_penalty": self.repeat_penalty,
            "top_k": self.top_k,
        }

    def llm_handler(self, **kwargs):
        return partial(self._llm.__call__, **kwargs)

    def __call__(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
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

        params = copy.deepcopy(self._params)

        if self.stop and stop is not None:
            raise ValueError("Already set the stop param in init")
        elif self.stop:
            params["stop"] = self.stop
        elif stop:
            params["stop"] = stop
        else:
            params["stop"] = []

        params["stream"] = stream
        return adapt(
            self.llm_handler(**params), cache_data_convert, update_cache_callback, prompt=prompt, **kwargs
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
