import time
from typing import Iterator

import openai

from gptcache import CacheError
from gptcache.adapter.adapter import adapt
from gptcache.utils.response import (
    get_stream_message_from_openai_answer,
    get_message_from_openai_answer,
    get_text_from_openai_answer,
)


class ChatCompletion(openai.ChatCompletion):
    """Openai ChatCompletion Wrapper"""

    @classmethod
    def llm_handler(cls, *llm_args, **llm_kwargs):
        try:
            return super().create(*llm_args, **llm_kwargs)
        except openai.error.OpenAIError as e:
            raise CacheError("openai error") from e

    @staticmethod
    def update_cache_callback(llm_data, update_cache_func):
        if not isinstance(llm_data, Iterator):
            update_cache_func(get_message_from_openai_answer(llm_data))
            return llm_data
        else:

            def hook_openai_data(it):
                total_answer = ""
                for item in it:
                    total_answer += get_stream_message_from_openai_answer(item)
                    yield item
                update_cache_func(total_answer)

            return hook_openai_data(llm_data)

    @classmethod
    def create(cls, *args, **kwargs):
        def cache_data_convert(cache_data):
            if kwargs.get("stream", False):
                return construct_stream_resp_from_cache(cache_data)
            return construct_resp_from_cache(cache_data)

        return adapt(
            cls.llm_handler,
            cache_data_convert,
            cls.update_cache_callback,
            *args,
            **kwargs
        )


def construct_resp_from_cache(return_message):
    return {
        "gptcache": True,
        "choices": [
            {
                "message": {"role": "assistant", "content": return_message},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        "object": "chat.completion",
    }


def construct_stream_resp_from_cache(return_message):
    created = int(time.time())
    return [
        {
            "choices": [
                {"delta": {"role": "assistant"}, "finish_reason": None, "index": 0}
            ],
            "created": created,
            "object": "chat.completion.chunk",
        },
        {
            "choices": [
                {
                    "delta": {"content": return_message},
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "created": created,
            "object": "chat.completion.chunk",
        },
        {
            "gptcache": True,
            "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
            "created": created,
            "object": "chat.completion.chunk",
        },
    ]


class Completion(openai.Completion):
    """Openai Completion Wrapper"""

    @classmethod
    def llm_handler(cls, *llm_args, **llm_kwargs):
        return super().create(*llm_args, **llm_kwargs)

    @staticmethod
    def cache_data_convert(cache_data):
        return construct_text_from_cache(cache_data)

    @staticmethod
    def update_cache_callback(llm_data, update_cache_func):
        update_cache_func(get_text_from_openai_answer(llm_data))
        return llm_data

    @classmethod
    def create(cls, *args, **kwargs):
        return adapt(
            cls.llm_handler,
            cls.cache_data_convert,
            cls.update_cache_callback,
            *args,
            **kwargs
        )


def construct_text_from_cache(return_text):
    return {
        "gptcache": True,
        "choices": [
            {
                "text": return_text,
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        "object": "text_completion",
    }
