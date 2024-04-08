import base64
import json
import os
import time
from io import BytesIO
from typing import Iterator, Any, List

from gptcache import cache
from gptcache.adapter.adapter import adapt
from gptcache.adapter.base import BaseCacheLLM
from gptcache.manager.scalar_data.base import Answer, DataType
from gptcache.utils import import_openai
from gptcache.utils.error import wrap_error
from gptcache.utils.response import (
    get_stream_message_from_openai_answer,
    get_message_from_openai_answer,
)
from gptcache.utils.token import token_counter

import_openai()

# pylint: disable=C0413
# pylint: disable=E1102
import openai
from litellm import completion

class liteChatCompletion(openai.ChatCompletion, BaseCacheLLM):
    """liteLLM ChatCompletion Wrapper

    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.processor.pre import get_prompt
            # init gptcache
            cache.init()
            cache.set_openai_key()

            from gptcache.adapter import litellm
            # run ChatCompletion model with gptcache
            # supported models: https://litellm.readthedocs.io/en/latest/supported/
            response = litellm.completion(
                          model='gpt-3.5-turbo',
                          messages=[
                            {
                                'role': 'user',
                                'content': "what's github"
                            }],
                        )
            response_content = response['choices'][0]['message']['content']
    """
    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs):
        # use this to make the call to litellm.completion()
        try:
            return completion(*llm_args, **llm_kwargs)
        except openai.OpenAIError as e:
            raise wrap_error(e) from e

    @staticmethod
    def _update_cache_callback(
        llm_data, update_cache_func, *args, **kwargs
    ):  # pylint: disable=unused-argument
        if not isinstance(llm_data, Iterator):
            update_cache_func(
                Answer(get_message_from_openai_answer(llm_data), DataType.STR)
            )
            return llm_data
        else:

            def hook_openai_data(it):
                total_answer = ""
                for item in it:
                    total_answer += get_stream_message_from_openai_answer(item)
                    yield item
                update_cache_func(Answer(total_answer, DataType.STR))

            return hook_openai_data(llm_data)

    @classmethod
    def create(cls, *args, **kwargs):
        chat_cache = kwargs.get("cache_obj", cache)
        enable_token_counter = chat_cache.config.enable_token_counter

        def cache_data_convert(cache_data):
            if enable_token_counter:
                input_token = _num_tokens_from_messages(kwargs.get("messages"))
                output_token = token_counter(cache_data)
                saved_token = [input_token, output_token]
            else:
                saved_token = [0, 0]
            if kwargs.get("stream", False):
                return _construct_stream_resp_from_cache(cache_data, saved_token)
            return _construct_resp_from_cache(cache_data, saved_token)

        kwargs = cls.fill_base_args(**kwargs)
        return adapt(
            cls._llm_handler,
            cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )

def _construct_resp_from_cache(return_message, saved_token):
    return {
        "gptcache": True,
        "saved_token": saved_token,
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


def _construct_stream_resp_from_cache(return_message, saved_token):
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
            "saved_token": saved_token,
        },
    ]

def _construct_text_from_cache(return_text):
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

def _num_tokens_from_messages(messages):
    """Returns the number of tokens used by a list of messages."""
    tokens_per_message = 3
    tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += token_counter(value)
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
