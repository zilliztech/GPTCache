import time
from typing import Iterator

import base64
from io import BytesIO
import os

import openai


from gptcache import CacheError
from gptcache.adapter.adapter import adapt
from gptcache.utils.response import (
    get_stream_message_from_openai_answer,
    get_message_from_openai_answer,
    get_text_from_openai_answer,
    get_image_from_openai_b64,
    get_image_from_openai_url
)
from gptcache.utils import import_pillow

import_pillow()

from PIL import Image as PILImage # pylint: disable=C0413


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

class Image(openai.Image):
    """Openai Image Wrapper"""

    @classmethod
    def create(cls, *args, **kwargs):
        def llm_handler(*llm_args, **llm_kwargs):
            try:
                return openai.Image.create(*llm_args, **llm_kwargs)
            except Exception as e:
                raise CacheError("openai error") from e

        def cache_data_convert(cache_data):
            return construct_image_create_resp_from_cache(
                image_data=cache_data,
                response_format=kwargs["response_format"],
                size=kwargs["size"]
                )

        def update_cache_callback(llm_data, update_cache_func):
            if kwargs["response_format"] == "b64_json":
                update_cache_func(get_image_from_openai_b64(llm_data))
                return llm_data
            elif kwargs["response_format"] == "url":
                update_cache_func(get_image_from_openai_url(llm_data))
                return llm_data

        return adapt(
            llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs
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


def construct_image_create_resp_from_cache(image_data, response_format, size):
    img_bytes = base64.b64decode((image_data))
    img_file = BytesIO(img_bytes)  # convert image to file-like object
    img = PILImage.open(img_file)
    new_size = tuple(int(a) for a in size.split("x"))
    if new_size != img.size:
        img = img.resize(new_size)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
    else:
        buffered = img_file

    if response_format == "url":
        target_url = os.path.abspath(str(int(time.time())) + ".jpeg")
        with open(target_url, "wb") as f:
            f.write(buffered.getvalue())
        image_data = target_url
    elif response_format == "b64_json":
        image_data = base64.b64encode(buffered.getvalue())
    else:
        raise AttributeError(f"Invalid response_format: {response_format} is not one of ['url', 'b64_json']")

    return {
        "created": int(time.time()),
        "data": [
            {response_format: image_data}
        ]
        }
