import time
from typing import Iterator, Any

import os
import openai

import base64
from io import BytesIO

from gptcache.utils import import_pillow
from gptcache.utils.error import CacheError
from gptcache.adapter.adapter import adapt
from gptcache.manager.scalar_data.base import Answer, DataType
from gptcache.utils.response import (
    get_stream_message_from_openai_answer,
    get_message_from_openai_answer,
    get_text_from_openai_answer,
    get_image_from_openai_b64,
    get_image_from_openai_url,
    get_audio_text_from_openai_answer,
)


class ChatCompletion(openai.ChatCompletion):
    """Openai ChatCompletion Wrapper

    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.processor.pre import get_prompt
            # init gptcache
            cache.init(pre_embedding_func=get_prompt)
            cache.set_openai_key()

            from gptcache.adapter import openai
            # run ChatCompletion model with gptcache
            response = openai.ChatCompletion.create(
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
    def llm_handler(cls, *llm_args, **llm_kwargs):
        try:
            return super().create(*llm_args, **llm_kwargs)
        except openai.error.OpenAIError as e:
            raise CacheError("openai error") from e

    @staticmethod
    def update_cache_callback(llm_data, update_cache_func, *args, **kwargs):  # pylint: disable=unused-argument
        if not isinstance(llm_data, Iterator):
            update_cache_func(Answer(get_message_from_openai_answer(llm_data), DataType.STR))
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


class Completion(openai.Completion):
    """Openai Completion Wrapper

    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.processor.pre import get_prompt
            # init gptcache
            cache.init(pre_embedding_func=get_prompt)
            cache.set_openai_key()

            from gptcache.adapter import openai
            # run Completion model with gptcache
            response = openai.Completion.create(model="text-davinci-003",
                                                prompt="Hello world.")
            response_text = response["choices"][0]["text"]
    """

    @classmethod
    def llm_handler(cls, *llm_args, **llm_kwargs):
        return super().create(*llm_args, **llm_kwargs)

    @staticmethod
    def cache_data_convert(cache_data):
        return construct_text_from_cache(cache_data)

    @staticmethod
    def update_cache_callback(llm_data, update_cache_func, *args, **kwargs):  # pylint: disable=unused-argument
        update_cache_func(Answer(get_text_from_openai_answer(llm_data), DataType.STR))
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


class Audio(openai.Audio):
    """Openai Audio Wrapper

    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.processor.pre import get_file_bytes
            # init gptcache
            cache.init(pre_embedding_func=get_file_bytes)
            cache.set_openai_key()

            from gptcache.adapter import openai
            # run audio transcribe model with gptcache
            audio_file= open("/path/to/audio.mp3", "rb")
            transcript = openai.Audio.transcribe("whisper-1", audio_file)

            # run audio transcribe model with gptcache
            audio_file= open("/path/to/audio.mp3", "rb")
            transcript = openai.Audio.translate("whisper-1", audio_file)
    """
    @classmethod
    def transcribe(cls, model: str, file: Any, *args, **kwargs):
        def llm_handler(*llm_args, **llm_kwargs):
            try:
                return openai.Audio.transcribe(*llm_args, **llm_kwargs)
            except Exception as e:
                raise CacheError("openai error") from e

        def cache_data_convert(cache_data):
            return construct_audio_text_from_cache(cache_data)

        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs):  # pylint: disable=unused-argument
            update_cache_func(Answer(get_audio_text_from_openai_answer(llm_data), DataType.STR))
            return llm_data

        return adapt(
            llm_handler, cache_data_convert, update_cache_callback, model=model, file=file, *args, **kwargs
        )

    @classmethod
    def translate(cls, model: str, file: Any, *args, **kwargs):
        def llm_handler(*llm_args, **llm_kwargs):
            try:
                return openai.Audio.translate(*llm_args, **llm_kwargs)
            except Exception as e:
                raise CacheError("openai error") from e

        def cache_data_convert(cache_data):
            return construct_audio_text_from_cache(cache_data)

        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs): # pylint: disable=unused-argument
            update_cache_func(Answer(get_audio_text_from_openai_answer(llm_data), DataType.STR))
            return llm_data

        return adapt(
            llm_handler, cache_data_convert, update_cache_callback, model=model, file=file, *args, **kwargs
        )


class Image(openai.Image):
    """Openai Image Wrapper

    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.processor.pre import get_prompt
            # init gptcache
            cache.init(pre_embedding_func=get_prompt)
            cache.set_openai_key()

            from gptcache.adapter import openai
            # run image generation model with gptcache
            response = openai.Image.create(
              prompt="a white siamese cat",
              n=1,
              size="256x256"
            )
            response_url = response['data'][0]['url']
    """

    @classmethod
    def create(cls, *args, **kwargs):
        response_format = kwargs.pop("response_format", "url")
        size = kwargs.pop("size", "256x256")

        def llm_handler(*llm_args, **llm_kwargs):
            try:
                return openai.Image.create(*llm_args, **llm_kwargs)
            except Exception as e:
                raise CacheError("openai error") from e

        def cache_data_convert(cache_data):
            return construct_image_create_resp_from_cache(
                image_data=cache_data,
                response_format=response_format,
                size=size
                )

        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs):  # pylint: disable=unused-argument
            if response_format == "b64_json":
                img_b64 = get_image_from_openai_b64(llm_data)
                if isinstance(img_b64, str):
                    img_b64 = img_b64.encode("ascii")
                update_cache_func(Answer(img_b64, DataType.IMAGE_BASE64))
            elif response_format == "url":
                update_cache_func(Answer(get_image_from_openai_url(llm_data), DataType.IMAGE_URL))
            return llm_data

        return adapt(
            llm_handler, cache_data_convert, update_cache_callback, response_format=response_format, size=size, *args, **kwargs
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
    import_pillow()
    from PIL import Image as PILImage  # pylint: disable=C0415

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
        image_data = base64.b64encode(buffered.getvalue()).decode("ascii")
    else:
        raise AttributeError(f"Invalid response_format: {response_format} is not one of ['url', 'b64_json']")

    return {
        "gptcache": True,
        "created": int(time.time()),
        "data": [
            {response_format: image_data}
        ]
        }


def construct_audio_text_from_cache(return_text):
    return {
        "gptcache": True,
        "text": return_text,
    }
