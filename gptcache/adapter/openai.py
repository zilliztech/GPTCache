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
from gptcache.utils import import_openai, import_pillow
from gptcache.utils.error import wrap_error
from gptcache.utils.response import (
    get_stream_message_from_openai_answer,
    get_message_from_openai_answer,
    get_text_from_openai_answer,
    get_image_from_openai_b64,
    get_image_from_openai_url,
    get_audio_text_from_openai_answer,
)
from gptcache.utils.token import token_counter

import_openai()

# pylint: disable=C0413
# pylint: disable=E1102
import openai


class ChatCompletion(openai.ChatCompletion, BaseCacheLLM):
    """Openai ChatCompletion Wrapper

    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.processor.pre import get_prompt
            # init gptcache
            cache.init()
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
    def _llm_handler(cls, *llm_args, **llm_kwargs):
        try:
            return super().create(*llm_args, **llm_kwargs) if cls.llm is None else cls.llm(*llm_args, **llm_kwargs)
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


class Completion(openai.Completion, BaseCacheLLM):
    """Openai Completion Wrapper

    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.processor.pre import get_prompt
            # init gptcache
            cache.init()
            cache.set_openai_key()

            from gptcache.adapter import openai
            # run Completion model with gptcache
            response = openai.Completion.create(model="text-davinci-003",
                                                prompt="Hello world.")
            response_text = response["choices"][0]["text"]
    """

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs):
        try:
            return super().create(*llm_args, **llm_kwargs) if not cls.llm else cls.llm(*llm_args, **llm_kwargs)
        except openai.OpenAIError as e:
            raise wrap_error(e) from e

    @staticmethod
    def _cache_data_convert(cache_data):
        return _construct_text_from_cache(cache_data)

    @staticmethod
    def _update_cache_callback(
        llm_data, update_cache_func, *args, **kwargs
    ):  # pylint: disable=unused-argument
        update_cache_func(Answer(get_text_from_openai_answer(llm_data), DataType.STR))
        return llm_data

    @classmethod
    def create(cls, *args, **kwargs):
        kwargs = cls.fill_base_args(**kwargs)
        return adapt(
            cls._llm_handler,
            cls._cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
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
                return super(Audio, cls).transcribe(*llm_args, **llm_kwargs)
            except openai.OpenAIError as e:
                raise wrap_error(e) from e

        def cache_data_convert(cache_data):
            return _construct_audio_text_from_cache(cache_data)

        def update_cache_callback(
            llm_data, update_cache_func, *args, **kwargs
        ):  # pylint: disable=unused-argument
            update_cache_func(
                Answer(get_audio_text_from_openai_answer(llm_data), DataType.STR)
            )
            return llm_data

        return adapt(
            llm_handler,
            cache_data_convert,
            update_cache_callback,
            model=model,
            file=file,
            *args,
            **kwargs,
        )

    @classmethod
    def translate(cls, model: str, file: Any, *args, **kwargs):
        def llm_handler(*llm_args, **llm_kwargs):
            try:
                return super(Audio, cls).translate(*llm_args, **llm_kwargs)
            except openai.OpenAIError as e:
                raise wrap_error(e) from e

        def cache_data_convert(cache_data):
            return _construct_audio_text_from_cache(cache_data)

        def update_cache_callback(
            llm_data, update_cache_func, *args, **kwargs
        ):  # pylint: disable=unused-argument
            update_cache_func(
                Answer(get_audio_text_from_openai_answer(llm_data), DataType.STR)
            )
            return llm_data

        return adapt(
            llm_handler,
            cache_data_convert,
            update_cache_callback,
            model=model,
            file=file,
            *args,
            **kwargs,
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
    def _llm_handler(cls, *llm_args, **llm_kwargs):
        try:
            return super().create(*llm_args, **llm_kwargs)
        except openai.OpenAIError as e:
            raise wrap_error(e) from e

    @classmethod
    def create(cls, *args, **kwargs):
        response_format = kwargs.pop("response_format", "url")
        size = kwargs.pop("size", "256x256")

        def cache_data_convert(cache_data):
            return _construct_image_create_resp_from_cache(
                image_data=cache_data, response_format=response_format, size=size
            )

        def update_cache_callback(
            llm_data, update_cache_func, *args, **kwargs
        ):  # pylint: disable=unused-argument
            if response_format == "b64_json":
                img_b64 = get_image_from_openai_b64(llm_data)
                if isinstance(img_b64, str):
                    img_b64 = img_b64.encode("ascii")
                update_cache_func(Answer(img_b64, DataType.IMAGE_BASE64))
            elif response_format == "url":
                update_cache_func(
                    Answer(get_image_from_openai_url(llm_data), DataType.IMAGE_URL)
                )
            return llm_data

        return adapt(
            cls._llm_handler,
            cache_data_convert,
            update_cache_callback,
            response_format=response_format,
            size=size,
            *args,
            **kwargs,
        )


class Moderation(openai.Moderation, BaseCacheLLM):
    """Openai Moderation Wrapper

    Example:
        .. code-block:: python

            from gptcache.adapter import openai
            from gptcache.adapter.api import init_similar_cache
            from gptcache.processor.pre import get_openai_moderation_input

            init_similar_cache(pre_func=get_openai_moderation_input)
            openai.Moderation.create(
                input="I want to kill them.",
            )
    """

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs):
        try:
            return super().create(*llm_args, **llm_kwargs) if not cls.llm else cls.llm(*llm_args, **llm_kwargs)
        except openai.OpenAIError as e:
            raise wrap_error(e) from e

    @classmethod
    def _cache_data_convert(cls, cache_data):
        return json.loads(cache_data)

    @classmethod
    def _update_cache_callback(
        cls, llm_data, update_cache_func, *args, **kwargs
    ):  # pylint: disable=unused-argument
        update_cache_func(Answer(json.dumps(llm_data, indent=4), DataType.STR))
        return llm_data

    @classmethod
    def create(cls, *args, **kwargs):
        kwargs = cls.fill_base_args(**kwargs)
        res = adapt(
            cls._llm_handler,
            cls._cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )

        input_request_param = kwargs.get("input")
        expect_res_len = 1
        if isinstance(input_request_param, List):
            expect_res_len = len(input_request_param)
        if len(res.get("results")) != expect_res_len:
            kwargs["cache_skip"] = True
            res = adapt(
                cls._llm_handler,
                cls._cache_data_convert,
                cls._update_cache_callback,
                *args,
                **kwargs,
            )
        return res


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


def _construct_image_create_resp_from_cache(image_data, response_format, size):
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
        raise AttributeError(
            f"Invalid response_format: {response_format} is not one of ['url', 'b64_json']"
        )

    return {
        "gptcache": True,
        "created": int(time.time()),
        "data": [{response_format: image_data}],
    }


def _construct_audio_text_from_cache(return_text):
    return {
        "gptcache": True,
        "text": return_text,
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
