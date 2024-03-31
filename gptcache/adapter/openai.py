from typing import Any, AsyncGenerator, Iterator

from gptcache import cache
from gptcache.adapter.adapter import adapt
from gptcache.manager.scalar_data.base import Answer, DataType
from gptcache.utils import import_openai
from gptcache.utils.error import wrap_error
from gptcache.utils.response import (
    # get_audio_text_from_openai_answer,
    # get_image_from_openai_b64,
    # get_image_from_openai_url,
    # get_message_from_openai_answer,
    get_message_from_openai_answer2,
    get_stream_message_from_openai_answer,
    # get_stream_message_from_openai_answer2,
    # get_text_from_openai_answer,
)
from gptcache.utils.token import token_counter
from ._util import (
    # _construct_audio_text_from_cache,
    # _construct_image_create_resp_from_cache,
    _construct_resp_from_cache,
    _construct_stream_resp_from_cache,
    # _construct_text_from_cache,
    _num_tokens_from_messages,
)

import_openai()

# pylint: disable=C0413
# pylint: disable=E1102
import openai
from openai import OpenAI


def cache_openai_chat_complete(client: OpenAI, **openai_kwargs: Any):
    def _llm_handler(**llm_kwargs):
        try:
            return client.chat.completions.create(**llm_kwargs)
        except openai.OpenAIError as e:
            raise wrap_error(e) from e

    def _update_cache_callback(
        llm_data, update_cache_func, *args, **kwargs
    ):  # pylint: disable=unused-argument
        if isinstance(llm_data, AsyncGenerator):

            async def hook_openai_data(it):
                total_answer = ""
                async for item in it:
                    total_answer += get_stream_message_from_openai_answer(item)
                    yield item
                update_cache_func(Answer(total_answer, DataType.STR))

            return hook_openai_data(llm_data)
        elif not isinstance(llm_data, Iterator):
            update_cache_func(
                Answer(get_message_from_openai_answer2(llm_data), DataType.STR)
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

    chat_cache = openai_kwargs.get("cache_obj", cache)
    enable_token_counter = chat_cache.config.enable_token_counter

    def cache_data_convert(cache_data):
        if enable_token_counter:
            input_token = _num_tokens_from_messages(openai_kwargs.get("messages"))
            output_token = token_counter(cache_data)
            saved_token = [input_token, output_token]
        else:
            saved_token = [0, 0]
        if openai_kwargs.get("stream", False):
            return _construct_stream_resp_from_cache(cache_data, saved_token)
        return _construct_resp_from_cache(cache_data, saved_token)

    return adapt(
        _llm_handler,
        cache_data_convert,
        _update_cache_callback,
        **openai_kwargs,
    )

deprecated_openai_str = "please update the openai version to 1.x, and use the cache_openai_xxx method"

# @deprecated(deprecated_openai_str)
# class ChatCompletion(openai.ChatCompletion, BaseCacheLLM):
#     """Openai ChatCompletion Wrapper

#     Example:
#         .. code-block:: python

#             from gptcache import cache
#             from gptcache.processor.pre import get_prompt
#             # init gptcache
#             cache.init()
#             cache.set_openai_key()

#             from gptcache.adapter import openai
#             # run ChatCompletion model with gptcache
#             response = openai.ChatCompletion.create(
#                           model='gpt-3.5-turbo',
#                           messages=[
#                             {
#                                 'role': 'user',
#                                 'content': "what's github"
#                             }],
#                         )
#             response_content = response['choices'][0]['message']['content']
#     """

#     @classmethod
#     def _llm_handler(cls, *llm_args, **llm_kwargs):
#         try:
#             return (
#                 super().create(*llm_args, **llm_kwargs)
#                 if cls.llm is None
#                 else cls.llm(*llm_args, **llm_kwargs)
#             )
#         except openai.OpenAIError as e:
#             raise wrap_error(e) from e

#     @classmethod
#     async def _allm_handler(cls, *llm_args, **llm_kwargs):
#         try:
#             return (
#                 (await super().acreate(*llm_args, **llm_kwargs))
#                 if cls.llm is None
#                 else await cls.llm(*llm_args, **llm_kwargs)
#             )
#         except openai.OpenAIError as e:
#             raise wrap_error(e) from e

#     @staticmethod
#     def _update_cache_callback(
#         llm_data, update_cache_func, *args, **kwargs
#     ):  # pylint: disable=unused-argument
#         if isinstance(llm_data, AsyncGenerator):

#             async def hook_openai_data(it):
#                 total_answer = ""
#                 async for item in it:
#                     total_answer += get_stream_message_from_openai_answer(item)
#                     yield item
#                 update_cache_func(Answer(total_answer, DataType.STR))

#             return hook_openai_data(llm_data)
#         elif not isinstance(llm_data, Iterator):
#             update_cache_func(
#                 Answer(get_message_from_openai_answer(llm_data), DataType.STR)
#             )
#             return llm_data
#         else:
#             def hook_openai_data(it):
#                 total_answer = ""
#                 for item in it:
#                     total_answer += get_stream_message_from_openai_answer(item)
#                     yield item
#                 update_cache_func(Answer(total_answer, DataType.STR))

#             return hook_openai_data(llm_data)

#     @classmethod
#     def create(cls, *args, **kwargs):
#         chat_cache = kwargs.get("cache_obj", cache)
#         enable_token_counter = chat_cache.config.enable_token_counter

#         def cache_data_convert(cache_data):
#             if enable_token_counter:
#                 input_token = _num_tokens_from_messages(kwargs.get("messages"))
#                 output_token = token_counter(cache_data)
#                 saved_token = [input_token, output_token]
#             else:
#                 saved_token = [0, 0]
#             if kwargs.get("stream", False):
#                 return _construct_stream_resp_from_cache(cache_data, saved_token)
#             return _construct_resp_from_cache(cache_data, saved_token)

#         kwargs = cls.fill_base_args(**kwargs)
#         return adapt(
#             cls._llm_handler,
#             cache_data_convert,
#             cls._update_cache_callback,
#             *args,
#             **kwargs,
#         )

#     @classmethod
#     async def acreate(cls, *args, **kwargs):
#         chat_cache = kwargs.get("cache_obj", cache)
#         enable_token_counter = chat_cache.config.enable_token_counter

#         def cache_data_convert(cache_data):
#             if enable_token_counter:
#                 input_token = _num_tokens_from_messages(kwargs.get("messages"))
#                 output_token = token_counter(cache_data)
#                 saved_token = [input_token, output_token]
#             else:
#                 saved_token = [0, 0]
#             if kwargs.get("stream", False):
#                 return async_iter(
#                     _construct_stream_resp_from_cache(cache_data, saved_token)
#                 )
#             return _construct_resp_from_cache(cache_data, saved_token)

#         kwargs = cls.fill_base_args(**kwargs)
#         return await aadapt(
#             cls._allm_handler,
#             cache_data_convert,
#             cls._update_cache_callback,
#             *args,
#             **kwargs,
#         )


async def async_iter(input_list):
    for item in input_list:
        yield item


# @deprecated(deprecated_openai_str)
# class Completion(openai.Completion, BaseCacheLLM):
#     """Openai Completion Wrapper

#     Example:
#         .. code-block:: python

#             from gptcache import cache
#             from gptcache.processor.pre import get_prompt
#             # init gptcache
#             cache.init()
#             cache.set_openai_key()

#             from gptcache.adapter import openai
#             # run Completion model with gptcache
#             response = openai.Completion.create(model="text-davinci-003",
#                                                 prompt="Hello world.")
#             response_text = response["choices"][0]["text"]
#     """

#     @classmethod
#     def _llm_handler(cls, *llm_args, **llm_kwargs):
#         try:
#             return (
#                 super().create(*llm_args, **llm_kwargs)
#                 if not cls.llm
#                 else cls.llm(*llm_args, **llm_kwargs)
#             )
#         except openai.OpenAIError as e:
#             raise wrap_error(e) from e

#     @classmethod
#     async def _allm_handler(cls, *llm_args, **llm_kwargs):
#         try:
#             return (
#                 (await super().acreate(*llm_args, **llm_kwargs))
#                 if cls.llm is None
#                 else await cls.llm(*llm_args, **llm_kwargs)
#             )
#         except openai.OpenAIError as e:
#             raise wrap_error(e) from e

#     @staticmethod
#     def _cache_data_convert(cache_data):
#         return _construct_text_from_cache(cache_data)

#     @staticmethod
#     def _update_cache_callback(
#         llm_data, update_cache_func, *args, **kwargs
#     ):  # pylint: disable=unused-argument
#         update_cache_func(Answer(get_text_from_openai_answer(llm_data), DataType.STR))
#         return llm_data

#     @classmethod
#     def create(cls, *args, **kwargs):
#         kwargs = cls.fill_base_args(**kwargs)
#         return adapt(
#             cls._llm_handler,
#             cls._cache_data_convert,
#             cls._update_cache_callback,
#             *args,
#             **kwargs,
#         )

#     @classmethod
#     async def acreate(cls, *args, **kwargs):
#         kwargs = cls.fill_base_args(**kwargs)
#         return await aadapt(
#             cls._allm_handler,
#             cls._cache_data_convert,
#             cls._update_cache_callback,
#             *args,
#             **kwargs,
#         )


# @deprecated(deprecated_openai_str)
# class Audio(openai.Audio):
#     """Openai Audio Wrapper

#     Example:
#         .. code-block:: python

#             from gptcache import cache
#             from gptcache.processor.pre import get_file_bytes
#             # init gptcache
#             cache.init(pre_embedding_func=get_file_bytes)
#             cache.set_openai_key()

#             from gptcache.adapter import openai
#             # run audio transcribe model with gptcache
#             audio_file= open("/path/to/audio.mp3", "rb")
#             transcript = openai.Audio.transcribe("whisper-1", audio_file)

#             # run audio transcribe model with gptcache
#             audio_file= open("/path/to/audio.mp3", "rb")
#             transcript = openai.Audio.translate("whisper-1", audio_file)
#     """

#     @classmethod
#     def transcribe(cls, model: str, file: Any, *args, **kwargs):
#         def llm_handler(*llm_args, **llm_kwargs):
#             try:
#                 return super(Audio, cls).transcribe(*llm_args, **llm_kwargs)
#             except openai.OpenAIError as e:
#                 raise wrap_error(e) from e

#         def cache_data_convert(cache_data):
#             return _construct_audio_text_from_cache(cache_data)

#         def update_cache_callback(
#             llm_data, update_cache_func, *args, **kwargs
#         ):  # pylint: disable=unused-argument
#             update_cache_func(
#                 Answer(get_audio_text_from_openai_answer(llm_data), DataType.STR)
#             )
#             return llm_data

#         return adapt(
#             llm_handler,
#             cache_data_convert,
#             update_cache_callback,
#             model=model,
#             file=file,
#             *args,
#             **kwargs,
#         )

#     @classmethod
#     def translate(cls, model: str, file: Any, *args, **kwargs):
#         def llm_handler(*llm_args, **llm_kwargs):
#             try:
#                 return super(Audio, cls).translate(*llm_args, **llm_kwargs)
#             except openai.OpenAIError as e:
#                 raise wrap_error(e) from e

#         def cache_data_convert(cache_data):
#             return _construct_audio_text_from_cache(cache_data)

#         def update_cache_callback(
#             llm_data, update_cache_func, *args, **kwargs
#         ):  # pylint: disable=unused-argument
#             update_cache_func(
#                 Answer(get_audio_text_from_openai_answer(llm_data), DataType.STR)
#             )
#             return llm_data

#         return adapt(
#             llm_handler,
#             cache_data_convert,
#             update_cache_callback,
#             model=model,
#             file=file,
#             *args,
#             **kwargs,
#         )


# @deprecated(deprecated_openai_str)
# class Image(openai.Image):
#     """Openai Image Wrapper

#     Example:
#         .. code-block:: python

#             from gptcache import cache
#             from gptcache.processor.pre import get_prompt
#             # init gptcache
#             cache.init(pre_embedding_func=get_prompt)
#             cache.set_openai_key()

#             from gptcache.adapter import openai
#             # run image generation model with gptcache
#             response = openai.Image.create(
#               prompt="a white siamese cat",
#               n=1,
#               size="256x256"
#             )
#             response_url = response['data'][0]['url']
#     """

#     @classmethod
#     def _llm_handler(cls, *llm_args, **llm_kwargs):
#         try:
#             return super().create(*llm_args, **llm_kwargs)
#         except openai.OpenAIError as e:
#             raise wrap_error(e) from e

#     @classmethod
#     def create(cls, *args, **kwargs):
#         response_format = kwargs.pop("response_format", "url")
#         size = kwargs.pop("size", "256x256")

#         def cache_data_convert(cache_data):
#             return _construct_image_create_resp_from_cache(
#                 image_data=cache_data, response_format=response_format, size=size
#             )

#         def update_cache_callback(
#             llm_data, update_cache_func, *args, **kwargs
#         ):  # pylint: disable=unused-argument
#             if response_format == "b64_json":
#                 img_b64 = get_image_from_openai_b64(llm_data)
#                 if isinstance(img_b64, str):
#                     img_b64 = img_b64.encode("ascii")
#                 update_cache_func(Answer(img_b64, DataType.IMAGE_BASE64))
#             elif response_format == "url":
#                 update_cache_func(
#                     Answer(get_image_from_openai_url(llm_data), DataType.IMAGE_URL)
#                 )
#             return llm_data

#         return adapt(
#             cls._llm_handler,
#             cache_data_convert,
#             update_cache_callback,
#             response_format=response_format,
#             size=size,
#             *args,
#             **kwargs,
#         )


# @deprecated(deprecated_openai_str)
# class Moderation(openai.Moderation, BaseCacheLLM):
#     """Openai Moderation Wrapper

#     Example:
#         .. code-block:: python

#             from gptcache.adapter import openai
#             from gptcache.adapter.api import init_similar_cache
#             from gptcache.processor.pre import get_openai_moderation_input

#             init_similar_cache(pre_func=get_openai_moderation_input)
#             openai.Moderation.create(
#                 input="I want to kill them.",
#             )
#     """

#     @classmethod
#     def _llm_handler(cls, *llm_args, **llm_kwargs):
#         try:
#             return (
#                 super().create(*llm_args, **llm_kwargs)
#                 if not cls.llm
#                 else cls.llm(*llm_args, **llm_kwargs)
#             )
#         except openai.OpenAIError as e:
#             raise wrap_error(e) from e

#     @classmethod
#     def _cache_data_convert(cls, cache_data):
#         return json.loads(cache_data)

#     @classmethod
#     def _update_cache_callback(
#         cls, llm_data, update_cache_func, *args, **kwargs
#     ):  # pylint: disable=unused-argument
#         update_cache_func(Answer(json.dumps(llm_data, indent=4), DataType.STR))
#         return llm_data

#     @classmethod
#     def create(cls, *args, **kwargs):
#         kwargs = cls.fill_base_args(**kwargs)
#         res = adapt(
#             cls._llm_handler,
#             cls._cache_data_convert,
#             cls._update_cache_callback,
#             *args,
#             **kwargs,
#         )

#         input_request_param = kwargs.get("input")
#         expect_res_len = 1
#         if isinstance(input_request_param, List):
#             expect_res_len = len(input_request_param)
#         if len(res.get("results")) != expect_res_len:
#             kwargs["cache_skip"] = True
#             res = adapt(
#                 cls._llm_handler,
#                 cls._cache_data_convert,
#                 cls._update_cache_callback,
#                 *args,
#                 **kwargs,
#             )
#         return res
