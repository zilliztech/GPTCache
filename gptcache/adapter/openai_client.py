import os
import time
from types import SimpleNamespace
from typing import AsyncGenerator, Iterator

from gptcache.adapter.adapter import aadapt, adapt
from gptcache.adapter.base import BaseCacheLLM
from gptcache.manager.scalar_data.base import Answer, DataType
from gptcache.utils import import_openai_client
from gptcache.utils.error import wrap_error

import_openai_client()

# pylint: disable=C0413
from openai import AsyncOpenAI, OpenAI


class ChatCompletion(BaseCacheLLM):
    """OpenAI ``ChatCompletion`` wrapper for the modern (>=1.0) OpenAI SDK.

    This adapter targets the client-based interface introduced in ``openai>=1.0``,
    i.e. ``client.chat.completions.create``, instead of the legacy
    ``openai.ChatCompletion.create`` interface used by
    :class:`gptcache.adapter.openai.ChatCompletion`.

    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.adapter import openai_client

            cache.init()

            response = openai_client.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "what's github"}],
            )
            answer = response.choices[0].message.content
    """

    client = None
    aclient = None

    @classmethod
    def _sync_client(cls):
        if cls.client is not None:
            return cls.client
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @classmethod
    def _async_client(cls):
        if cls.aclient is not None:
            return cls.aclient
        return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs):
        try:
            if cls.llm is not None:
                return cls.llm(*llm_args, **llm_kwargs)  # pylint: disable=not-callable
            return cls._sync_client().chat.completions.create(*llm_args, **llm_kwargs)
        except Exception as e:  # pylint: disable=W0703
            raise wrap_error(e) from e

    @classmethod
    async def _allm_handler(cls, *llm_args, **llm_kwargs):
        try:
            if cls.llm is not None:
                return cls.llm(*llm_args, **llm_kwargs)  # pylint: disable=not-callable
            return await cls._async_client().chat.completions.create(
                *llm_args, **llm_kwargs
            )
        except Exception as e:  # pylint: disable=W0703
            raise wrap_error(e) from e

    @staticmethod
    def _update_cache_callback(
        llm_data, update_cache_func, *args, **kwargs
    ):  # pylint: disable=unused-argument
        if isinstance(llm_data, AsyncGenerator):

            async def hook_async_openai_data(it):
                total_answer = ""
                async for item in it:
                    total_answer += get_stream_message_from_openai_client_answer(item)
                    yield item
                update_cache_func(Answer(total_answer, DataType.STR))

            return hook_async_openai_data(llm_data)
        if isinstance(llm_data, Iterator):

            def hook_openai_data(it):
                total_answer = ""
                for item in it:
                    total_answer += get_stream_message_from_openai_client_answer(item)
                    yield item
                update_cache_func(Answer(total_answer, DataType.STR))

            return hook_openai_data(llm_data)
        update_cache_func(
            Answer(get_message_from_openai_client_answer(llm_data), DataType.STR)
        )
        return llm_data

    @classmethod
    def create(cls, *args, **kwargs):
        model = kwargs.get("model")

        def cache_data_convert(cache_data):
            if kwargs.get("stream", False):
                return _construct_stream_resp_from_cache(cache_data, model)
            return _construct_resp_from_cache(cache_data, model)

        kwargs = cls.fill_base_args(**kwargs)
        return adapt(
            cls._llm_handler,
            cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )

    @classmethod
    async def acreate(cls, *args, **kwargs):
        model = kwargs.get("model")

        def cache_data_convert(cache_data):
            if kwargs.get("stream", False):
                return async_iter(_construct_stream_resp_from_cache(cache_data, model))
            return _construct_resp_from_cache(cache_data, model)

        kwargs = cls.fill_base_args(**kwargs)
        return await aadapt(
            cls._allm_handler,
            cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )


async def async_iter(input_list):
    for item in input_list:
        yield item


def get_message_from_openai_client_answer(resp):
    try:
        return resp.choices[0].message.content
    except (AttributeError, KeyError, TypeError):
        return resp["choices"][0]["message"]["content"]


def get_stream_message_from_openai_client_answer(chunk):
    try:
        delta = chunk.choices[0].delta
    except (AttributeError, KeyError, TypeError):
        delta = chunk["choices"][0]["delta"]
    if delta is None:
        return ""
    content = getattr(delta, "content", "")
    return content or ""


def _construct_resp_from_cache(return_message, model=None):
    message = SimpleNamespace(role="assistant", content=return_message)
    choice = SimpleNamespace(index=0, finish_reason="stop", message=message)
    return SimpleNamespace(
        gptcache=True,
        id="chatcmpl-gptcache",
        choices=[choice],
        created=int(time.time()),
        model=model or "gptcache",
        object="chat.completion",
        usage=SimpleNamespace(
            completion_tokens=0, prompt_tokens=0, total_tokens=0
        ),
    )


def _construct_stream_resp_from_cache(return_message, model=None):
    created = int(time.time())
    return [
        SimpleNamespace(
            id="chatcmpl-gptcache",
            choices=[
                SimpleNamespace(
                    index=0, finish_reason=None,
                    delta=SimpleNamespace(role="assistant", content=None),
                )
            ],
            created=created,
            model=model or "gptcache",
            object="chat.completion.chunk",
        ),
        SimpleNamespace(
            id="chatcmpl-gptcache",
            choices=[
                SimpleNamespace(
                    index=0, finish_reason=None,
                    delta=SimpleNamespace(role=None, content=return_message),
                )
            ],
            created=created,
            model=model or "gptcache",
            object="chat.completion.chunk",
        ),
        SimpleNamespace(
            gptcache=True,
            id="chatcmpl-gptcache",
            choices=[
                SimpleNamespace(
                    index=0, finish_reason="stop",
                    delta=SimpleNamespace(role=None, content=None),
                )
            ],
            created=created,
            model=model or "gptcache",
            object="chat.completion.chunk",
        ),
    ]
