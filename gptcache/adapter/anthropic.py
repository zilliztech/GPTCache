import os
from types import SimpleNamespace
from typing import AsyncGenerator, Iterator

from gptcache.adapter.adapter import aadapt, adapt
from gptcache.adapter.base import BaseCacheLLM
from gptcache.manager.scalar_data.base import Answer, DataType
from gptcache.utils import import_anthropic
from gptcache.utils.error import wrap_error
from gptcache.utils.response import (
    get_message_from_anthropic_answer,
    get_stream_message_from_anthropic_answer,
)

import_anthropic()

# pylint: disable=C0413
import anthropic


class ChatCompletion(BaseCacheLLM):
    """Anthropic Claude messages wrapper.

    The interface mirrors :class:`gptcache.adapter.openai.ChatCompletion`, so that
    you can cache Claude ``messages`` requests with minimal code changes.

    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.adapter import anthropic

            cache.init()
            # make sure ANTHROPIC_API_KEY is exported

            response = anthropic.ChatCompletion.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1024,
                messages=[{"role": "user", "content": "what's github"}],
            )
            answer = response.content[0].text
    """

    client = None
    aclient = None

    @classmethod
    def _sync_client(cls):
        if cls.client is not None:
            return cls.client
        return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    @classmethod
    def _async_client(cls):
        if cls.aclient is not None:
            return cls.aclient
        return anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs):
        try:
            if cls.llm is not None:
                return cls.llm(*llm_args, **llm_kwargs)  # pylint: disable=not-callable
            return cls._sync_client().messages.create(*llm_args, **llm_kwargs)
        except anthropic.APIError as e:
            raise wrap_error(e) from e

    @classmethod
    async def _allm_handler(cls, *llm_args, **llm_kwargs):
        try:
            if cls.llm is not None:
                return cls.llm(*llm_args, **llm_kwargs)  # pylint: disable=not-callable
            return await cls._async_client().messages.create(*llm_args, **llm_kwargs)
        except anthropic.APIError as e:
            raise wrap_error(e) from e

    @staticmethod
    def _update_cache_callback(
        llm_data, update_cache_func, *args, **kwargs
    ):  # pylint: disable=unused-argument
        if isinstance(llm_data, AsyncGenerator):

            async def hook_async_anthropic_data(it):
                total_answer = ""
                async for item in it:
                    total_answer += get_stream_message_from_anthropic_answer(item)
                    yield item
                update_cache_func(Answer(total_answer, DataType.STR))

            return hook_async_anthropic_data(llm_data)
        if isinstance(llm_data, Iterator):

            def hook_anthropic_data(it):
                total_answer = ""
                for item in it:
                    total_answer += get_stream_message_from_anthropic_answer(item)
                    yield item
                update_cache_func(Answer(total_answer, DataType.STR))

            return hook_anthropic_data(llm_data)
        update_cache_func(
            Answer(get_message_from_anthropic_answer(llm_data), DataType.STR)
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


def _construct_resp_from_cache(return_message, model=None):
    try:
        from anthropic.types import Message, TextBlock, Usage  # pylint: disable=import-outside-toplevel

        return Message(
            id="chatcmpl-gptcache",
            content=[TextBlock(type="text", text=return_message)],
            role="assistant",
            model=model or "gptcache",
            stop_reason="end_turn",
            stop_sequence=None,
            type="message",
            usage=Usage(input_tokens=0, output_tokens=0),
        )
    except Exception:  # pylint: disable=W0703
        content_block = SimpleNamespace(type="text", text=return_message)
        return SimpleNamespace(
            gptcache=True,
            id="chatcmpl-gptcache",
            content=[content_block],
            role="assistant",
            model=model or "gptcache",
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
        )


def _construct_stream_resp_from_cache(return_message, model=None):
    del model
    return [
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text=return_message),
        ),
        SimpleNamespace(type="message_stop"),
    ]
