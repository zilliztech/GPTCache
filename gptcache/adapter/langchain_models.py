from typing import Optional, List, Any

from gptcache.adapter.adapter import adapt
from gptcache.utils import import_pydantic, import_langchain
from gptcache.manager.scalar_data.base import Answer, DataType

import_pydantic()
import_langchain()

# pylint: disable=C0413
from pydantic import BaseModel
from langchain.llms.base import LLM
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, LLMResult, AIMessage, ChatGeneration, ChatResult


class LangChainLLMs(LLM, BaseModel):
    """LangChain LLM Wrapper.

    :param llm: LLM from langchain.llms.
    :type llm: Any

    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.processor.pre import get_prompt
            # init gptcache
            cache.init(pre_embedding_func=get_prompt)
            cache.set_openai_key()

            from langchain.llms import OpenAI
            from gptcache.adapter.langchain_models import LangChainLLMs
            # run llm with gptcache
            llm = LangChainLLMs(llm=OpenAI(temperature=0))
            llm("Hello world")
    """

    llm: Any

    @property
    def _llm_type(self) -> str:
        return "gptcache_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return adapt(
            self.llm,
            cache_data_convert,
            update_cache_callback,
            prompt=prompt,
            stop=stop,
            **kwargs
        )

    def __call__(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return self._call(prompt=prompt, stop=stop, **kwargs)


# pylint: disable=protected-access
class LangChainChat(BaseChatModel, BaseModel):
    """LangChain LLM Wrapper.

    :param chat: LLM from langchain.chat_models.
    :type chat: Any

    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.processor.pre import get_prompt
            # init gptcache
            cache.init(pre_embedding_func=get_prompt)
            cache.set_openai_key()
            from langchain.chat_models import ChatOpenAI
            from gptcache.adapter.langchain_models import LangChainChat
            # run chat model with gptcache
            chat = LangChainLLMs(chat=ChatOpenAI(temperature=0))
            chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
    """

    chat: Any

    def _generate(self, messages: Any, stop: Optional[List[str]] = None, **kwargs):
        return adapt(
            self.chat._generate,
            cache_msg_data_convert,
            update_cache_msg_callback,
            messages=messages,
            stop=stop,
            **kwargs
        )

    async def _agenerate(self, messages: List[List[BaseMessage]], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        return adapt(
            self.chat._agenerate,
            cache_msg_data_convert,
            update_cache_msg_callback,
            messages=messages,
            stop=stop,
            **kwargs
        )

    def __call__(self, messages: Any, stop: Optional[List[str]] = None, **kwargs):
        res = self._generate(messages=messages, stop=stop, **kwargs)
        return res.generations[0].message


def cache_data_convert(cache_data):
    return cache_data


def update_cache_callback(llm_data, update_cache_func, *args, **kwargs):  # pylint: disable=unused-argument
    update_cache_func(Answer(llm_data, DataType.STR))
    return llm_data


def cache_msg_data_convert(cache_data):
    llm_res = ChatResult(generations=[ChatGeneration(text="",
                                                     generation_info=None,
                                                     message=AIMessage(content=cache_data, additional_kwargs={}))],
                         llm_output=None)
    return llm_res


def update_cache_msg_callback(llm_data, update_cache_func, *args, **kwargs):  # pylint: disable=unused-argument
    update_cache_func(llm_data.generations[0].text)
    return llm_data
