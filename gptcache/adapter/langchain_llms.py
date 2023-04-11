from typing import Optional, List, Any

from gptcache.adapter.adapter import adapt
from gptcache.utils import import_pydantic, import_langchain

import_pydantic()
import_langchain()

# pylint: disable=C0413
from pydantic import BaseModel
from langchain.llms.base import LLM


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
            from gptcache.adapter.langchain_llms import LangChainLLMs
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


def cache_data_convert(cache_data):
    return cache_data


def update_cache_callback(llm_data, update_cache_func):
    update_cache_func(llm_data)
    return llm_data
