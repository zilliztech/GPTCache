from typing import Optional, List
from .adapter import adapt


class LangChainLLMs:
    def __init__(self, llm):
        self._llm = llm

    def __call__(self, prompt: str, stop: Optional[List[str]] = None, **kwargs):
        # TODO handle the array result
        return adapt(self._llm, cache_data_convert, update_cache_callback, prompt=prompt, stop=stop, **kwargs)


def cache_data_convert(cache_data):
    return cache_data


def update_cache_callback(llm_data, update_cache_func):
    update_cache_func(llm_data)
    return llm_data
