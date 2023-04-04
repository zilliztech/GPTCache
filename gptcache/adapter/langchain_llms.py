from typing import Optional, List
from .adapter import adapt


class LangChainLLMs:
    def __init__(self, llm):
        self._llm = llm

    def __call__(self, prompt: str, stop: Optional[List[str]] = None, **kwargs):
        return adapt(self, cache_data_convert, update_cache_callback, prompt=prompt, stop=stop, **kwargs)


def cache_data_convert(cache_data):
    return {
        "gptcache": True,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": cache_data
                },
                "finish_reason": "stop",
                "index": 0
            }
        ]
    }    


def update_cache_callback(llm_data, update_cache_func):
    update_cache_func(llm_data)
    return llm_data

from langchain.llms import OpenAI


llm_handler = LangChainLLMs(OpenAI(model_name="text-ada-001", n=2, best_of=2))
