from abc import ABCMeta
from typing import Any, Dict, Callable, Optional


class BaseCacheLLM(metaclass=ABCMeta):
    """Base LLM, When you have enhanced llm without using the original llm api,
    you can use this class as a proxy to use the ability of the cache.

    NOTE: Please make sure that the custom llm returns the same value as the original llm.

    For example, if you use the openai proxy, you perform delay statistics before sending the openai request,
    and then you package this part of the function, so you may have a separate package, which is different from openai.
    If the api request parameters and return results you wrap are the same as the original ones,
    then you can use this class to obtain cache-related capabilities.

    Example:
        .. code-block:: python

            import time

            import openai

            from gptcache import Cache
            from gptcache.adapter import openai as cache_openai


            def proxy_openai_chat_complete(*args, **kwargs):
                start_time = time.time()
                res = openai.ChatCompletion.create(*args, **kwargs)
                print("Consume Time Spent =", round((time.time() - start_time), 2))
                return res


            llm_cache = Cache()

            cache_openai.ChatCompletion.llm = proxy_openai_chat_complete
            cache_openai.ChatCompletion.cache_args = {"cache_obj": llm_cache}

            cache_openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": "What's GitHub?",
                    }
                ],
            )
    """

    llm: Optional[Callable] = None
    """
    On a cache miss, if that variable is set, it will be called;
    if not, it will call the original llm.
    """

    cache_args: Dict[str, Any] = {}
    """
    It can be used to set some cache-related public parameters.
    If you don't want to set the same parameters every time when using cache, say cache_obj, you can use it.
    """

    @classmethod
    def fill_base_args(cls, **kwargs):
        """ Fill the base args to the cache args
        """
        for key, value in cls.cache_args.items():
            if key not in kwargs:
                kwargs[key] = value

        return kwargs
