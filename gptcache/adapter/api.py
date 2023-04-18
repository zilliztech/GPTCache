from typing import Any
from gptcache.adapter.adapter import adapt


def _cache_data_converter(cache_data):
    """For cache results, do nothing"""
    return cache_data


def _update_cache_callback_none(llm_data, update_cache_func) -> None:  # pylint: disable=W0613
    """When updating cached data, do nothing, because currently only cached queries are processed"""
    return None


def _llm_handle_none(*llm_args, **llm_kwargs) -> None:  # pylint: disable=W0613
    """Do nothing on a cache miss"""
    return None


def _update_cache_callback(llm_data, update_cache_func):
    """Save the `llm_data` to cache storage"""
    update_cache_func(llm_data)


def put(prompt: str, data: Any, **kwargs) -> None:
    """save api, save qa pair information to GPTCache
    Please make sure that the `pre_embedding_func` param is `get_prompt` when initializing the cache

    Example:
        .. code-block:: python

            from gptcache.adapter.api import save
            from gptcache.processor.pre import get_prompt

            cache.init(pre_embedding_func=get_prompt)
            put("hello", "foo")
    """
    def llm_handle(*llm_args, **llm_kwargs):  # pylint: disable=W0613
        return data

    adapt(
        llm_handle,
        _cache_data_converter,
        _update_cache_callback,
        cache_skip=True,
        prompt=prompt,
        **kwargs,
    )


def get(prompt: str, **kwargs) -> Any:
    """search api, search the cache data according to the `prompt`
        Please make sure that the `pre_embedding_func` param is `get_prompt` when initializing the cache

        Example:
            .. code-block:: python

                from gptcache.adapter.api import save
                from gptcache.processor.pre import get_prompt

                cache.init(pre_embedding_func=get_prompt)
                put("hello", "foo")
                print(get("hello"))
        """
    res = adapt(
        _llm_handle_none,
        _cache_data_converter,
        _update_cache_callback_none,
        prompt=prompt,
        **kwargs,
    )
    return res
