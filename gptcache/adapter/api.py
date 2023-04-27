from typing import Any, Optional, Callable, List

from gptcache import Cache, cache, Config
from gptcache.adapter.adapter import adapt
from gptcache.embedding import Onnx
from gptcache.manager import manager_factory
from gptcache.processor.post import first
from gptcache.processor.pre import get_prompt
from gptcache.similarity_evaluation import SearchDistanceEvaluation


def _cache_data_converter(cache_data):
    """For cache results, do nothing"""
    return cache_data


def _update_cache_callback_none(
    llm_data, update_cache_func, *args, **kwargs  # pylint: disable=W0613
) -> None:
    """When updating cached data, do nothing, because currently only cached queries are processed"""
    return None


def _llm_handle_none(*llm_args, **llm_kwargs) -> None:  # pylint: disable=W0613
    """Do nothing on a cache miss"""
    return None


def _update_cache_callback(
    llm_data, update_cache_func, *args, **kwargs
):  # pylint: disable=W0613
    """Save the `llm_data` to cache storage"""
    update_cache_func(llm_data)


def put(prompt: str, data: Any, **kwargs) -> None:
    """put api, put qa pair information to GPTCache
    Please make sure that the `pre_embedding_func` param is `get_prompt` when initializing the cache

    :param prompt: the cache data key, usually question text
    :type prompt: str
    :param data: the cache data value, usually answer text
    :type data: Any
    :param kwargs: list of user-defined parameters
    :type kwargs: Dict

    Example:
        .. code-block:: python

            from gptcache.adapter.api import put
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
    """get api, get the cache data according to the `prompt`
    Please make sure that the `pre_embedding_func` param is `get_prompt` when initializing the cache

    :param prompt: the cache data key, usually question text
    :type prompt: str
    :param kwargs: list of user-defined parameters
    :type kwargs: Dict

    Example:
        .. code-block:: python

            from gptcache.adapter.api import put, get
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


def init_similar_cache(
    data_dir: str = "api_cache",
    cache_obj: Optional[Cache] = None,
    post_func: Callable[[List[Any]], Any] = first,
    config: Config = Config(),
):
    """Provide a quick way to initialize cache for api service

    :param data_dir: cache data storage directory
    :type data_dir: str
    :param cache_obj: specify to initialize the Cache object, if not specified, initialize the global object
    :type cache_obj: Optional[Cache]
    :param post_func: post-processing of the cached result list, the most similar result is taken by default
    :type post_func: Callable[[List[Any]], Any]
    :param config: cache configuration, the core is similar threshold
    :type config: gptcache.Config
    :return: None

    Example:
        .. code-block:: python

            from gptcache.adapter.api import put, get, init_similar_cache

            init_similar_cache()
            put("hello", "foo")
            print(get("hello"))
    """
    onnx = Onnx()
    data_manager = manager_factory(
        "sqlite,faiss", data_dir=data_dir, vector_params={"dimension": onnx.dimension}
    )
    evaluation = SearchDistanceEvaluation()
    cache_obj = cache_obj if cache_obj else cache
    cache_obj.init(
        pre_embedding_func=get_prompt,
        embedding_func=onnx.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=evaluation,
        post_process_messages_func=post_func,
        config=config,
    )
