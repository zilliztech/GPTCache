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


def init_similar_cache(
    data_dir: str = "api_cache",
    cache_obj: Optional[Cache] = None,
    post_func: Callable[[List[Any]], Any] = first,
    config: Config = Config(),
):
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
