import os
from typing import Any, Optional, Callable, List

import gptcache.processor.pre
import gptcache.processor.post
from gptcache.utils.yaml import yaml
from gptcache import Cache, cache, Config
from gptcache.processor.post import first
from gptcache.adapter.adapter import adapt
from gptcache.manager import manager_factory
from gptcache.processor.pre import get_prompt
from gptcache.embedding import Onnx, Huggingface, SBERT, FastText, Data2VecAudio, Timm, ViT, OpenAI, Cohere
from gptcache.similarity_evaluation import (
    SearchDistanceEvaluation, NumpyNormEvaluation, OnnxModelEvaluation,
    ExactMatchEvaluation, KReciprocalEvaluation
)


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
    """search api, search the cache data according to the `prompt`
    Please make sure that the `pre_embedding_func` param is `get_prompt` when initializing the cache

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


def init_similar_cache_from_config(
    data_dir: str = "api_cache",
    cache_obj: Optional[Cache] = None,
    config: Config = Config(),
    config_dir: str = None
):

    if config_dir:
        with open(config_dir, "r", encoding="utf-8") as f:
            init_conf = yaml.load(f, Loader=yaml.Loader)
    else:
        init_conf = {}

    model_src = init_conf.get("model_source", "onnx")
    model_name = init_conf.get("model_name")
    model_config = init_conf.get("model_config", {})
    if model_name:
        model_config["model"] = model_name
    embedding_model = _get_model(model_src, os.getenv("API_KEY", None), model_config)

    sca = init_conf.get("scalar_storage", "sqlite")
    vec = init_conf.get("vector_storage", "faiss")
    obj = init_conf.get("object_stroage")
    data_manager = manager_factory(
        ",".join([sca, vec, obj] if obj else [sca, vec]),
        data_dir=data_dir,
        vector_params={"dimension": embedding_model.dimension}
    )

    eval_strategy = init_conf.get("evaluation", "distance")
    eval_kws = init_conf.get("evaluation_kws")
    evaluation = _get_eval(eval_strategy, eval_kws)

    cache_obj = cache_obj if cache_obj else cache

    pre_prcocess = init_conf.get("pre_function", "get_prompt")
    pre_func = _get_pre_func(pre_prcocess)

    post_process = init_conf.get("post_function", "first")
    post_func = _get_post_func(post_process)

    cache_obj.init(
        pre_embedding_func=pre_func,
        embedding_func=embedding_model.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=evaluation,
        post_process_messages_func=post_func,
        config=config,
    )


def _get_model(model_src, api_key, model_config):
    model_src = model_src.lower()

    if model_src == "onnx":
        return Onnx(**model_config)
    if model_src == "huggingface":
        return Huggingface(**model_config)
    if model_src == "sbert":
        return SBERT(**model_config)
    if model_src == "fasttext":
        return FastText(**model_config)
    if model_src == "data2vecaudio":
        return Data2VecAudio(**model_config)
    if model_src == "timm":
        return Timm(**model_config)
    if model_src == "vit":
        return ViT(**model_config)
    if model_src == "openai":
        return OpenAI(api_key=api_key, **model_config)
    if model_src == "cohere":
        return Cohere(api_key=api_key, **model_config)


def _get_eval(strategy, kws):
    strategy = strategy.lower()

    if "distance" in strategy:
        return SearchDistanceEvaluation(**kws) if kws else SearchDistanceEvaluation()
    if "np" in strategy:
        return NumpyNormEvaluation(**kws) if kws else NumpyNormEvaluation()
    if "exact" in strategy:
        return ExactMatchEvaluation()
    if "onnx" in strategy:
        return OnnxModelEvaluation(**kws) if kws else OnnxModelEvaluation()
    if "kreciprocal" in strategy:
        return KReciprocalEvaluation(**kws)


def _get_pre_func(pre_prcocess):
    return getattr(gptcache.processor.pre, pre_prcocess)


def _get_post_func(post_process):
    return getattr(gptcache.processor.post, post_process)
