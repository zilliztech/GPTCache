# pylint: disable=wrong-import-position
from typing import Any, Optional, Callable

import gptcache.processor.post
import gptcache.processor.pre
from gptcache import Cache, cache, Config
from gptcache.adapter.adapter import adapt
from gptcache.embedding import (
    Onnx,
    Huggingface,
    SBERT,
    FastText,
    Data2VecAudio,
    Timm,
    ViT,
    OpenAI,
    Cohere,
    Rwkv,
    PaddleNLP,
    UForm,
)
from gptcache.embedding.base import BaseEmbedding
from gptcache.manager import manager_factory
from gptcache.manager.data_manager import DataManager
from gptcache.processor.context import (
    SummarizationContextProcess,
    SelectiveContextProcess,
    ConcatContextProcess,
)
from gptcache.processor.post import temperature_softmax
from gptcache.processor.pre import get_prompt
from gptcache.similarity_evaluation import (
    SearchDistanceEvaluation,
    NumpyNormEvaluation,
    OnnxModelEvaluation,
    ExactMatchEvaluation,
    KReciprocalEvaluation,
    SimilarityEvaluation,
    CohereRerankEvaluation,
    SequenceMatchEvaluation,
    TimeEvaluation,
    SbertCrossencoderEvaluation
)
from gptcache.utils import import_ruamel


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
    pre_func: Callable = get_prompt,
    embedding: Optional[BaseEmbedding] = None,
    data_manager: Optional[DataManager] = None,
    evaluation: Optional[SimilarityEvaluation] = None,
    post_func: Callable = temperature_softmax,
    config: Config = Config(),
):
    """Provide a quick way to initialize cache for api service

    :param data_dir: cache data storage directory
    :type data_dir: str
    :param cache_obj: specify to initialize the Cache object, if not specified, initialize the global object
    :type cache_obj: Optional[Cache]
    :param pre_func: pre-processing of the cache input text
    :type pre_func: Callable
    :param embedding: embedding object
    :type embedding: BaseEmbedding
    :param data_manager: data manager object
    :type data_manager: DataManager
    :param evaluation: similarity evaluation object
    :type evaluation: SimilarityEvaluation
    :param post_func: post-processing of the cached result list, the most similar result is taken by default
    :type post_func: Callable[[List[Any]], Any]
    :param config: cache configuration, the core is similar threshold
    :type config: Config
    :return: None

    Example:
        .. code-block:: python

            from gptcache.adapter.api import put, get, init_similar_cache

            init_similar_cache()
            put("hello", "foo")
            print(get("hello"))
    """
    if not embedding:
        embedding = Onnx()
    if not data_manager:
        data_manager = manager_factory(
            "sqlite,faiss",
            data_dir=data_dir,
            vector_params={"dimension": embedding.dimension},
        )
    if not evaluation:
        evaluation = SearchDistanceEvaluation()
    cache_obj = cache_obj if cache_obj else cache
    cache_obj.init(
        pre_embedding_func=pre_func,
        embedding_func=embedding.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=evaluation,
        post_process_messages_func=post_func,
        config=config,
    )


def init_similar_cache_from_config(config_dir: str, cache_obj: Optional[Cache] = None):
    import_ruamel()
    from ruamel.yaml import YAML  # pylint: disable=C0415

    if config_dir:
        with open(config_dir, "r", encoding="utf-8") as f:
            yaml = YAML(typ="unsafe", pure=True)
            init_conf = yaml.load(f)
    else:
        init_conf = {}

    # Due to the problem with the first naming, it is reserved to ensure compatibility
    embedding = init_conf.get("model_source", "")
    if not embedding:
        embedding = init_conf.get("embedding", "onnx")
    # ditto
    embedding_config = init_conf.get("model_config", {})
    if not embedding_config:
        embedding_config = init_conf.get("embedding_config", {})
    embedding_model = _get_model(embedding, embedding_config)

    storage_config = init_conf.get("storage_config", {})
    storage_config.setdefault("manager", "sqlite,faiss")
    storage_config.setdefault("data_dir", "gptcache_data")
    storage_config.setdefault("vector_params", {})
    storage_config["vector_params"] = storage_config["vector_params"] or {}
    storage_config["vector_params"]["dimension"] = embedding_model.dimension
    data_manager = manager_factory(**storage_config)

    eval_strategy = init_conf.get("evaluation", "distance")
    # Due to the problem with the first naming, it is reserved to ensure compatibility
    eval_config = init_conf.get("evaluation_kws", {})
    if not eval_config:
        eval_config = init_conf.get("evaluation_config", {})
    evaluation = _get_eval(eval_strategy, eval_config)

    cache_obj = cache_obj if cache_obj else cache

    pre_process = init_conf.get("pre_context_function")
    if pre_process:
        pre_func = _get_pre_context_function(
            pre_process, init_conf.get("pre_context_config")
        )
        pre_func = pre_func.pre_process
    else:
        pre_process = init_conf.get("pre_function", "get_prompt")
        pre_func = _get_pre_func(pre_process)

    post_process = init_conf.get("post_function", "first")
    post_func = _get_post_func(post_process)

    config_kws = init_conf.get("config", {}) or {}
    config = Config(**config_kws)

    cache_obj.init(
        pre_embedding_func=pre_func,
        embedding_func=embedding_model.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=evaluation,
        post_process_messages_func=post_func,
        config=config,
    )

    return init_conf


def _get_model(model_src, model_config=None):
    model_src = model_src.lower()
    model_config = model_config or {}

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
        return OpenAI(**model_config)
    if model_src == "cohere":
        return Cohere(**model_config)
    if model_src == "rwkv":
        return Rwkv(**model_config)
    if model_src == "paddlenlp":
        return PaddleNLP(**model_config)
    if model_src == "uform":
        return UForm(**model_config)


def _get_eval(strategy, kws=None):
    strategy = strategy.lower()
    kws = kws or {}

    if "distance" in strategy:
        return SearchDistanceEvaluation(**kws)
    if "np" in strategy:
        return NumpyNormEvaluation(**kws)
    if "exact" in strategy:
        return ExactMatchEvaluation()
    if "onnx" in strategy:
        return OnnxModelEvaluation(**kws)
    if "kreciprocal" in strategy:
        return KReciprocalEvaluation(**kws)
    if "cohere" in strategy:
        return CohereRerankEvaluation(**kws)
    if "sequence_match" in strategy:
        return SequenceMatchEvaluation(**kws)
    if "time" in strategy:
        return TimeEvaluation(**kws)
    if "sbert_crossencoder" in strategy:
        return SbertCrossencoderEvaluation(**kws)


def _get_pre_func(pre_process):
    return getattr(gptcache.processor.pre, pre_process)


def _get_pre_context_function(pre_context_process, kws=None):
    pre_context_process = pre_context_process.lower()
    kws = kws or {}
    if pre_context_process in "summarization":
        return SummarizationContextProcess(**kws)
    if pre_context_process in "selective":
        return SelectiveContextProcess(**kws)
    if pre_context_process in "concat":
        return ConcatContextProcess()


def _get_post_func(post_process):
    return getattr(gptcache.processor.post, post_process)
