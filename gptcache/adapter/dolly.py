from typing import Any

from gptcache.adapter.adapter import adapt
from gptcache.manager.scalar_data.base import Answer, DataType
from gptcache.utils import import_huggingface, import_torch

import_torch()
import_huggingface()

from transformers import pipeline  # pylint: disable=wrong-import-position


class Dolly:
    """Wrapper for Dolly (https://github.com/databrickslabs/dolly.git).

    Example using from_model:
        .. code-block:: python

            from gptcache import cache
            from gptcache.processor.pre import get_inputs
            cache.init(pre_embedding_func=get_inputs)

            from gptcache.adapter.dolly import Dolly
            dolly = Dolly.from_model(
                model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device=0
            )

    Example passing pipeline in directly:
        .. code-block:: python

            import torch
            from transformers import pipeline
            from gptcache import cache
            from gptcache.processor.pre import get_inputs
            cache.init(pre_embedding_func=get_inputs)
            from gptcache.adapter.dolly import Dolly

            pipe = pipeline(
                model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device=0
            )
            dolly = Dolly(pipe)
    """

    def __init__(self, dolly_pipeline: Any):
        self._dolly_pipeline = dolly_pipeline

    @classmethod
    def from_model(cls, model: str, **kwargs):
        pipe = pipeline(model=model, **kwargs)
        return cls(pipe)

    def __call__(self, prompt: str, **kwargs):
        return adapt(
            self._dolly_pipeline,
            _cache_data_convert,
            _update_cache_callback,
            inputs=prompt,
            **kwargs
        )


def _cache_data_convert(cache_data):
    return [{"generated_text": cache_data, "gptcache": True}]


def _update_cache_callback(llm_data, update_cache_func, *args, **kwargs):  # pylint: disable=unused-argument
    update_cache_func(Answer(llm_data[0]["generated_text"], DataType.STR))
    return llm_data
