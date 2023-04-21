from gptcache.utils.error import CacheError
from gptcache.adapter.adapter import adapt
from gptcache.utils import import_replicate
from gptcache.manager.scalar_data.base import DataType, Question, Answer

import_replicate()

import replicate  # pylint: disable=C0413


class Client(replicate.client.Client):
    """replicate.client.Client Wrapper

    Example:
        .. code-block:: python

            from gptcache import cache
            from gptcache.adapter import replicate
            from gptcache.processor.pre import get_input_str
            from gptcache.embedding import Timm
            from gptcache.similarity_evaluation.onnx import OnnxModelEvaluation

            # init gptcache
            timm = Timm('resnet18')
            cache_base = CacheBase('sqlite')
            vector_base = VectorBase('faiss', dimension=timm.dimension)
            object_base = ObjectBase('local', path='./objects')
            data_manager = get_data_manager(cache_base, vector_base, object_base)

            cache.init(
                pre_embedding_func=get_input_image_file_name,
                data_manager=data_manager,
                embedding_func=timm.to_embeddings,
                similarity_evaluation=OnnxModelEvaluation()
                )

            # run replicate clinet with gptcache
            output = replicate.run(
                        "andreasjansson/blip-2:4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608",
                        input={"image": open("/path/to/merlion.png", "rb"),
                               "question": "Which city is this photo taken on?"}
                    )
    """
    def run(self, model_version: str, **kwargs):
        if "input" in kwargs and "question" in kwargs["input"] and "image" in kwargs["input"]:
            cache_context = {"deps": [
                {"name": "text", "data": kwargs["input"]["question"], "dep_type": DataType.STR},
                {"name": "image", "data": kwargs["input"]["image"].name, "dep_type": DataType.STR},
                ]}
        else:
            cache_context = {}
        def llm_handler(*llm_args, **llm_kwargs):
            try:
                return replicate.run(*llm_args, **llm_kwargs)
            except Exception as e:
                raise CacheError("replicate error") from e

        def cache_data_convert(cache_data):
            return cache_data

        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs):  # pylint: disable=unused-argument
            if "input" in kwargs and "question" in kwargs["input"] and "image" in kwargs["input"]:
                question = Question.from_dict({
                    "content": "pre_embedding_data",
                    "deps": [
                        {"name": "text", "data": kwargs["input"]["question"], "dep_type": DataType.STR},
                        {"name": "image", "data": kwargs["input"]["image"].name, "dep_type": DataType.STR},
                    ]
                })
                update_cache_func(Answer(llm_data, DataType.STR), question=question)
            else:
                update_cache_func(llm_data)
            return llm_data


        return adapt(
            llm_handler, cache_data_convert, update_cache_callback,
            model_version=model_version,
            cache_context=cache_context,
            require_object_store=False,
            **kwargs
        )


default_client = Client()
run = default_client.run
