import base64
import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import List

from gptcache.adapter.adapter import adapt
from gptcache.manager.scalar_data.base import Answer, DataType
from gptcache.utils import (
    import_stability, import_pillow
)
from gptcache.utils.error import CacheError

import_pillow()
import_stability()

from PIL import Image as PILImage  # pylint: disable=C0413
from stability_sdk import client  # pylint: disable=C0413
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation  # pylint: disable=C0413


class StabilityInference(client.StabilityInference):
    """client.StabilityInference Wrapper

    Example:
        .. code-block:: python

            import os
            import io
            from PIL import Image

            from gptcache import cache
            from gptcache.processor.pre import get_prompt
            from gptcache.adapter.stability_sdk import StabilityInference, generation

            # init gptcache
            cache.init(pre_embedding_func=get_prompt)

            # run with gptcache
            os.environ['STABILITY_KEY'] = 'key-goes-here'

            stability_api = StabilityInference(
                key=os.environ['STABILITY_KEY'], # API Key reference.
                verbose=False, # Print debug messages.
                engine="stable-diffusion-xl-beta-v2-2-2", # Set the engine to use for generation.
            )

            answers = stability_api.generate(
                prompt="a cat sitting besides a dog",
                width=256,
                height=256
                )

            for resp in answers:
                for artifact in resp.artifacts:
                    if artifact.type == generation.ARTIFACT_IMAGE:
                        img = Image.open(io.BytesIO(artifact.binary))
                        img.save('path/to/save/image.png')
    """

    def _llm_handler(self, *llm_args, **llm_kwargs):
        try:
            return super().generate(*llm_args, **llm_kwargs)
        except Exception as e:
            raise CacheError("stability error") from e

    def generate(self, *args, **kwargs):
        width = kwargs.get("width", 512)
        height = kwargs.get("height", 512)

        def cache_data_convert(cache_data):
            return _construct_resp_from_cache(cache_data, width=width, height=height)

        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs):  # pylint: disable=unused-argument
            def hook_stream_data(it):
                to_save = []
                for resp in it:
                    for artifact in resp.artifacts:
                        try:
                            if artifact.finish_reason == generation.FILTER:
                                warnings.warn(
                                    "Your request activated the API's safety filters and could not be processed."
                                    "Please modify the prompt and try again.")
                                continue
                        except AttributeError:
                            pass
                        if artifact.type == generation.ARTIFACT_IMAGE:
                            img_b64 = base64.b64encode(artifact.binary)
                            to_save.append(img_b64)
                    yield resp
                update_cache_func(Answer(to_save[0], DataType.IMAGE_BASE64))

            return hook_stream_data(llm_data)

        return adapt(
            self._llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs
        )


def _construct_resp_from_cache(img_64, height, width):
    img_bytes = base64.b64decode((img_64))
    img_file = BytesIO(img_bytes)
    img = PILImage.open(img_file)
    new_size = (width, height)
    if new_size != img.size:
        img = img.resize(new_size)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
    yield MockAnswer(artifacts=[MockArtifact(type=generation.ARTIFACT_IMAGE, binary=img_bytes)])


@dataclass
class MockArtifact:
    type: int
    binary: bytes


@dataclass
class MockAnswer:
    artifacts: List[MockArtifact]


