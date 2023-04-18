import base64
from io import BytesIO

from gptcache.adapter.adapter import adapt
from gptcache.manager.scalar_data.base import Answer, AnswerType
from gptcache.utils.error import CacheError
from gptcache.utils import (
    import_pillow, import_diffusers, import_huggingface
    )

import_pillow()
import_huggingface()
import_diffusers()

from PIL import Image  # pylint: disable=C0413
import diffusers  # pylint: disable=C0413


class StableDiffusionPipeline(diffusers.StableDiffusionPipeline):
    """Diffuser StableDiffusionPipeline Wrapper"""

    def llm_handler(self, *llm_args, **llm_kwargs):
        try:
            return super().__call__(*llm_args, **llm_kwargs)
        except Exception as e:
            raise CacheError("diffuser error") from e

    def __call__(self, *args, **kwargs):
        def cache_data_convert(cache_data):
            return construct_resp_from_cache(cache_data)

        def update_cache_callback(llm_data, update_cache_func):
            img = llm_data["images"][0]
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue())
            update_cache_func(Answer(img_b64, AnswerType.IMAGE_BASE64))
            return llm_data

        return adapt(
            self.llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs
        )


def construct_resp_from_cache(img_64):
    im_bytes = base64.b64decode(img_64)   # im_bytes is a binary image
    im_file = BytesIO(im_bytes)  # convert image to file-like object
    img = Image.open(im_file)
    return diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput(images=[img], nsfw_content_detected=None)

