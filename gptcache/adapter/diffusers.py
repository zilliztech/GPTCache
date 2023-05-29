import base64
from io import BytesIO

from gptcache.adapter.adapter import adapt
from gptcache.manager.scalar_data.base import Answer, DataType
from gptcache.utils import (
    import_pillow, import_diffusers, import_huggingface
)
from gptcache.utils.error import CacheError

import_pillow()
import_huggingface()
import_diffusers()

from PIL import Image  # pylint: disable=C0413
import diffusers  # pylint: disable=C0413


class StableDiffusionPipeline(diffusers.StableDiffusionPipeline):
    """Diffuser StableDiffusionPipeline Wrapper

    Example:
        .. code-block:: python

            import torch

            from gptcache import cache
            from gptcache.processor.pre import get_prompt
            from gptcache.adapter.diffusers import StableDiffusionPipeline

            # init gptcache
            cache.init(pre_embedding_func=get_prompt)

            # run with gptcache
            model_id = "stabilityai/stable-diffusion-2-1"
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to("cuda")

            prompt = "a photo of an astronaut riding a horse on mars"
            image = pipe(prompt=prompt).images[0]
    """

    def _llm_handler(self, *llm_args, **llm_kwargs):
        try:
            return super().__call__(*llm_args, **llm_kwargs)
        except Exception as e:
            raise CacheError("diffuser error") from e

    def __call__(self, *args, **kwargs):
        def cache_data_convert(cache_data):
            return _construct_resp_from_cache(cache_data)

        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs):  # pylint: disable=unused-argument
            img = llm_data["images"][0]
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue())
            update_cache_func(Answer(img_b64, DataType.IMAGE_BASE64))
            return llm_data

        return adapt(
            self._llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs
        )


def _construct_resp_from_cache(img_64):
    im_bytes = base64.b64decode(img_64)   # im_bytes is a binary image
    im_file = BytesIO(im_bytes)  # convert image to file-like object
    img = Image.open(im_file)
    return diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput(images=[img], nsfw_content_detected=None)

