import time

import torch
from PIL import ImageChops
from diffusers import DPMSolverMultistepScheduler

from gptcache.adapter.diffusers import StableDiffusionPipeline
from gptcache.processor.pre import get_prompt
from gptcache import cache

from gptcache.embedding import Onnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.manager import get_data_manager, CacheBase, VectorBase, ObjectBase


# onnx = Onnx()
# cache_base = CacheBase('sqlite')
# vector_base = VectorBase('milvus', host='localhost', port='19530', collection_name='gptcache_image', dimension=onnx.dimension)
# object_base = ObjectBase('local', path='./images')
# data_manager = get_data_manager(cache_base, vector_base, object_base)

cache.init(
    pre_embedding_func=get_prompt,
    # embedding_func=onnx.to_embeddings,
    # data_manager=data_manager,
    # similarity_evaluation=SearchDistanceEvaluation(),
    )


model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
start = time.time()
image1 = pipe(prompt=prompt).images[0]
print("First time generation:", time.time() - start)

start = time.time()
image2 = pipe(prompt=prompt).images[0]
print("Second time generation:", time.time() - start)

# Compare generated images
diff = ImageChops.difference(image1, image2)
assert not diff.getbbox(), "Got different images."