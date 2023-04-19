import time

import torch
from diffusers import DPMSolverMultistepScheduler

from gptcache.adapter.diffusers import StableDiffusionPipeline
from gptcache.processor.pre import get_prompt
from gptcache import cache

cache.init(
    pre_embedding_func=get_prompt,
)


model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
start = time.time()
image = pipe(prompt=prompt).images[0]
print(time.time() - start)