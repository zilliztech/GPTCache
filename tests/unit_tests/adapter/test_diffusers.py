from gptcache import cache
from gptcache.adapter import diffusers as cache_diffusers
from gptcache.processor.pre import get_prompt
from gptcache.utils import import_pillow, import_diffusers, import_torch, import_huggingface

import_pillow()
import_huggingface()
import_diffusers()
import_torch()

from PIL import ImageChops
import torch

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer


def get_dummy_components():
    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        block_out_channels=(32, 64),
        layers_per_block=2,
        sample_size=32,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32,
    )
    torch.manual_seed(0)
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    torch.manual_seed(0)
    vae = AutoencoderKL(
        block_out_channels=[32, 64],
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
        latent_channels=4,
    )
    torch.manual_seed(0)
    text_encoder_config = CLIPTextConfig(
        bos_token_id=0,
        eos_token_id=2,
        hidden_size=32,
        intermediate_size=37,
        layer_norm_eps=1e-05,
        num_attention_heads=4,
        num_hidden_layers=5,
        pad_token_id=1,
        vocab_size=1000,
    )
    text_encoder = CLIPTextModel(text_encoder_config)
    tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

    components = {
        "unet": unet,
        "scheduler": scheduler,
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "safety_checker": None,
        "feature_extractor": None,
    }
    return components


def test_stable_diffusion():
    cache.init(pre_embedding_func=get_prompt)
    prompt = "test"
    kwargs = get_dummy_components()
        
    pipe1 = cache_diffusers.StableDiffusionPipeline(**kwargs)
    origin_image = pipe1(prompt=prompt)["images"][0]

    pipe2 = cache_diffusers.StableDiffusionPipeline(**kwargs)
    return_image = pipe2(prompt=prompt)["images"][0]

    diff = ImageChops.difference(origin_image, return_image)
    assert not diff.getbbox()

