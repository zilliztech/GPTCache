import requests
from io import BytesIO
from gptcache.utils import import_pillow, import_vit

import_vit()
import_pillow()

from PIL import Image
from gptcache.embedding import ViT

def test_timm():
    url = 'https://raw.githubusercontent.com/zilliztech/GPTCache/main/docs/GPTCache.png'
    image_bytes = requests.get(url).content
    image_data = BytesIO(image_bytes)  # Convert image to file-like object
    image = Image.open(image_data)
    encoder = ViT(model="google/vit-base-patch16-384")
    embed = encoder.to_embeddings(image)
    assert len(embed) == encoder.dimension

if __name__ == "__main__":
    test_timm()