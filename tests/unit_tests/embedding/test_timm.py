import requests
from PIL import Image
from gptcache.embedding import Timm

def test_timm():
    url = 'https://raw.githubusercontent.com/zilliztech/GPTCache/main/docs/GPTCache.png'
    image = Image.open(requests.get(url, stream=True).raw)  # Read image url as PIL.Image      

    encoder = Timm(model='resnet18')
    image_tensor = encoder.preprocess(image)
    embed = encoder.to_embeddings(image_tensor)
    assert len(embed) == encoder.dimension


if __name__ == "__main__":
    test_timm()