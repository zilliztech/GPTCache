import requests
from io import BytesIO
from gptcache.embedding import Timm

def test_timm():
    url = 'https://raw.githubusercontent.com/zilliztech/GPTCache/main/docs/GPTCache.png'
    image_bytes = requests.get(url).content
    image_file = BytesIO(image_bytes)  # Convert image to file-like object

    encoder = Timm(model='resnet50')
    embed = encoder.to_embeddings(image_file)
    assert len(embed) == encoder.dimension


if __name__ == "__main__":
    test_timm()