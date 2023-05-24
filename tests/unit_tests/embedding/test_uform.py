from io import BytesIO

import requests

from gptcache.embedding.uform import UForm
from gptcache.utils import import_uform, import_pillow
from gptcache.utils.error import ParamError

import_uform()
import_pillow()


def test_uform():
    encoder = UForm()
    embed = encoder.to_embeddings("Hello, world.")
    assert len(embed) == encoder.dimension

    url = 'https://raw.githubusercontent.com/zilliztech/GPTCache/main/docs/GPTCache.png'
    image_bytes = requests.get(url).content
    image_file = BytesIO(image_bytes)

    encoder = UForm(embedding_type="image")
    embed = encoder.to_embeddings(image_file)
    assert len(embed) == encoder.dimension

    is_exception = False
    try:
        UForm(embedding_type="foo")
    except ParamError:
        is_exception = True
    assert is_exception