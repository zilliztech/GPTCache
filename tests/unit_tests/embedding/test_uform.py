from io import BytesIO

import requests

from gptcache.adapter.api import _get_model
from gptcache.utils import import_uform, import_pillow
from gptcache.utils.error import ParamError

import_uform()
import_pillow()


def test_uform():
    encoder = _get_model("uform")
    embed = encoder.to_embeddings("Hello, world.")
    assert len(embed) == encoder.dimension

    url = "https://raw.githubusercontent.com/zilliztech/GPTCache/main/docs/GPTCache.png"
    image_bytes = requests.get(url).content
    image_file = BytesIO(image_bytes)

    encoder = _get_model("uform", model_config={"embedding_type": "image"})
    embed = encoder.to_embeddings(image_file)
    assert len(embed) == encoder.dimension

    is_exception = False
    try:
        _get_model("uform", model_config={"embedding_type": "foo"})
    except ParamError:
        is_exception = True
    assert is_exception
