import base64
from io import BytesIO, BufferedReader
from unittest.mock import patch

from gptcache.utils import import_replicate, import_pillow

import_replicate()
import_pillow()

import replicate
from PIL import Image

from gptcache import cache
from gptcache.adapter import replicate
from gptcache.processor.pre import get_input_str
from gptcache.utils.response import get_image_from_openai_url


def test_run():
    test_response = {"data": [{"url": "https://replicate.delivery/pbxt/IJEPmgAlL2zNBNDoRRKFegTEcxnlRhoQxlNjPHSZEy0pSIKn/gg_bridge.jpeg"}]}
    img_bytes = base64.b64decode(get_image_from_openai_url(test_response))
    img_file = BytesIO(img_bytes)  # convert image to file-like object
    img = Image.open(img_file)

    b_handle = BytesIO()
    img.save(b_handle, format="JPEG")
    b_handle.seek(0)
    b_handle.name = "gg_bridge.jpeg"
    expected_img_data = BufferedReader(b_handle)

    expect_answer = "san francisco"

    cache.init(pre_embedding_func=get_input_str)

    with patch("replicate.run") as mock_create:
        mock_create.return_value = expect_answer

        answer_text = replicate.run(
                        "andreasjansson/blip-2:4b32258c42e9efd428",
                        input={"image": expected_img_data,
                               "question": "Which city is this photo taken on?"}
                    )
        assert answer_text == expect_answer

    answer_text = replicate.run(
                    "andreasjansson/blip-2:4b32258c42e9efd4288bb9910",
                    input={"image": expected_img_data,
                           "question": "Which city is this photo taken on?"}
                )
    assert answer_text == expect_answer
