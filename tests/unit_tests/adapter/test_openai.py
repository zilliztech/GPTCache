from unittest.mock import patch
from gptcache.utils.response import (
    get_stream_message_from_openai_answer,
    get_message_from_openai_answer,
    get_text_from_openai_answer,
    get_image_from_openai_b64,
    get_image_from_path,
    get_image_from_openai_url,
    get_audio_text_from_openai_answer
)
from gptcache.adapter import openai
from gptcache import cache
from gptcache.manager import get_data_manager
from gptcache.processor.pre import get_prompt, get_file_bytes

import os
import base64
from urllib.request import urlopen
from io import BytesIO
try:
    from PIL import Image
except ModuleNotFoundError:
    from gptcache.utils.dependency_control import prompt_install
    prompt_install("pillow")
    from PIL import Image



def test_stream_openai():
    cache.init()
    question = "calculate 1+1"
    expect_answer = "the result is 2"

    with patch("openai.ChatCompletion.create") as mock_create:
        datas = [
            {
                "choices": [
                    {"delta": {"role": "assistant"}, "finish_reason": None, "index": 0}
                ],
                "created": 1677825464,
                "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
                "model": "gpt-3.5-turbo-0301",
                "object": "chat.completion.chunk",
            },
            {
                "choices": [
                    {
                        "delta": {"content": "the result"},
                        "finish_reason": None,
                        "index": 0,
                    }
                ],
                "created": 1677825464,
                "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
                "model": "gpt-3.5-turbo-0301",
                "object": "chat.completion.chunk",
            },
            {
                "choices": [
                    {"delta": {"content": " is 2"}, "finish_reason": None, "index": 0}
                ],
                "created": 1677825464,
                "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
                "model": "gpt-3.5-turbo-0301",
                "object": "chat.completion.chunk",
            },
            {
                "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                "created": 1677825464,
                "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
                "model": "gpt-3.5-turbo-0301",
                "object": "chat.completion.chunk",
            },
        ]
        mock_create.return_value = iter(datas)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            stream=True,
        )

        all_text = ""
        for res in response:
            all_text += get_stream_message_from_openai_answer(res)
        assert all_text == expect_answer, all_text

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
    )
    answer_text = get_message_from_openai_answer(response)
    assert answer_text == expect_answer, answer_text


def test_completion():
    cache.init(pre_embedding_func=get_prompt)
    question = "what is your name?"
    expect_answer = "gptcache"

    with patch("openai.Completion.create") as mock_create:
        mock_create.return_value = {
            "choices": [
                {"text": expect_answer,
                 "finish_reason": None,
                 "index": 0}
            ],
            "created": 1677825464,
            "id": "cmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "text-davinci-003",
            "object": "text_completion",
        }

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=question
        )
        answer_text = get_text_from_openai_answer(response)
        assert answer_text == expect_answer

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=question
    )
    answer_text = get_text_from_openai_answer(response)
    assert answer_text == expect_answer


def test_image_create():
    cache.init(pre_embedding_func=get_prompt)
    prompt1 = "test url"# bytes
    test_url = "https://raw.githubusercontent.com/zilliztech/GPTCache/dev/docs/GPTCache.png"
    test_response = {
        "created": 1677825464,
        "data": [
            {"url": test_url}
        ]
        }
    prompt2 = "test base64"
    img_bytes = base64.b64decode(get_image_from_openai_url(test_response))
    img_file = BytesIO(img_bytes)  # convert image to file-like object
    img = Image.open(img_file)
    img = img.resize((256, 256))
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    expected_img_data = base64.b64encode(buffered.getvalue())

    ###### Return base64 ######
    with patch("openai.Image.create") as mock_create_b64:
        mock_create_b64.return_value = {
            "created": 1677825464,
            "data": [
                {'b64_json': expected_img_data}
        ]
        } 

        response = openai.Image.create(
            prompt=prompt1,
            size="256x256",
            response_format="b64_json"
        )
        img_returned = get_image_from_openai_b64(response)
        assert img_returned == expected_img_data

    response = openai.Image.create(
            prompt=prompt1,
            size="256x256",
            response_format="b64_json"
        )
    img_returned = get_image_from_openai_b64(response)
    assert img_returned == expected_img_data

    ###### Return url ######
    with patch("openai.Image.create") as mock_create_url:
        mock_create_url.return_value = {
            "created": 1677825464,
            "data": [
                {'url': test_url}
        ]
        } 

        response = openai.Image.create(
            prompt=prompt2,
            size="256x256",
            response_format="url"
        )
        answer_url = response["data"][0]["url"]
        assert test_url == answer_url
    
    response = openai.Image.create(
            prompt=prompt2,
            size="256x256",
            response_format="url"
        )
    img_returned = get_image_from_path(response)
    assert img_returned == expected_img_data
    os.remove(response["data"][0]["url"])


def test_audio_transcribe():
    cache.init(pre_embedding_func=get_file_bytes)
    url = "https://github.com/towhee-io/examples/releases/download/data/blues.00000.mp3"
    audio_file = urlopen(url)
    expect_answer = "One bourbon, one scotch and one bill Hey Mr. Bartender, come here I want another drink and I want it now My baby she gone, " \
                    "she been gone tonight I ain't seen my baby since night of her life One bourbon, one scotch and one bill"

    with patch("openai.Audio.transcribe") as mock_create:
        mock_create.return_value = {
            "text": expect_answer
        }

        response = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file
        )
        answer_text = get_audio_text_from_openai_answer(response)
        assert answer_text == expect_answer

    response = openai.Audio.transcribe(
        model="whisper-1",
        file=audio_file
    )
    answer_text = get_audio_text_from_openai_answer(response)
    assert answer_text == expect_answer


def test_audio_translate():
    cache.init(pre_embedding_func=get_file_bytes,
               data_manager=get_data_manager(data_path="data_map1.txt"))
    url = "https://github.com/towhee-io/examples/releases/download/data/blues.00000.mp3"
    audio_file = urlopen(url)
    expect_answer = "One bourbon, one scotch and one bill Hey Mr. Bartender, come here I want another drink and I want it now My baby she gone, " \
                    "she been gone tonight I ain't seen my baby since night of her life One bourbon, one scotch and one bill"

    with patch("openai.Audio.translate") as mock_create:
        mock_create.return_value = {
            "text": expect_answer
        }

        response = openai.Audio.translate(
            model="whisper-1",
            file=audio_file
        )
        answer_text = get_audio_text_from_openai_answer(response)
        assert answer_text == expect_answer

    response = openai.Audio.translate(
        model="whisper-1",
        file=audio_file
    )
    answer_text = get_audio_text_from_openai_answer(response)
    assert answer_text == expect_answer
