import asyncio
import base64
import os
import random
from io import BytesIO
from unittest.mock import AsyncMock, patch
from urllib.request import urlopen

import pytest

from gptcache import Cache, cache
from gptcache.adapter import openai
from gptcache.adapter.api import init_similar_cache
from gptcache.config import Config
from gptcache.manager import get_data_manager
from gptcache.processor.pre import (
    get_file_bytes,
    get_file_name,
    get_openai_moderation_input,
    get_prompt,
    last_content,
)
from gptcache.utils.error import CacheError
from gptcache.utils.response import (
    get_audio_text_from_openai_answer,
    get_image_from_openai_b64,
    get_image_from_openai_url,
    get_image_from_path,
    get_message_from_openai_answer,
    get_stream_message_from_openai_answer,
    get_text_from_openai_answer,
)

try:
    from PIL import Image
except ModuleNotFoundError:
    from gptcache.utils.dependency_control import prompt_install

    prompt_install("pillow")
    from PIL import Image


@pytest.mark.parametrize("enable_token_counter", (True, False))
def test_normal_openai(enable_token_counter):
    cache.init(config=Config(enable_token_counter=enable_token_counter))
    question = "calculate 1+3"
    expect_answer = "the result is 4"
    with patch("openai.ChatCompletion.create") as mock_create:
        datas = {
            "choices": [
                {
                    "message": {"content": expect_answer, "role": "assistant"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "created": 1677825464,
            "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion.chunk",
        }
        mock_create.return_value = datas

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
        )

        assert get_message_from_openai_answer(response) == expect_answer, response

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
    )
    answer_text = get_message_from_openai_answer(response)
    assert answer_text == expect_answer, answer_text


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_token_counter", (True, False))
async def test_normal_openai_async(enable_token_counter):
    cache.init(config=Config(enable_token_counter=enable_token_counter))
    question = "calculate 1+3"
    expect_answer = "the result is 4"
    import openai as real_openai

    with patch.object(
        real_openai.ChatCompletion, "acreate", new_callable=AsyncMock
    ) as mock_acreate:
        datas = {
            "choices": [
                {
                    "message": {"content": expect_answer, "role": "assistant"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "created": 1677825464,
            "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion.chunk",
        }
        mock_acreate.return_value = datas

        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
        )

        assert get_message_from_openai_answer(response) == expect_answer, response

    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
    )
    answer_text = get_message_from_openai_answer(response)
    assert answer_text == expect_answer, answer_text


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


@pytest.mark.asyncio
async def test_stream_openai_async():
    cache.init()
    question = "calculate 1+4"
    expect_answer = "the result is 5"
    import openai as real_openai

    with patch.object(
        real_openai.ChatCompletion, "acreate", new_callable=AsyncMock
    ) as mock_acreate:
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
                    {"delta": {"content": " is 5"}, "finish_reason": None, "index": 0}
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

        async def acreate(*args, **kwargs):
            for item in datas:
                yield item
                await asyncio.sleep(0)

        mock_acreate.return_value = acreate()

        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            stream=True,
        )
        all_text = ""
        async for res in response:
            all_text += get_stream_message_from_openai_answer(res)
        assert all_text == expect_answer, all_text

    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
        stream=True,
    )
    answer_text = ""
    async for res in response:
        answer_text += get_stream_message_from_openai_answer(res)
    assert answer_text == expect_answer, answer_text


def test_completion():
    cache.init(pre_embedding_func=get_prompt)
    question = "what is your name?"
    expect_answer = "gptcache"

    with patch("openai.Completion.create") as mock_create:
        mock_create.return_value = {
            "choices": [{"text": expect_answer, "finish_reason": None, "index": 0}],
            "created": 1677825464,
            "id": "cmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "text-davinci-003",
            "object": "text_completion",
        }

        response = openai.Completion.create(model="text-davinci-003", prompt=question)
        answer_text = get_text_from_openai_answer(response)
        assert answer_text == expect_answer

    response = openai.Completion.create(model="text-davinci-003", prompt=question)
    answer_text = get_text_from_openai_answer(response)
    assert answer_text == expect_answer


@pytest.mark.asyncio
async def test_completion_async():
    cache.init(pre_embedding_func=get_prompt)
    question = "what is your name?"
    expect_answer = "gptcache"

    with patch("openai.Completion.acreate", new_callable=AsyncMock) as mock_acreate:
        mock_acreate.return_value = {
            "choices": [{"text": expect_answer, "finish_reason": None, "index": 0}],
            "created": 1677825464,
            "id": "cmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "text-davinci-003",
            "object": "text_completion",
        }

        response = await openai.Completion.acreate(
            model="text-davinci-003", prompt=question
        )
        answer_text = get_text_from_openai_answer(response)
        assert answer_text == expect_answer

    response = await openai.Completion.acreate(
        model="text-davinci-003", prompt=question
    )
    answer_text = get_text_from_openai_answer(response)
    assert answer_text == expect_answer


@pytest.mark.asyncio
async def test_completion_error_wrapping():
    cache.init(pre_embedding_func=get_prompt)
    import openai as real_openai

    with patch("openai.Completion.acreate", new_callable=AsyncMock) as mock_acreate:
        mock_acreate.side_effect = real_openai.OpenAIError
        with pytest.raises(real_openai.OpenAIError) as e:
            await openai.Completion.acreate(model="text-davinci-003", prompt="boom")
        assert isinstance(e.value, CacheError)

    with patch("openai.Completion.create") as mock_create:
        mock_create.side_effect = real_openai.OpenAIError
        with pytest.raises(real_openai.OpenAIError) as e:
            openai.Completion.create(model="text-davinci-003", prompt="boom")
        assert isinstance(e.value, CacheError)


def test_image_create():
    cache.init(pre_embedding_func=get_prompt)
    prompt1 = "test url"  # bytes
    test_url = (
        "https://raw.githubusercontent.com/zilliztech/GPTCache/dev/docs/GPTCache.png"
    )
    test_response = {"created": 1677825464, "data": [{"url": test_url}]}
    prompt2 = "test base64"
    img_bytes = base64.b64decode(get_image_from_openai_url(test_response))
    img_file = BytesIO(img_bytes)  # convert image to file-like object
    img = Image.open(img_file)
    img = img.resize((256, 256))
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    expected_img_data = base64.b64encode(buffered.getvalue()).decode("ascii")

    ###### Return base64 ######
    with patch("openai.Image.create") as mock_create_b64:
        mock_create_b64.return_value = {
            "created": 1677825464,
            "data": [{"b64_json": expected_img_data}],
        }

        response = openai.Image.create(
            prompt=prompt1, size="256x256", response_format="b64_json"
        )
        img_returned = get_image_from_openai_b64(response)
        assert img_returned == expected_img_data

    response = openai.Image.create(
        prompt=prompt1, size="256x256", response_format="b64_json"
    )
    img_returned = get_image_from_openai_b64(response)
    assert img_returned == expected_img_data

    ###### Return url ######
    with patch("openai.Image.create") as mock_create_url:
        mock_create_url.return_value = {
            "created": 1677825464,
            "data": [{"url": test_url}],
        }

        response = openai.Image.create(
            prompt=prompt2, size="256x256", response_format="url"
        )
        answer_url = response["data"][0]["url"]
        assert test_url == answer_url

    response = openai.Image.create(
        prompt=prompt2, size="256x256", response_format="url"
    )
    img_returned = get_image_from_path(response).decode("ascii")
    assert img_returned == expected_img_data
    os.remove(response["data"][0]["url"])


def test_audio_transcribe():
    cache.init(pre_embedding_func=get_file_name)
    url = "https://github.com/towhee-io/examples/releases/download/data/blues.00000.mp3"
    audio_file = urlopen(url)
    audio_file.name = url
    expect_answer = (
        "One bourbon, one scotch and one bill Hey Mr. Bartender, come here I want another drink and I want it now My baby she gone, "
        "she been gone tonight I ain't seen my baby since night of her life One bourbon, one scotch and one bill"
    )

    with patch("openai.Audio.transcribe") as mock_create:
        mock_create.return_value = {"text": expect_answer}

        response = openai.Audio.transcribe(model="whisper-1", file=audio_file)
        answer_text = get_audio_text_from_openai_answer(response)
        assert answer_text == expect_answer

    response = openai.Audio.transcribe(model="whisper-1", file=audio_file)
    answer_text = get_audio_text_from_openai_answer(response)
    assert answer_text == expect_answer


def test_audio_translate():
    cache.init(
        pre_embedding_func=get_file_bytes,
        data_manager=get_data_manager(data_path="data_map1.txt"),
    )
    url = "https://github.com/towhee-io/examples/releases/download/data/blues.00000.mp3"
    audio_file = urlopen(url)
    audio_file.name = url
    expect_answer = (
        "One bourbon, one scotch and one bill Hey Mr. Bartender, come here I want another drink and I want it now My baby she gone, "
        "she been gone tonight I ain't seen my baby since night of her life One bourbon, one scotch and one bill"
    )

    with patch("openai.Audio.translate") as mock_create:
        mock_create.return_value = {"text": expect_answer}

        response = openai.Audio.translate(model="whisper-1", file=audio_file)
        answer_text = get_audio_text_from_openai_answer(response)
        assert answer_text == expect_answer

    audio_file.name = "download/data/blues.00000.mp3"
    response = openai.Audio.translate(model="whisper-1", file=audio_file)
    answer_text = get_audio_text_from_openai_answer(response)
    assert answer_text == expect_answer


def test_moderation():
    init_similar_cache(
        data_dir=str(random.random()), pre_func=get_openai_moderation_input
    )
    expect_violence = 0.8864422
    with patch("openai.Moderation.create") as mock_create:
        mock_create.return_value = {
            "id": "modr-7IxkwrKvfnNJJIBsXAc0mfcpGaQJF",
            "model": "text-moderation-004",
            "results": [
                {
                    "categories": {
                        "hate": False,
                        "hate/threatening": False,
                        "self-harm": False,
                        "sexual": False,
                        "sexual/minors": False,
                        "violence": True,
                        "violence/graphic": False,
                    },
                    "category_scores": {
                        "hate": 0.18067425,
                        "hate/threatening": 0.0032884814,
                        "self-harm": 1.8089558e-09,
                        "sexual": 9.759996e-07,
                        "sexual/minors": 1.3364182e-08,
                        "violence": 0.8864422,
                        "violence/graphic": 3.2011528e-08,
                    },
                    "flagged": True,
                }
            ],
        }
        response = openai.Moderation.create(
            input=["I want to kill them."],
        )
        assert (
            response.get("results")[0].get("category_scores").get("violence")
            == expect_violence
        )

    response = openai.Moderation.create(
        input="I want to kill them.",
    )
    assert (
        response.get("results")[0].get("category_scores").get("violence")
        == expect_violence
    )

    expect_violence = 0.88708615
    with patch("openai.Moderation.create") as mock_create:
        mock_create.return_value = {
            "id": "modr-7Ixe5Bvq4wqzZb1xtOxGxewg0G87F",
            "model": "text-moderation-004",
            "results": [
                {
                    "flagged": False,
                    "categories": {
                        "sexual": False,
                        "hate": False,
                        "violence": False,
                        "self-harm": False,
                        "sexual/minors": False,
                        "hate/threatening": False,
                        "violence/graphic": False,
                    },
                    "category_scores": {
                        "sexual": 1.5214279e-06,
                        "hate": 2.0188916e-06,
                        "violence": 1.8034231e-09,
                        "self-harm": 1.0547879e-10,
                        "sexual/minors": 2.6696927e-09,
                        "hate/threatening": 8.445262e-12,
                        "violence/graphic": 5.324232e-10,
                    },
                },
                {
                    "flagged": True,
                    "categories": {
                        "sexual": False,
                        "hate": False,
                        "violence": True,
                        "self-harm": False,
                        "sexual/minors": False,
                        "hate/threatening": False,
                        "violence/graphic": False,
                    },
                    "category_scores": {
                        "sexual": 9.5307604e-07,
                        "hate": 0.18386655,
                        "violence": 0.88708615,
                        "self-harm": 1.7594172e-09,
                        "sexual/minors": 1.3112497e-08,
                        "hate/threatening": 0.0032587533,
                        "violence/graphic": 3.1731048e-08,
                    },
                },
            ],
        }
        response = openai.Moderation.create(
            input=["hello, world", "I want to kill them."],
        )
        assert not response.get("results")[0].get("flagged")
        assert (
            response.get("results")[1].get("category_scores").get("violence")
            == expect_violence
        )

    response = openai.Moderation.create(
        input=["hello, world", "I want to kill them."],
    )
    assert not response.get("results")[0].get("flagged")
    assert (
        response.get("results")[1].get("category_scores").get("violence")
        == expect_violence
    )


def test_base_llm_cache():
    cache_obj = Cache()
    init_similar_cache(
        data_dir=str(random.random()), pre_func=last_content, cache_obj=cache_obj
    )
    question = "What's Github"
    expect_answer = "Github is a great place to start"

    with patch("openai.ChatCompletion.create") as mock_create:
        datas = {
            "choices": [
                {
                    "message": {"content": expect_answer, "role": "assistant"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "created": 1677825464,
            "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion.chunk",
        }
        mock_create.return_value = datas

        import openai as real_openai

        def proxy_openai_chat_complete_exception(*args, **kwargs):
            raise real_openai.error.APIConnectionError("connect fail")

        openai.ChatCompletion.llm = proxy_openai_chat_complete_exception

        is_openai_exception = False
        try:
            openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question},
                ],
                cache_obj=cache_obj,
            )
        except real_openai.error.APIConnectionError:
            is_openai_exception = True

        assert is_openai_exception

        is_proxy = False

        def proxy_openai_chat_complete(*args, **kwargs):
            nonlocal is_proxy
            is_proxy = True
            return real_openai.ChatCompletion.create(*args, **kwargs)

        openai.ChatCompletion.llm = proxy_openai_chat_complete

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            cache_obj=cache_obj,
        )
        assert is_proxy

        assert get_message_from_openai_answer(response) == expect_answer, response

    is_exception = False
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
        )
    except Exception:
        is_exception = True
    assert is_exception

    openai.ChatCompletion.cache_args = {"cache_obj": cache_obj}

    print(openai.ChatCompletion.fill_base_args(foo="hello"))

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
    )

    openai.ChatCompletion.llm = None
    openai.ChatCompletion.cache_args = {}
    assert get_message_from_openai_answer(response) == expect_answer, response


@pytest.mark.asyncio
async def test_base_llm_cache_async():
    cache_obj = Cache()
    init_similar_cache(
        data_dir=str(random.random()), pre_func=last_content, cache_obj=cache_obj
    )
    question = "What's Github"
    expect_answer = "Github is a great place to start"
    import openai as real_openai

    with patch.object(
        real_openai.ChatCompletion, "acreate", new_callable=AsyncMock
    ) as mock_acreate:
        datas = {
            "choices": [
                {
                    "message": {"content": expect_answer, "role": "assistant"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "created": 1677825464,
            "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion.chunk",
        }
        mock_acreate.return_value = datas

        async def proxy_openai_chat_complete_exception(*args, **kwargs):
            raise real_openai.error.APIConnectionError("connect fail")

        openai.ChatCompletion.llm = proxy_openai_chat_complete_exception

        is_openai_exception = False
        try:
            await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question},
                ],
                cache_obj=cache_obj,
            )
        except real_openai.error.APIConnectionError:
            is_openai_exception = True

        assert is_openai_exception

        is_proxy = False

        def proxy_openai_chat_complete(*args, **kwargs):
            nonlocal is_proxy
            is_proxy = True
            return real_openai.ChatCompletion.acreate(*args, **kwargs)

        openai.ChatCompletion.llm = proxy_openai_chat_complete

        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            cache_obj=cache_obj,
        )
        assert is_proxy

        assert get_message_from_openai_answer(response) == expect_answer, response

    is_exception = False
    try:
        resp = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
        )
    except Exception:
        is_exception = True
    assert is_exception

    openai.ChatCompletion.cache_args = {"cache_obj": cache_obj}

    print(openai.ChatCompletion.fill_base_args(foo="hello"))

    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
    )

    openai.ChatCompletion.llm = None
    openai.ChatCompletion.cache_args = {}
    assert get_message_from_openai_answer(response) == expect_answer, response


# def test_audio_api():
#     data2vec = Data2VecAudio()
#     data_manager = manager_factory("sqlite,faiss,local", "audio_api", vector_params={"dimension": data2vec.dimension})
#     cache.init(
#         pre_embedding_func=get_prompt,
#         embedding_func=data2vec.to_embeddings,
#         data_manager=data_manager,
#         similarity_evaluation=SearchDistanceEvaluation(),
#     )
#     # url = "https://github.com/towhee-io/examples/releases/download/data/blues.00000.mp3"
#     url = "https://github.com/towhee-io/examples/releases/download/data/ah_yes.wav"
#     expect_answer = (
#         "One bourbon, one scotch and one bill Hey Mr. Bartender, come here I want another drink and I want it now My baby she gone, "
#         "she been gone tonight I ain't seen my baby since night of her life One bourbon, one scotch and one bill"
#     )
#     put(prompt=url, data=expect_answer)
#
#     assert get(prompt=url) == expect_answer
