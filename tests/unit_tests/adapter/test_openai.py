from unittest.mock import patch
from gptcache.utils.response import (
    get_stream_message_from_openai_answer,
    get_message_from_openai_answer,
    get_text_from_openai_answer,
)
from gptcache.adapter import openai
from gptcache import cache
from gptcache.processor.pre import get_prompt


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
