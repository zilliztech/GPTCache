from gptcache.utils.response import (
    get_message_from_openai_answer,
    get_stream_message_from_openai_answer,
)


def test_get_message_from_openai_answer():
    message = get_message_from_openai_answer(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": "hello", "role": "assistant"},
                }
            ],
            "created": 1677825456,
            "id": "chatcmpl-6ptKqrhgRoVchm58Bby0UvJzq2ZuQ",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 301,
                "prompt_tokens": 36,
                "total_tokens": 337,
            },
        }
    )
    assert message == "hello"


def test_get_stream_message_from_openai_answer():
    message = get_stream_message_from_openai_answer(
        {
            "choices": [
                {"delta": {"role": "assistant"}, "finish_reason": None, "index": 0}
            ],
            "created": 1677825464,
            "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion.chunk",
        }
    )
    assert message == ""

    message = get_stream_message_from_openai_answer(
        {
            "choices": [{"delta": {"content": "2"}, "finish_reason": None, "index": 0}],
            "created": 1677825464,
            "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion.chunk",
        }
    )
    assert message == "2"
