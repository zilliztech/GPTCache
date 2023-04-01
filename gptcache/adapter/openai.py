from typing import Iterator

import openai

from .adapter import adapt


class ChatCompletion:
    @classmethod
    def create(cls, *args, **kwargs):
        def llm_handler(*llm_args, **llm_kwargs):
            return openai.ChatCompletion.create(*llm_args, **llm_kwargs)

        def cache_data_convert(cache_data):
            if kwargs.get("stream", False):
                return construct_stream_resp_from_cache(cache_data)
            return construct_resp_from_cache(cache_data)

        def update_cache_callback(llm_data, update_cache_func):
            if not isinstance(llm_data, Iterator):
                update_cache_func(get_message_from_openai_answer(llm_data))
                return llm_data
            else:
                def hook_openai_data(it):
                    total_answer = ""
                    for item in it:
                        total_answer += get_stream_message_from_openai_answer(item)
                        yield item
                    update_cache_func(total_answer)

                return hook_openai_data(llm_data)

        return adapt(llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs)


def construct_resp_from_cache(return_message):
    return {
        "gptcache": True,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": return_message
                },
                "finish_reason": "stop",
                "index": 0
            }
        ]
    }


def construct_stream_resp_from_cache(return_message):
    return [
        {
            "choices": [
                {
                    "delta": {
                        "role": "assistant"
                    },
                    "finish_reason": None,
                    "index": 0
                }
            ],
        },
        {
            "choices": [
                {
                    "delta": {
                        "content": return_message
                    },
                    "finish_reason": None,
                    "index": 0
                }
            ],
        },
        {
            "gptcache": True,
            "choices": [
                {
                  "delta": {},
                  "finish_reason": "stop",
                  "index": 0
                }
            ],
        }
    ]


def get_message_from_openai_answer(openai_data):
    return openai_data['choices'][0]['message']['content']


def get_stream_message_from_openai_answer(openai_data):
    return openai_data['choices'][0]['delta'].get('content', '')
