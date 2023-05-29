import os
from unittest.mock import patch

from gptcache import Cache
from gptcache.adapter.langchain_models import LangChainLLMs, LangChainChat, _cache_msg_data_convert
from gptcache.processor.pre import get_prompt
from gptcache.utils import import_pydantic, import_langchain

import_pydantic()
import_langchain()

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


def test_langchain_llms():
    question = "test_langchain_llms"
    expect_answer = "hello"

    llm_cache = Cache()
    llm_cache.init(
        pre_embedding_func=get_prompt,
    )

    os.environ["OPENAI_API_KEY"] = "API"
    langchain_openai = OpenAI(model_name="text-ada-001")
    llm = LangChainLLMs(llm=langchain_openai)

    with patch("openai.Completion.create") as mock_create:
        mock_create.return_value = {
              "choices": [
                {
                  "finish_reason": "stop",
                  "index": 0,
                  "text": expect_answer,
                }
              ],
              "created": 1677825456,
              "id": "chatcmpl-6ptKqrhgRoVchm58Bby0UvJzq2ZuQ",
              "model": "gpt-3.5-turbo-0301",
              "object": "chat.completion",
              "usage": {
                "completion_tokens": 301,
                "prompt_tokens": 36,
                "total_tokens": 337
              }
            }

        answer = llm(prompt=question, cache_obj=llm_cache)
        assert expect_answer == answer

    answer = llm(prompt=question, cache_obj=llm_cache)
    assert expect_answer == answer


def get_msg_func(data, **_):
    return data.get("messages")[-1].content


def test_langchain_chats():
    question = [HumanMessage(content="test_langchain_chats")]
    msg = "chat models"
    expect_answer = {
        "role": "assistant",
        "message": msg,
        "content": msg,
    }

    llm_cache = Cache()
    llm_cache.init(
        pre_embedding_func=get_msg_func,
    )

    os.environ["OPENAI_API_KEY"] = "API"
    langchain_openai = ChatOpenAI(temperature=0)
    chat = LangChainChat(chat=langchain_openai)

    with patch("openai.ChatCompletion.create") as mock_create:
        mock_create.return_value = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": expect_answer,
                }
            ],
            "delta": {"role": "assistant"},
            "created": 1677825456,
            "id": "chatcmpl-6ptKqrhgRoVchm58Bby0UvJzq2ZuQ",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 301,
                "prompt_tokens": 36,
                "total_tokens": 337
            }
        }

        answer = chat(messages=question, cache_obj=llm_cache)
        assert answer == _cache_msg_data_convert(msg).generations[0].message

    answer = chat(messages=question, cache_obj=llm_cache)
    assert answer == _cache_msg_data_convert(msg).generations[0].message
