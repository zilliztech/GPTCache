import asyncio
import os
import random
from unittest.mock import patch

from gptcache import Cache, Config
from gptcache.adapter import openai
from gptcache.adapter.api import init_similar_cache, get
from gptcache.adapter.langchain_models import LangChainLLMs, LangChainChat, _cache_msg_data_convert
from gptcache.processor.pre import get_prompt, last_content_without_template, get_messages_last_content
from gptcache.utils import import_pydantic, import_langchain
from gptcache.utils.response import get_message_from_openai_answer

import_pydantic()
import_langchain()

from langchain import OpenAI, PromptTemplate
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
    llm = LangChainLLMs(llm=langchain_openai,cache_obj=llm_cache)
    assert str(langchain_openai) == str(llm)

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

        answer = llm(prompt=question)
        assert expect_answer == answer

    answer = llm(prompt=question)
    assert expect_answer == answer


def test_langchain_chats():
    question = [HumanMessage(content="test_langchain_chats")]
    question2 = [HumanMessage(content="test_langchain_chats2")]
    msg = "chat models"
    expect_answer = {
        "role": "assistant",
        "message": msg,
        "content": msg,
    }

    llm_cache = Cache()
    llm_cache.init(
        pre_embedding_func=get_messages_last_content,
    )

    os.environ["OPENAI_API_KEY"] = "API"
    langchain_openai = ChatOpenAI(temperature=0)
    chat = LangChainChat(chat=langchain_openai,cache_obj=llm_cache)

    assert chat.get_num_tokens("hello") == langchain_openai.get_num_tokens("hello")
    assert chat.get_num_tokens_from_messages(messages=[HumanMessage(content="test_langchain_chats")]) \
           == langchain_openai.get_num_tokens_from_messages(messages=[HumanMessage(content="test_langchain_chats")])

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

        answer = chat(messages=question)
        assert answer == _cache_msg_data_convert(msg).generations[0].message

    with patch("openai.ChatCompletion.acreate") as mock_create:
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

        answer = asyncio.run(chat.agenerate([question2]))
        assert answer.generations[0][0].text == _cache_msg_data_convert(msg).generations[0].text

    answer = chat(messages=question)
    assert answer == _cache_msg_data_convert(msg).generations[0].message

    answer = asyncio.run(chat.agenerate([question]))
    assert answer.generations[0][0].text == _cache_msg_data_convert(msg).generations[0].text

    answer = asyncio.run(chat.agenerate([question2]))
    assert answer.generations[0][0].text == _cache_msg_data_convert(msg).generations[0].text


def test_last_content_without_template():
    string_prompt = PromptTemplate.from_template("tell me a joke about {subject}")
    template = string_prompt.template
    cache_obj = Cache()
    data_dir = str(random.random())
    init_similar_cache(data_dir=data_dir, cache_obj=cache_obj, pre_func=last_content_without_template, config=Config(template=template))

    subject_str = "animal"
    expect_answer = "this is a joke"

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
                {"role": "user", "content": string_prompt.format(subject=subject_str)},
            ],
            cache_obj=cache_obj,
        )
        assert get_message_from_openai_answer(response) == expect_answer, response

    cache_obj.flush()

    init_similar_cache(data_dir=data_dir, cache_obj=cache_obj)

    cache_res = get(str([subject_str]), cache_obj=cache_obj)
    print(str([subject_str]))
    assert cache_res == expect_answer, cache_res
