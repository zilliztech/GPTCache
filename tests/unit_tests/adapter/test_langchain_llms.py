import os
from unittest.mock import patch

from gptcache import Cache
from gptcache.adapter.langchain_llms import LangChainLLMs
from gptcache.processor.pre import get_prompt
from gptcache.utils import import_pydantic, import_langchain

import_pydantic()
import_langchain()

from langchain import OpenAI


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
