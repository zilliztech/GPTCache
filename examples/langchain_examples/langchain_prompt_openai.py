import os

import openai
import time
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

from gptcache.adapter.langchain_models import LangChainLLMs
from gptcache import Cache
from gptcache.processor.pre import get_prompt

openai.api_key = os.getenv("OPENAI_API_KEY")

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = OpenAI()

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_cache = Cache()
llm_cache.init(
    pre_embedding_func=get_prompt,
)

before = time.time()
cached_llm = LangChainLLMs(llm=llm)
answer = cached_llm(prompt=question, cache_obj=llm_cache)
print(answer)
print("Read through Time Spent =", time.time() - before)

before = time.time()
answer = cached_llm(prompt=question, cache_obj=llm_cache)
print(answer)
print("Cache Hit Time Spent =", time.time() - before)
