import os
import time

import openai
from langchain import PromptTemplate
from langchain.llms import OpenAI

from gptcache import Cache
from gptcache.adapter.api import init_similar_cache
from gptcache.adapter.langchain_models import LangChainLLMs
from gptcache.processor.pre import get_prompt

openai.api_key = os.getenv("OPENAI_API_KEY")

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = OpenAI()

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_cache = Cache()
init_similar_cache(pre_func=get_prompt, cache_obj=llm_cache)


before = time.time()
cached_llm = LangChainLLMs(llm=llm)
answer = cached_llm(prompt=question, cache_obj=llm_cache)
print(answer)
print("Read through Time Spent =", time.time() - before)

before = time.time()
question = "What is the winner Super Bowl in the year Justin Bieber was born?"
answer = cached_llm(prompt=question, cache_obj=llm_cache)
print(answer)
print("Cache Hit Time Spent =", time.time() - before)
