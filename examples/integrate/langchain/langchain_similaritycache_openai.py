import os

import openai
import time
from langchain.llms import OpenAI
from langchain import PromptTemplate

from gptcache.adapter.langchain_models import LangChainLLMs
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache import Cache
from gptcache.embedding import Onnx
from gptcache.processor.pre import get_prompt
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

openai.api_key = os.getenv("OPENAI_API_KEY")

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = OpenAI()

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_cache = Cache()
onnx = Onnx()
cache_base = CacheBase('sqlite')
vector_base = VectorBase('faiss', dimension=onnx.dimension)
data_manager = get_data_manager(cache_base, vector_base, max_size=10, clean_size=2)
llm_cache.init(
    pre_embedding_func=get_prompt,
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
)


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
