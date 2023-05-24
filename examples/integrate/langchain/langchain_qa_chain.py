import time

from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

from gptcache import cache
from gptcache.adapter.api import init_similar_cache
from gptcache.adapter.langchain_models import LangChainLLMs


def get_content_func(data, **_):
    return data.get("prompt").split("Question:")[-1]


init_similar_cache(pre_func=get_content_func)
cache.set_openai_key()

mkt_qa = load_qa_chain(llm=LangChainLLMs(llm=OpenAI(temperature=0)), chain_type="stuff")

msg = "What is Traditional marketing?"


before = time.time()
answer = mkt_qa.run(question=msg, input_documents=[Document(page_content="marketing is hello world")])
print(answer)
print("Time Spent:", time.time() - before)

before = time.time()
answer = mkt_qa.run(question=msg, input_documents=[Document(page_content="marketing is hello world")])
print(answer)
print("Time Spent:", time.time() - before)
