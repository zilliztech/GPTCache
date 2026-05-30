import hashlib
import timeit

from gptcache import Cache
from gptcache.adapter.api import init_similar_cache
from gptcache.processor.pre import get_prompt
from gptcache.manager.factory import manager_factory
from langchain.globals import set_llm_cache
from langchain_community.cache import GPTCache

from langchain.globals import set_llm_cache
from langchain_openai import OpenAI

# def init_gptcache(cache_obj: Cache, llm: str):
#     cache_obj.init(
#         pre_embedding_func=get_prompt,
#         data_manager=manager_factory(
#             manager="map",
#             data_dir=f"map_cache_{llm}"
#         ),
#     )

def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()


def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    init_similar_cache(cache_obj=cache_obj, data_dir=f"similar_cache_{hashed_llm}")

set_llm_cache(GPTCache(init_gptcache))

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", n=2, best_of=2)

execution_time = timeit.timeit(lambda: llm("Tell me a joke"), number=1)
print(f"Execution time: {execution_time} seconds")

execution_time = timeit.timeit(lambda: llm("Tell me a joke"), number=1)
print(f"Execution time: {execution_time} seconds")

execution_time = timeit.timeit(lambda: llm("Tell me joke"), number=1)
print(f"Execution time: {execution_time} seconds")