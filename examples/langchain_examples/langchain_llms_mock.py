import os

from langchain import Cohere
from langchain.llms import OpenAI

from gptcache.adapter.langchain_llms import LangChainLLMs
from gptcache import cache, Cache
from gptcache.processor.pre import get_prompt

OpenAI.api_key = os.getenv("OPENAI_API_KEY")
Cohere.cohere_api_key = os.getenv("COHERE_API_KEY")


def run():
    data_file = "data_map.txt"
    has_data = os.path.isfile(data_file)
    llm_cache = Cache()
    llm_cache.init(
        pre_embedding_func=get_prompt,
    )

    if not has_data:
        for i in range(10):
            question = f"foo{i}"
            answer = f"receiver the foo {i}"
            cache.data_manager.save(question, answer, cache.embedding_func(question))

    question = "foo0"

    langchain_openai = OpenAI(model_name="text-ada-001")
    llm = LangChainLLMs(llm=langchain_openai)
    answer = llm(prompt=question, cache_obj=llm_cache)
    print(answer)
    answer = llm(prompt=question, cache_obj=llm_cache)
    print(answer)

    # TODO install cohere auto
    langchain_cohere = Cohere()
    llm = LangChainLLMs(llm=langchain_cohere)
    answer = llm(prompt=question, cache_obj=llm_cache)
    print(answer)


if __name__ == '__main__':
    run()
