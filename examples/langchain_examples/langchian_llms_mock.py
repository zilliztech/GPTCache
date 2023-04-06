import os

from langchain import Cohere
from langchain.llms import OpenAI

from gptcache.adapter.langchain_llms import LangChainLLMs
from gptcache import cache, Cache
from gptcache.processor.post import nop as postnop
from gptcache.processor.pre import get_prompt


def run():
    data_file = "data_map.txt"
    has_data = os.path.isfile(data_file)
    llm_cache = Cache()
    llm_cache.init(
        pre_embedding_func=get_prompt,
        post_process_messages_func=postnop,
    )

    if not has_data:
        for i in range(10):
            question = f"foo{i}"
            answer = f"receiver the foo {i}"
            cache.data_manager.save(question, answer, cache.embedding_func(question))

    question = "foo0"

    os.environ["OPENAI_API_KEY"] = "API"
    langchain_openai = OpenAI(model_name="text-ada-001")
    llm = LangChainLLMs(langchain_openai)
    answer = llm(question, cache_obj=llm_cache)
    print(answer)
    answer = llm(question, cache_obj=llm_cache)
    print(answer)

    # TODO install cohere auto
    os.environ["COHERE_API_KEY"] = "API_KEY"
    langchain_cohere = Cohere()
    llm = LangChainLLMs(langchain_cohere)
    answer = llm(question, cache_obj=llm_cache)
    print(answer)


if __name__ == '__main__':
    run()
