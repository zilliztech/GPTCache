import os

from langchain import Cohere
from langchain.llms import OpenAI

from gptcache.adapter.langchain_llms import LangChainLLMs
from gptcache import cache
from gptcache.processor.pre import get_prompt

OpenAI.api_key = os.getenv("OPENAI_API_KEY")
Cohere.cohere_api_key = os.getenv("COHERE_API_KEY")


def run():
    cache.init(
        pre_embedding_func=get_prompt,
    )

    question = 'what is chatgpt'

    langchain_openai = OpenAI(model_name='text-ada-001')
    llm = LangChainLLMs(llm=langchain_openai)
    answer = llm(question)
    print(answer)

    # TODO install cohere auto
    langchain_cohere = Cohere()
    llm = LangChainLLMs(llm=langchain_cohere)
    answer = llm(question)
    print(answer)


if __name__ == '__main__':
    run()
