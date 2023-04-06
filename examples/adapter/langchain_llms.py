import os

from langchain import Cohere
from langchain.llms import OpenAI

from gptcache.adapter.langchain_llms import LangChainLLMs
from gptcache import cache
from gptcache.processor.post import nop as postnop
from gptcache.processor.pre import get_prompt


def run():
    cache.init(
        pre_embedding_func=get_prompt,
        post_process_messages_func=postnop,
    )

    question = 'what is chatgpt'

    os.environ['OPENAI_API_KEY'] = 'API'
    langchain_openai = OpenAI(model_name='text-ada-001')
    llm = LangChainLLMs(langchain_openai)
    answer = llm(question)
    print(answer)

    # TODO install cohere auto
    os.environ['COHERE_API_KEY'] = 'API_KEY'
    langchain_cohere = Cohere()
    llm = LangChainLLMs(langchain_cohere)
    answer = llm(question)
    print(answer)


if __name__ == '__main__':
    run()
